"""Additive Uni-Sign fine-tuning entrypoint for trainable VAE motion encoding.

The original ``fine_tuning.py`` remains the pose/ST-GCN entrypoint.  This file
keeps the same mT5 objective, optimizer/DeepSpeed infrastructure, BLEU-based
checkpoint selection, and generation behavior while swapping only:

1. the dataset: canonical online ``[T,133]`` How2Sign motion;
2. the encoder: trainable pretrained KL-VAE encoder instead of ST-GCN.

Run from the Uni-Sign repository root:

    deepspeed fine_tuning_vae.py --config configs/how2sign_vae_encoder.cfg
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timm.optim import create_optimizer
from transformers import get_scheduler

import utils
from datasets_vae import How2SignVaeDataset
from fine_tuning import train_one_epoch
from models import get_requires_grad_dict
from models_vae import Uni_Sign_VAE
from SLRT_metrics import translation_performance
from vae_encoder import DOWNSAMPLE_FACTOR, FEATURE_DIM, LATENT_DIM


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "how2sign_vae_encoder.cfg"


def _resolve_path(value: str, config_path: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Runtime paths follow the original Uni-Sign convention: relative to
        # the repository root, not relative to the config directory.
        path = REPO_ROOT / path
    return str(path.resolve())


def load_runtime_config(path: Union[str, Path]) -> tuple[Path, dict]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE encoder config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for section in ("dataset", "encoder", "model", "training", "evaluation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"config requires an object section named {section!r}")
    return config_path, config


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre_parser.parse_known_args()
    config_path, config = load_runtime_config(known.config)

    parser = argparse.ArgumentParser(
        "Uni-Sign trainable-VAE fine-tuning",
        parents=[utils.get_args_parser()],
    )
    parser.add_argument("--config", default=str(config_path))
    parser.set_defaults(**config["training"])
    args = parser.parse_args()

    config_path, config = load_runtime_config(args.config)
    dataset = config["dataset"]
    encoder = config["encoder"]
    model = config["model"]
    evaluation = config["evaluation"]

    if dataset.get("name") != "How2Sign":
        raise ValueError("the additive VAE path currently supports dataset.name=How2Sign")
    if int(dataset.get("nfeats", -1)) != FEATURE_DIM:
        raise ValueError(f"dataset.nfeats must be {FEATURE_DIM}")
    if int(dataset.get("unit_length", -1)) != DOWNSAMPLE_FACTOR:
        raise ValueError(f"dataset.unit_length must be {DOWNSAMPLE_FACTOR}")
    if int(encoder.get("latent_dim", -1)) != LATENT_DIM:
        raise ValueError(f"encoder.latent_dim must be {LATENT_DIM}")
    if int(encoder.get("downsample_factor", -1)) != DOWNSAMPLE_FACTOR:
        raise ValueError(f"encoder.downsample_factor must be {DOWNSAMPLE_FACTOR}")
    if encoder.get("trainable") is not True:
        raise ValueError("encoder.trainable must be true")
    if encoder.get("use_posterior") != "mean":
        raise ValueError("encoder.use_posterior must be 'mean'")
    if args.dataset != "How2Sign" or args.task != "SLT":
        raise ValueError("this entrypoint requires --dataset How2Sign --task SLT")
    if args.rgb_support:
        raise ValueError("--rgb_support belongs to the original ST-GCN path")

    args.config_path = str(config_path)
    args.motion_root = _resolve_path(str(dataset["root"]), config_path)
    args.vae_mean_path = _resolve_path(str(dataset["mean_path"]), config_path)
    args.vae_std_path = _resolve_path(str(dataset["std_path"]), config_path)
    args.motion_max_length = int(dataset.get("max_motion_length", 400))
    args.motion_unit_length = int(dataset["unit_length"])
    args.vae_checkpoint = _resolve_path(str(encoder["checkpoint"]), config_path)
    args.mt5_path = _resolve_path(str(model["mt5_path"]), config_path)
    args.prompt = str(model["prompt"])
    args.max_target_length = int(model.get("max_target_length", 50))
    args.max_new_tokens = int(evaluation.get("max_new_tokens", 100))
    args.num_beams = int(evaluation.get("num_beams", 4))
    return args


def build_dataset(args: argparse.Namespace, split: str) -> How2SignVaeDataset:
    return How2SignVaeDataset(
        root=args.motion_root,
        split=split,
        mean_path=args.vae_mean_path,
        std_path=args.vae_std_path,
        max_motion_length=args.motion_max_length,
        unit_length=args.motion_unit_length,
    )


def evaluate(
    args,
    data_loader,
    model,
    model_without_ddp,
    phase: str,
) -> dict[str, float]:
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None

    predictions: list[torch.Tensor] = []
    references: list[str] = []
    names: list[str] = []
    with torch.no_grad():
        for src_input, tgt_input in metric_logger.log_every(
            data_loader, 10, f"{phase}:"
        ):
            if target_dtype is not None:
                for key, value in src_input.items():
                    if isinstance(value, torch.Tensor):
                        src_input[key] = value.to(target_dtype).cuda()
            stack_out = model(src_input, tgt_input)
            metric_logger.update(loss=stack_out["loss"].item())
            output = model_without_ddp.generate(
                stack_out,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions.extend(output)
            references.extend(tgt_input["gt_sentence"])
            names.extend(src_input["name_batch"])

    if not predictions:
        raise ValueError(f"{phase} loader produced no predictions")
    tokenizer = model_without_ddp.mt5_tokenizer
    prediction_ids = pad_sequence(
        predictions,
        batch_first=True,
        padding_value=tokenizer.eos_token_id,
    )
    decoded = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
    bleu, rouge = translation_performance(references, decoded)
    for key, value in bleu.items():
        metric_logger.meters[key].update(value)
    metric_logger.meters["rouge"].update(rouge)

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"{phase}_tmp_pres.txt").open("w", encoding="utf-8") as handle:
            for name, prediction in zip(names, decoded):
                handle.write(f"sample: {name}, prediction: {prediction}\n")
        with (output_dir / f"{phase}_tmp_refs.txt").open("w", encoding="utf-8") as handle:
            for name, reference in zip(names, references):
                handle.write(f"sample: {name}, ground-truth: {reference}\n")
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def main(args: argparse.Namespace) -> None:
    utils.init_distributed_mode_ds(args)
    utils.set_seed(args.seed)
    print(args)

    print("Creating 133D How2Sign datasets:")
    train_data = build_dataset(args, "train")
    val_data = build_dataset(args, "val")
    test_data = build_dataset(args, "test")
    print(train_data)
    print(val_data)
    print(test_data)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, shuffle=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=val_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(val_data),
        pin_memory=args.pin_mem,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(test_data),
        pin_memory=args.pin_mem,
    )

    print("Creating Uni-Sign VAE model:")
    model = Uni_Sign_VAE(args=args)
    model.cuda()
    model.train()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    if args.finetune:
        print(f"Loading full VAE-encoder Uni-Sign checkpoint: {args.finetune}")
        state_dict = torch.load(args.finetune, map_location="cpu")["model"]
        result = model.load_state_dict(state_dict, strict=True)
        print("Missing keys:\n", "\n".join(result.missing_keys))
        print("Unexpected keys:\n", "\n".join(result.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    print(f"number of params: {utils.count_parameters_in_MB(model_without_ddp)}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(
            args.warmup_epochs
            * len(train_loader)
            / args.gradient_accumulation_steps
        ),
        num_training_steps=int(
            args.epochs * len(train_loader) / args.gradient_accumulation_steps
        ),
    )
    model, optimizer, lr_scheduler = utils.init_deepspeed(
        args, model, optimizer, lr_scheduler
    )
    model_without_ddp = model.module.module
    print(optimizer)

    output_dir = Path(args.output_dir)
    if args.eval:
        if utils.is_main_process():
            print("Validation result")
            evaluate(args, val_loader, model, model_without_ddp, "val")
            print("Test result")
            evaluate(args, test_loader, model, model_without_ddp, "test")
        return

    start_time = time.time()
    best_bleu4 = float("-inf")
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, train_loader, optimizer, epoch
        )

        if args.output_dir:
            utils.save_on_master(
                {"model": get_requires_grad_dict(model_without_ddp)},
                output_dir / f"checkpoint_{epoch}.pth",
            )

        test_stats: dict[str, float] = {}
        if utils.is_main_process():
            val_stats = evaluate(
                args, val_loader, model, model_without_ddp, "val"
            )
            test_stats = evaluate(
                args, test_loader, model, model_without_ddp, "test"
            )
            if val_stats["bleu4"] > best_bleu4:
                best_bleu4 = val_stats["bleu4"]
                if args.output_dir:
                    utils.save_on_master(
                        {"model": get_requires_grad_dict(model_without_ddp)},
                        output_dir / "best_checkpoint.pth",
                    )
            print(
                f"BLEU-4 on {len(val_data)} validation samples: "
                f"{val_stats['bleu4']:.2f}"
            )
            print(f"Max validation BLEU-4: {best_bleu4:.2f}")

            log_stats = {
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
                "epoch": epoch,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "log.txt").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_stats) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parsed_args = parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
