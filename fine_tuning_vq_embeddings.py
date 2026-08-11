"""Train mT5 on cached SOKE body/left/right VQ embeddings.

This entrypoint is independent of the original pose/ST-GCN and online KL-VAE
paths.  Train and evaluation use the same JSON-syntax ``.cfg`` file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timm.optim import create_optimizer
from transformers import get_scheduler

import utils
from datasets_vq_embeddings import (
    ThreePartVQEmbeddingDataset,
    VQ_CODEBOOK_SIZES,
    VQ_EMBEDDING_DIM,
    VQ_NUM_PARTS,
)
from models_vq_embeddings import Uni_Sign_VQ_Embedding
from SLRT_metrics import translation_performance


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "how2sign_youtube3d_vq_embeddings.cfg"


def _resolve_path(value: str, config_path: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def load_runtime_config(path: Union[str, Path]) -> tuple[Path, dict]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"VQ embedding config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for section in ("dataset", "adapter", "model", "training", "evaluation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"config requires an object section named {section!r}")
    return config_path, config


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre_parser.parse_known_args()
    config_path, config = load_runtime_config(known.config)

    parser = argparse.ArgumentParser(
        "Uni-Sign cached-VQ-embedding fine-tuning",
        parents=[utils.get_args_parser()],
    )
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument(
        "--validate-cache",
        action="store_true",
        help="Validate configured splits, manifest, annotations, and embeddings, then exit.",
    )
    parser.set_defaults(**config["training"])
    args = parser.parse_args()

    config_path, config = load_runtime_config(args.config)
    dataset = config["dataset"]
    adapter = config["adapter"]
    model = config["model"]
    evaluation = config["evaluation"]

    if dataset.get("name") != "How2Sign_YouTube3D_VQ":
        raise ValueError("dataset.name must be 'How2Sign_YouTube3D_VQ'")
    if int(dataset.get("num_parts", -1)) != VQ_NUM_PARTS:
        raise ValueError(f"dataset.num_parts must be {VQ_NUM_PARTS}")
    if int(dataset.get("embedding_dim", -1)) != VQ_EMBEDDING_DIM:
        raise ValueError(f"dataset.embedding_dim must be {VQ_EMBEDDING_DIM}")
    codebook_sizes = tuple(int(value) for value in dataset.get("codebook_sizes", []))
    if codebook_sizes != VQ_CODEBOOK_SIZES:
        raise ValueError(f"dataset.codebook_sizes must be {VQ_CODEBOOK_SIZES}")
    if args.task != "SLT":
        raise ValueError("the VQ embedding entrypoint supports --task SLT only")
    if args.llm_type != "mt5":
        raise ValueError("the VQ embedding entrypoint currently supports mT5 only")
    if args.rgb_support:
        raise ValueError("--rgb_support is incompatible with cached VQ embeddings")

    annotation_roots = dataset.get("annotation_roots", {})
    if not isinstance(annotation_roots, dict):
        raise ValueError("dataset.annotation_roots must be an object")
    args.config_path = str(config_path)
    args.runtime_config = config
    args.vq_embedding_root = _resolve_path(dataset["embedding_root"], config_path)
    args.vq_split_index = _resolve_path(dataset["split_index"], config_path)
    args.vq_token_manifest = _resolve_path(dataset["token_manifest"], config_path)
    args.vq_checkpoint = _resolve_path(dataset["vq_checkpoint"], config_path)
    args.vq_annotation_roots = {
        source: _resolve_path(path, config_path)
        for source, path in annotation_roots.items()
    }
    args.vq_train_sources = tuple(dataset.get("train_sources", []))
    args.vq_eval_sources = tuple(dataset.get("evaluation_sources", []))
    args.vq_train_splits = tuple(dataset.get("train_splits", ["train"]))
    args.vq_evaluation_split = str(dataset.get("evaluation_split", "test"))
    args.vq_max_embedding_steps = int(dataset.get("max_embedding_steps", 0))
    args.vq_part_hidden_dim = int(adapter.get("part_hidden_dim", 256))
    args.vq_adapter_dropout = float(adapter.get("dropout", 0.0))
    args.mt5_path = _resolve_path(model["mt5_path"], config_path)
    args.prompt = str(model["prompt"])
    args.max_target_length = int(model.get("max_target_length", 50))
    args.max_new_tokens = int(evaluation.get("max_new_tokens", 100))
    args.num_beams = int(evaluation.get("num_beams", 4))
    args.eval_batch_size = int(evaluation.get("batch_size", args.batch_size))
    args.output_dir = _resolve_path(args.output_dir, config_path)
    return args


def build_dataset(
    args: argparse.Namespace,
    split: str,
) -> ThreePartVQEmbeddingDataset:
    sources = args.vq_train_sources if split == "train" else args.vq_eval_sources
    selected_splits = (
        args.vq_train_splits
        if split == "train"
        else (args.vq_evaluation_split,)
    )
    return ThreePartVQEmbeddingDataset(
        embedding_root=args.vq_embedding_root,
        annotation_roots=args.vq_annotation_roots,
        sources=sources,
        split=selected_splits,
        split_index_path=args.vq_split_index,
        token_manifest_path=args.vq_token_manifest,
        vq_checkpoint=args.vq_checkpoint,
        max_embedding_steps=args.vq_max_embedding_steps,
    )


def move_src_input_to_device(src_input: dict[str, object], model) -> None:
    target_dtype = torch.float32
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model.fp16_enabled():
        target_dtype = torch.float16
    for key, value in src_input.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            value = value.to(dtype=target_dtype)
        src_input[key] = value.cuda(non_blocking=True)


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    """DeepSpeed training loop independent of the pose/ST-GCN entrypoint."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}/{args.epochs}]"
    optimizer.zero_grad()

    for step, (src_input, tgt_input) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        move_src_input_to_device(src_input, model)
        stack_out = model(src_input, tgt_input)
        loss = stack_out["loss"]
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            sample_names = src_input.get("name_batch", [])
            print(
                f"Loss is {loss_value}, stopping before backward/step. "
                f"step={step}, samples={sample_names}"
            )
            sys.exit(1)
        model.backward(loss)
        model.step()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        key: meter.global_avg for key, meter in metric_logger.meters.items()
    }


def get_trainable_state_dict(model) -> dict[str, torch.Tensor]:
    """Save trainable parameters and non-parameter buffers without pose helpers."""
    parameters = dict(model.named_parameters())
    return {
        name: value
        for name, value in model.state_dict().items()
        if name not in parameters or parameters[name].requires_grad
    }


def evaluate(
    args,
    data_loader,
    model,
    model_without_ddp,
    phase: str,
) -> dict[str, float]:
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    predictions: list[torch.Tensor] = []
    references: list[str] = []
    names: list[str] = []

    with torch.no_grad():
        for src_input, tgt_input in metric_logger.log_every(
            data_loader, 10, f"{phase}:"
        ):
            move_src_input_to_device(src_input, model)
            # Evaluation runs only on rank 0. Bypass the DDP/DeepSpeed wrapper
            # so its forward does not wait for collectives from idle ranks.
            stack_out = model_without_ddp(src_input, tgt_input)
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
        with (output_dir / f"{phase}_tmp_pres.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            for name, prediction in zip(names, decoded):
                handle.write(f"sample: {name}, prediction: {prediction}\n")
        with (output_dir / f"{phase}_tmp_refs.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            for name, reference in zip(names, references):
                handle.write(f"sample: {name}, ground-truth: {reference}\n")
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def checkpoint_payload(args, model_without_ddp) -> dict[str, object]:
    return {
        "model": get_trainable_state_dict(model_without_ddp),
        "vq_provenance": {
            "embedding_root": args.vq_embedding_root,
            "split_index": args.vq_split_index,
            "token_manifest": args.vq_token_manifest,
            "vq_checkpoint": args.vq_checkpoint,
            "num_parts": VQ_NUM_PARTS,
            "embedding_dim": VQ_EMBEDDING_DIM,
            "codebook_sizes": VQ_CODEBOOK_SIZES,
        },
        "config": args.runtime_config,
    }


def main(args: argparse.Namespace) -> None:
    utils.init_distributed_mode_ds(args)
    utils.set_seed(args.seed)
    print(args)

    print("Creating cached three-part VQ datasets:")
    train_data = build_dataset(args, "train")
    eval_data = build_dataset(args, "test")
    print(train_data)
    print(eval_data)
    print(
        "WARNING: the supplied cache has train/test only; best checkpoint "
        "selection therefore uses the configured test sources."
    )

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
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(eval_data),
        pin_memory=args.pin_mem,
    )

    print("Creating three-part-VQ -> mT5 model:")
    model = Uni_Sign_VQ_Embedding(args=args)
    model.cuda()
    model.train()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    if args.finetune:
        print(f"Loading VQ-embedding Uni-Sign checkpoint: {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        result = model.load_state_dict(checkpoint["model"], strict=True)
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
    updates_per_epoch = len(train_loader) / args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * updates_per_epoch),
        num_training_steps=int(args.epochs * updates_per_epoch),
    )
    model, optimizer, lr_scheduler = utils.init_deepspeed(
        args, model, optimizer, lr_scheduler
    )
    model_without_ddp = model.module.module
    print(optimizer)

    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(args.runtime_config, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.eval:
        if utils.is_main_process():
            print("Evaluation result")
            evaluate(args, eval_loader, model, model_without_ddp, "test")
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
            checkpoint_name = (
                "last_checkpoint.pth"
                if args.save_last_only
                else f"checkpoint_{epoch}.pth"
            )
            utils.save_on_master(
                checkpoint_payload(args, model_without_ddp),
                output_dir / checkpoint_name,
            )

        if utils.is_main_process():
            eval_stats = evaluate(
                args, eval_loader, model, model_without_ddp, "test"
            )
            if eval_stats["bleu4"] > best_bleu4:
                best_bleu4 = eval_stats["bleu4"]
                if args.output_dir:
                    utils.save_on_master(
                        checkpoint_payload(args, model_without_ddp),
                        output_dir / "best_checkpoint.pth",
                    )
            print(f"BLEU-4 on {len(eval_data)} test samples: {eval_stats['bleu4']:.2f}")
            print(f"Max test BLEU-4: {best_bleu4:.2f}")

            log_stats = {
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"test_{key}": value for key, value in eval_stats.items()},
                "epoch": epoch,
            }
            with (output_dir / "log.txt").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            torch.distributed.barrier()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parsed_args = parse_args()
    if parsed_args.validate_cache:
        print(build_dataset(parsed_args, "train"))
        print(build_dataset(parsed_args, "test"))
        print("VQ cache validation passed")
        sys.exit(0)
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
