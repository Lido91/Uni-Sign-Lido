"""Fine-tuning entrypoint where the KL-VAE fully replaces the ST-GCN encoder.

Same data pipeline as the original path (``datasets.S2T_Dataset`` over the
native How2Sign/OpenASL RTMPose pickles), same mT5 objective, DeepSpeed
infrastructure, BLEU-4 checkpoint selection, and generation — the ONLY change
is the pose encoder: the joint How2Sign+OpenASL pose2d KL-VAE — permanently
frozen — (``models_pose2d_klvae.Uni_Sign_Pose2D_KLVAE``) followed by one
trainable linear head into the mT5 embedding space.

Train (OpenASL pretrain, then How2Sign finetune — see
script/train_pose2d_klvae.sh):

    deepspeed fine_tuning_pose2d_klvae.py \
        --dataset OpenASL --task SLT --dtype fp32 \
        --vae_checkpoint <MoMADiff kl_vae_h2s_openasl_pose2d .ckpt> \
        --output_dir out/pose2d_klvae_openasl

Evaluate:

    deepspeed fine_tuning_pose2d_klvae.py ... --eval \
        --finetune out/pose2d_klvae_openasl/best_checkpoint.pth
"""

import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from timm.optim import create_optimizer
from transformers import get_scheduler

import utils
from config import dev_label_paths, test_label_paths, train_label_paths
from datasets_pose2d_klvae import (
    FRAME_SAMPLING_MODES,
    S2T_Dataset_Pose2D_KLVAE,
)
from fine_tuning import evaluate, train_one_epoch
from models import get_requires_grad_dict
from models_pose2d_klvae import Uni_Sign_Pose2D_KLVAE

DEFAULT_VAE_META = (
    "/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/"
    "kl_vae_h2s_openasl_pose2d/meta"
)


def get_pose2d_klvae_args_parser():
    parser = argparse.ArgumentParser(
        "Uni-Sign pose2d KL-VAE scripts",
        parents=[utils.get_args_parser()],
    )
    parser.add_argument(
        "--vae_checkpoint",
        required=True,
        help="Trained MoMADiff kl_vae_h2s_openasl_pose2d checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--vae_mean_path",
        default=f"{DEFAULT_VAE_META}/mean.npy",
        help="(138,) train-set mean persisted by the KL-VAE run.",
    )
    parser.add_argument(
        "--vae_std_path",
        default=f"{DEFAULT_VAE_META}/std.npy",
        help="(138,) train-set std persisted by the KL-VAE run.",
    )
    parser.add_argument(
        "--frame_sampling",
        default="vae_linear",
        choices=list(FRAME_SAMPLING_MODES),
        help="vae_linear (default): deterministic linear-index resampling "
             "into [40, max_length], matching the KL-VAE corpus; "
             "unisign_random: the original random subsampling (ablation).",
    )
    return parser


def validate_args(args):
    if args.dataset not in ("How2Sign", "OpenASL"):
        raise ValueError(
            "the pose2d KL-VAE was trained on How2Sign+OpenASL; "
            "--dataset must be How2Sign or OpenASL"
        )
    if args.task != "SLT":
        raise ValueError("this entrypoint supports --task SLT only")
    if args.rgb_support:
        raise ValueError("--rgb_support belongs to the original ST-GCN path")
    if args.llm_type != "mt5":
        raise ValueError("the KL-VAE head projects into the mT5 (T5) space; "
                         "--llm_type must be mt5")


def main(args):
    utils.init_distributed_mode_ds(args)
    print(args)
    utils.set_seed(args.seed)

    print("Creating dataset:")
    train_data = S2T_Dataset_Pose2D_KLVAE(
        path=train_label_paths[args.dataset], args=args, phase="train"
    )
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, shuffle=True
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    eval_args = copy.copy(args)
    if args.eval_on_how2sign_test:
        eval_args.dataset = "How2Sign"
        print(
            "WARNING: evaluating directly on How2Sign test; "
            "best_checkpoint.pth will be selected using How2Sign test metrics."
        )

    test_data = S2T_Dataset_Pose2D_KLVAE(
        path=test_label_paths[eval_args.dataset], args=eval_args, phase="test"
    )
    print(test_data)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(test_data),
        pin_memory=args.pin_mem,
    )

    has_distinct_dev = "How2Sign" not in eval_args.dataset
    if has_distinct_dev:
        dev_data = S2T_Dataset_Pose2D_KLVAE(
            path=dev_label_paths[eval_args.dataset], args=eval_args, phase="dev"
        )
        print(dev_data)
        dev_dataloader = DataLoader(
            dev_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dev_data.collate_fn,
            sampler=torch.utils.data.SequentialSampler(dev_data),
            pin_memory=args.pin_mem,
        )
    else:
        dev_dataloader = test_dataloader
    selection_phase = "dev" if has_distinct_dev else "test"

    print("Creating pose2d KL-VAE model:")
    model = Uni_Sign_Pose2D_KLVAE(args=args)
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != "":
        print("***********************************")
        print("Load Checkpoint...")
        print("***********************************")
        state_dict = torch.load(args.finetune, map_location="cpu")["model"]
        # checkpoints of this path only store trainable tensors (head + mT5
        # + stat buffers); the permanently frozen VAE always comes from
        # --vae_checkpoint, hence strict=False
        ret = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys ({len(ret.missing_keys)}): \n",
              "\n".join(ret.missing_keys[:10]))
        print(f"Unexpected keys ({len(ret.unexpected_keys)}): \n",
              "\n".join(ret.unexpected_keys[:10]))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(
            args.warmup_epochs * len(train_dataloader)
            / args.gradient_accumulation_steps
        ),
        num_training_steps=int(
            args.epochs * len(train_dataloader)
            / args.gradient_accumulation_steps
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
            if has_distinct_dev:
                print("📄 dev result")
                evaluate(eval_args, dev_dataloader, model, model_without_ddp,
                         phase="dev")
            print("📄 test result")
            evaluate(eval_args, test_dataloader, model, model_without_ddp,
                     phase="test")
        return

    start_time = time.time()
    max_accuracy = 0
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args, model, train_dataloader, optimizer, epoch
        )

        if args.output_dir:
            checkpoint_name = (
                "last_checkpoint.pth" if args.save_last_only
                else f"checkpoint_{epoch}.pth"
            )
            utils.save_on_master(
                {"model": get_requires_grad_dict(model_without_ddp)},
                output_dir / checkpoint_name,
            )

        # single gpu inference
        if utils.is_main_process():
            selection_stats = evaluate(
                eval_args,
                dev_dataloader,
                model,
                model_without_ddp,
                phase=selection_phase,
            )
            if has_distinct_dev:
                evaluate(eval_args, test_dataloader, model, model_without_ddp,
                         phase="test")

            if max_accuracy < selection_stats["bleu4"]:
                max_accuracy = selection_stats["bleu4"]
                if args.output_dir:
                    utils.save_on_master(
                        {"model": get_requires_grad_dict(model_without_ddp)},
                        output_dir / "best_checkpoint.pth",
                    )

            print(
                f"BLEU-4 on {eval_args.dataset} {selection_phase} "
                f"({len(dev_dataloader)} batches): "
                f"{selection_stats['bleu4']:.2f}"
            )
            print(f"Max BLEU-4: {max_accuracy:.2f}%")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in selection_stats.items()},
                "eval_dataset": eval_args.dataset,
                "eval_split": selection_phase,
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print("Training time {}".format(
        str(datetime.timedelta(seconds=int(total_time)))
    ))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parsed_args = get_pose2d_klvae_args_parser().parse_args()
    validate_args(parsed_args)
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
