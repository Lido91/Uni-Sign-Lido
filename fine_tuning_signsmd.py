"""Fine-tuning entrypoint: baseline Uni-Sign (ST-GCN + mT5) fused with
SignSMD structured description text.

Same data pipeline, DeepSpeed infrastructure, BLEU-4 checkpoint selection,
and generation as the original ``fine_tuning.py`` — the ONLY changes are:

1. dataset: ``datasets_signsmd.S2T_Dataset_SignSMD`` loads either one flat
   description or the four standalone part texts plus event JSON;
2. model, selected by ``--fusion``:
   - ``early`` (default): ``models_signsmd.Uni_Sign_SignSMD`` prepends the
     description as real mT5 tokens ahead of the unchanged
     [translate prompt][pose] input (one encoder pass);
   - ``dual``: ``models_signsmd_dual.Uni_Sign_SignSMD_Dual`` encodes pose
     and description in separate passes of the shared encoder and fuses
     the hidden states (doc/signsmd_dual_stream_fusion_plan.md).
   - ``paired``: ``models_signsmd_paired.Uni_Sign_SignSMD_Paired`` embeds
     each short event phrase independently and injects it into the matching
     pose part only over the event's trainer-aligned step span.

With ``--fusion early`` or ``--fusion dual --fusion_gate none`` the fused
model has EXACTLY the same parameters as the baseline ``models.Uni_Sign``
(pose-only), so ``--finetune`` accepts baseline checkpoints with
strict=True and these paths' checkpoints load back into the baseline the
   same way.  The dual gate variants and paired model add parameters and
   warm-start with strict=False; paired is functionally pose-only at step 0
   because its final injection gate is exactly zero-initialized.

Train (How2Sign, warm-started from a baseline checkpoint):

    deepspeed fine_tuning_signsmd.py \
        --dataset How2Sign --task SLT --dtype fp32 \
        --finetune <baseline_or_stage_checkpoint.pth> \
        --signsmd_text_root data/How2Sign/signsmd2d_v0_1/texts \
        --output_dir out/signsmd_fusion_h2s

Evaluate:

    deepspeed fine_tuning_signsmd.py ... --eval \
        --finetune out/signsmd_fusion_h2s/best_checkpoint.pth
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
from datasets_signsmd import (
    FRAME_SAMPLING_MODES,
    SIGNSMD_MISSING_POLICIES,
    S2T_Dataset_SignSMD,
)
from fine_tuning import evaluate, train_one_epoch
from models import get_requires_grad_dict
from models_signsmd import Uni_Sign_SignSMD
from models_signsmd_dual import Uni_Sign_SignSMD_Dual
from models_signsmd_paired import Uni_Sign_SignSMD_Paired


def get_signsmd_args_parser():
    parser = argparse.ArgumentParser(
        "Uni-Sign SignSMD fusion scripts",
        parents=[utils.get_args_parser()],
    )
    parser.add_argument(
        "--signsmd_text_root",
        required=True,
        help="Directory of offline-compiled SignSMD descriptions, one "
             "<clip-name>.txt per sample for early/dual, or four "
             "<part>/<clip-name>.txt files for paired fusion.",
    )
    parser.add_argument(
        "--signsmd_event_root",
        default="",
        help="Paired only: directory containing <clip-name>.json event/span "
             "contracts. Default: the events sibling of --signsmd_text_root.",
    )
    parser.add_argument(
        "--eval_signsmd_text_root",
        default="",
        help="Optional SignSMD text root for a different evaluation dataset, "
             "notably How2Sign under --eval-on-how2sign-test. Defaults to "
             "--signsmd_text_root.",
    )
    parser.add_argument(
        "--eval_signsmd_event_root",
        default="",
        help="Paired counterpart of --eval_signsmd_text_root. Defaults to "
             "the training event root (or its derived events sibling).",
    )
    parser.add_argument(
        "--events_required",
        action="store_true",
        help="Paired only: fail on any missing event JSON instead of using "
             "an empty event list.",
    )
    parser.add_argument(
        "--signsmd_max_tokens",
        type=int,
        default=512,
        help="Hard cap on mT5 tokens for the description segment "
             "(tail-truncated; prefer compile-time compaction).",
    )
    parser.add_argument(
        "--signsmd_part_max_tokens",
        type=int,
        default=128,
        help="Paired only: independent token cap for EACH event phrase; "
             "phrases and parts never share one truncation budget.",
    )
    parser.add_argument(
        "--signsmd_on_missing",
        default="error",
        choices=list(SIGNSMD_MISSING_POLICIES),
        help="error (default): fail fast on a missing description file; "
             "empty: substitute an empty description (smoke tests only).",
    )
    parser.add_argument(
        "--frame_sampling",
        default="linear",
        choices=list(FRAME_SAMPLING_MODES),
        help="linear (default): deterministic linear-index subsampling of "
             "clips longer than --max_length, so the SignSMD text's "
             "percentage spans align exactly with the pose timeline; "
             "unisign_random: baseline random subsampling (ablation).",
    )
    parser.add_argument(
        "--fusion",
        default="early",
        choices=["early", "dual", "paired"],
        help="early (default): SMD tokens concatenated with the pose input "
             "of a single encoder pass (models_signsmd); dual: pose and SMD "
             "encoded in separate passes of the shared encoder, fused at "
             "the hidden-state level (models_signsmd_dual); paired: event "
             "phrases are injected into matching part/time pose features.",
    )
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=1.0,
        help="dual only: fixed scale on the SMD hidden states before "
             "concatenation (1.0 = plain concat; still zero parameters).",
    )
    parser.add_argument(
        "--fusion_gate",
        default="none",
        choices=["none", "global", "sample"],
        help="dual only: learnable scale on the SMD hidden states — "
             "global: sigmoid of one learnable logit; sample: per-sample "
             "sigmoid(W[pool(H_P);pool(H_S)]). Both init at ~0.1 and add "
             "smd_gate_* parameters (warm start becomes strict=False).",
    )
    parser.add_argument(
        "--smd_dropout",
        type=float,
        default=0.0,
        help="dual only: per-sample probability of zeroing the SMD "
             "segment's attention mask during training (blocks the "
             "SMD->translation shortcut).",
    )
    parser.add_argument(
        "--text_event_dropout",
        type=float,
        default=0.3,
        help="Paired only: per-sample probability of dropping all part-text "
             "events during training (default 0.3).",
    )
    parser.add_argument(
        "--eval_mask_text",
        action="store_true",
        help="Paired only, with --eval: zero all event features to measure "
             "the integrity of the pose-only path.",
    )
    parser.add_argument(
        "--eval_mask_smd",
        action="store_true",
        help="dual only, with --eval: mask the SMD segment for every "
             "sample — pose-stream-integrity diagnostic vs the pose-only "
             "baseline.",
    )
    return parser


def validate_args(args):
    if args.dataset not in ("How2Sign", "OpenASL"):
        raise ValueError(
            "SignSMD descriptions are compiled for the RTMPose "
            "How2Sign/OpenASL corpora; --dataset must be How2Sign or OpenASL"
        )
    if args.task != "SLT":
        raise ValueError("this entrypoint supports --task SLT only")
    if args.rgb_support:
        raise ValueError(
            "--rgb_support is not supported on the SignSMD fusion path yet"
        )
    if args.llm_type != "mt5":
        raise ValueError(
            "the SignSMD segment is tokenized with the mT5 tokenizer; "
            "--llm_type must be mt5"
        )
    if args.signsmd_max_tokens < 1:
        raise ValueError("--signsmd_max_tokens must be >= 1")
    if args.signsmd_part_max_tokens < 1:
        raise ValueError("--signsmd_part_max_tokens must be >= 1")
    if args.fusion == "dual":
        if not 0.0 <= args.smd_dropout < 1.0:
            raise ValueError("--smd_dropout must be in [0, 1)")
        if args.fusion_gate != "none" and args.fusion_alpha != 1.0:
            raise ValueError(
                "--fusion_alpha and --fusion_gate are mutually exclusive: "
                "the gate replaces the fixed scale"
            )
        if args.eval_mask_smd and not args.eval:
            raise ValueError(
                "--eval_mask_smd is an evaluation-time diagnostic; "
                "combine it with --eval"
            )
        if (
            args.eval_mask_text
            or args.events_required
            or args.signsmd_event_root
            or args.eval_signsmd_event_root
        ):
            raise ValueError(
                "paired event flags require --fusion paired"
            )
    elif args.fusion == "paired":
        if (
            args.fusion_alpha != 1.0
            or args.fusion_gate != "none"
            or args.smd_dropout != 0.0
            or args.eval_mask_smd
        ):
            raise ValueError(
                "--fusion_alpha/--fusion_gate/--smd_dropout/--eval_mask_smd "
                "require --fusion dual"
            )
        if args.frame_sampling != "linear":
            raise ValueError(
                "--fusion paired requires --frame_sampling linear so event "
                "spans index the pose sequence exactly"
            )
        if not 0.0 <= args.text_event_dropout <= 1.0:
            raise ValueError("--text_event_dropout must be in [0, 1]")
        if args.eval_mask_text and not args.eval:
            raise ValueError(
                "--eval_mask_text is an evaluation-time diagnostic; "
                "combine it with --eval"
            )
    else:
        if (
            args.fusion_alpha != 1.0
            or args.fusion_gate != "none"
            or args.smd_dropout != 0.0
            or args.eval_mask_smd
        ):
            raise ValueError(
                "--fusion_alpha/--fusion_gate/--smd_dropout/--eval_mask_smd "
                "require --fusion dual"
            )
        if (
            args.eval_mask_text
            or args.events_required
            or args.signsmd_event_root
            or args.eval_signsmd_event_root
        ):
            raise ValueError(
                "event roots/--events_required/--eval_mask_text require "
                "--fusion paired"
            )


def main(args):
    utils.init_distributed_mode_ds(args)
    print(args)
    utils.set_seed(args.seed)

    print("Creating dataset:")
    train_data = S2T_Dataset_SignSMD(
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
        if args.eval_signsmd_text_root:
            eval_args.signsmd_text_root = args.eval_signsmd_text_root
        elif args.dataset != "How2Sign":
            raise ValueError(
                "--eval-on-how2sign-test changes the dataset from "
                f"{args.dataset} to How2Sign; pass "
                "--eval_signsmd_text_root for the How2Sign corpus"
            )
        if args.eval_signsmd_event_root:
            eval_args.signsmd_event_root = args.eval_signsmd_event_root
        elif args.fusion == "paired" and args.dataset != "How2Sign":
            raise ValueError(
                "--fusion paired with --eval-on-how2sign-test requires "
                "--eval_signsmd_event_root for the How2Sign event corpus"
            )
        print(
            "WARNING: evaluating directly on How2Sign test; "
            "best_checkpoint.pth will be selected using How2Sign test metrics."
        )

    test_data = S2T_Dataset_SignSMD(
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
        dev_data = S2T_Dataset_SignSMD(
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

    print(f"Creating SignSMD fusion model ({args.fusion}):")
    if args.fusion == "dual":
        model = Uni_Sign_SignSMD_Dual(args=args)
    elif args.fusion == "paired":
        model = Uni_Sign_SignSMD_Paired(args=args)
    else:
        model = Uni_Sign_SignSMD(args=args)
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
        # Early and gate-free dual fusion have exactly the baseline parameter
        # set. Paired always adds text projection/gate parameters and therefore
        # warm-starts baseline or earlier paired checkpoints with strict=False.
        strict = args.fusion == "early" or (
            args.fusion == "dual" and args.fusion_gate == "none"
        )
        ret = model.load_state_dict(state_dict, strict=strict)
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
    parsed_args = get_signsmd_args_parser().parse_args()
    validate_args(parsed_args)
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
