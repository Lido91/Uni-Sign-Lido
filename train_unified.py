"""Three-stage training entrypoint for the unified (SLT + T2P) mT5 model.

Stages mirror JanusFlow's recipe:

  Stage 1 (adaptation):  train ONLY the new generation components
      (latent_in_proj, mask_emb, cond_out_head, diff_head) on the T2P
      objective; mT5, ST-GCN and the KL-VAE stay frozen.
  Stage 2 (unified):     unfreeze mT5; mixed-task training — every step
      takes one SLT batch (CE) and one T2P batch (diffusion), summed loss.
  Stage 3 (SFT):         additionally unfreeze the ST-GCN; same mixed-task
      objective on the downstream dataset.

The KL-VAE is permanently frozen in every stage and never saved: unified
checkpoints contain everything EXCEPT vae_encoder/vae_decoder and load with
strict=False. ``--slt_init`` seeds mT5 + ST-GCN from a trained SLT
checkpoint (pass it in every stage; frozen modules are not re-saved).

    deepspeed train_unified.py --train_stage 1 --dataset OpenASL --dtype fp32 \
        --vae_checkpoint <net_best_hand.pth> --slt_init <slt best_checkpoint.pth>
"""

import argparse
import copy
import datetime
import json
import os
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from timm.optim import create_optimizer
from transformers import get_scheduler

import utils
from config import dev_label_paths, test_label_paths, train_label_paths
from datasets import S2T_Dataset
from datasets_pose2d_klvae import (
    FRAME_SAMPLING_MODES,
    S2T_Dataset_Pose2D_KLVAE,
)
from diffusion_gen.factory import (
    create_gaussian_diffusion,
    create_gaussian_diffusion_ddim,
)
from diffusion_gen.resample import create_named_schedule_sampler
from fine_tuning import (
    evaluate,
    handle_nonfinite_loss,
    move_src_input_to_device,
    train_one_epoch,
)
from models_unified import Uni_Sign_Unified

DEFAULT_VAE_META = (
    "/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/"
    "kl_vae_h2s_openasl_pose2d/meta"
)

SLT_SIDE_MODULES = (
    "proj_linear", "gcn_modules", "fusion_gcn_modules", "pose_proj",
)


def get_unified_args_parser():
    parser = argparse.ArgumentParser(
        "Uni-Sign unified (SLT+T2P) scripts",
        parents=[utils.get_args_parser()],
    )
    parser.add_argument("--train_stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--vae_checkpoint", required=True,
                        help="MoMADiff kl_vae_h2s_openasl_pose2d checkpoint.")
    parser.add_argument("--vae_mean_path", default=f"{DEFAULT_VAE_META}/mean.npy")
    parser.add_argument("--vae_std_path", default=f"{DEFAULT_VAE_META}/std.npy")
    parser.add_argument("--frame_sampling", default="vae_linear",
                        choices=list(FRAME_SAMPLING_MODES))
    parser.add_argument("--latent_mean_path", default="",
                        help="(512,) per-dim latent mean from "
                             "check_latent_scale.py --save_latent_stats; "
                             "empty disables latent standardization.")
    parser.add_argument("--latent_std_path", default="")
    parser.add_argument("--slt_init", default="",
                        help="Trained SLT checkpoint seeding mT5 + ST-GCN "
                             "(pass in every stage).")
    parser.add_argument("--slt_loss_weight", type=float, default=1.0)
    parser.add_argument("--t2p_loss_weight", type=float, default=1.0)
    parser.add_argument("--t2p_max_length", type=int, default=400,
                        help="Frame cap for the T2P loader (SLT keeps "
                             "--max_length).")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--diff_layers", type=int, default=4)
    parser.add_argument("--diff_ff_size", type=int, default=1024)
    parser.add_argument("--trans_infer_timesteps", type=int, default=9)
    parser.add_argument("--guidance_param", type=float, default=3.0)
    # experiment tracking
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="uni-sign-unified")
    parser.add_argument("--wandb_run_name", default="",
                        help="Defaults to the output_dir basename.")
    # crash recovery (epoch granularity)
    parser.add_argument("--resume", default="auto", choices=["auto", "off"],
                        help="'auto': if <output_dir>/resume_state exists, "
                             "restore model + optimizer + lr scheduler + "
                             "epoch (and re-attach the wandb run) and "
                             "continue; 'off': always start fresh. To "
                             "intentionally restart a stage from scratch, "
                             "delete <output_dir>/resume_state and "
                             "wandb_run_id.txt (or pass --resume off). "
                             "Resuming requires the SAME number of GPUs "
                             "(ZeRO optimizer shards are per-rank).")
    return parser


def validate_args(args):
    if args.dataset not in ("How2Sign", "OpenASL"):
        raise ValueError("--dataset must be How2Sign or OpenASL")
    if args.task != "SLT":
        raise ValueError("this entrypoint supports --task SLT only")
    if args.rgb_support:
        raise ValueError("--rgb_support belongs to the original ST-GCN path")
    if args.llm_type != "mt5":
        raise ValueError("--llm_type must be mt5")


# keys never stored in unified checkpoints: the frozen codec comes from
# --vae_checkpoint and the z-score stats from --vae_mean_path/--vae_std_path
# (storing them could silently override the stats of a newer VAE)
VAE_OWNED_PREFIXES = (
    "vae_encoder.", "vae_decoder.", "motion_mean", "motion_std",
    "latent_mean", "latent_std",
)
# generation-side keys a *unified* checkpoint must contain
GEN_ADAPTER_PREFIXES = (
    "latent_in_proj.", "mask_emb", "cond_out_head.", "diff_head.",
)


def unified_state_dict(model):
    """Full state minus the VAE-owned tensors (see VAE_OWNED_PREFIXES)."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(VAE_OWNED_PREFIXES)
    }


class _EvalShim:
    """Rank-0 validation must run on the RAW module: the DDP/DeepSpeed
    wrappers issue collectives that the other (idle or training) ranks would
    never match. This shim only adds the dtype probes that
    fine_tuning.move_src_input_to_device expects from an engine."""

    def __init__(self, module, args):
        self._module = module
        self._dtype = args.dtype

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def bfloat16_enabled(self):
        return self._dtype == "bf16"

    def fp16_enabled(self):
        return self._dtype == "fp16"

    def __getattr__(self, name):
        return getattr(self._module, name)


def apply_stage_freezing(model, stage):
    """VAE is already frozen by construction; stage controls the rest.

    Frozen modules are also registered in ``model.frozen_eval_modules`` so
    ``Uni_Sign_Unified.train()`` keeps them in eval mode — plain
    requires_grad=False would still let their BatchNorm update running
    statistics (and dropout fire), silently violating the stage contract.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True
    for parameter in model.vae_encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.vae_decoder.parameters():
        parameter.requires_grad = False
    frozen_eval = []
    if stage <= 2:
        for name in SLT_SIDE_MODULES:
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad = False
            frozen_eval.append(name)
        model.part_para.requires_grad = False
    if stage <= 1:
        for parameter in model.mt5_model.parameters():
            parameter.requires_grad = False
        frozen_eval.append("mt5_model")
    model.frozen_eval_modules = tuple(frozen_eval)


def load_partial(model, path, label, allowed_missing_prefixes):
    """strict=False load with an explicit contract: every missing key must
    match an allowed prefix, otherwise the checkpoint is the wrong kind
    (e.g. a plain SLT checkpoint passed as --finetune would leave the whole
    generation head randomly initialized — fail loudly instead)."""
    print(f"Loading {label}: {path}")
    state_dict = torch.load(path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    ret = model.load_state_dict(state_dict, strict=False)
    violations = [
        key for key in ret.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    print(f"  missing keys: {len(ret.missing_keys)} "
          f"(disallowed: {len(violations)}); "
          f"unexpected keys: {len(ret.unexpected_keys)}")
    if violations:
        raise ValueError(
            f"{label} at {path} does not provide required keys, e.g. "
            f"{violations[:5]} — wrong checkpoint kind?"
        )


def build_t2p_args(args):
    t2p_args = copy.copy(args)
    t2p_args.max_length = args.t2p_max_length
    return t2p_args


@torch.no_grad()
def evaluate_t2p_loss(args, data_loader, model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    for src_input, tgt_input in metric_logger.log_every(
        data_loader, 10, "t2p-val:"
    ):
        move_src_input_to_device(src_input, model)
        out = model(src_input, tgt_input, task="t2p")
        metric_logger.update(loss=out["loss"].item())
    print("Averaged t2p stats:", metric_logger)
    return {f"t2p_{k}": m.global_avg for k, m in metric_logger.meters.items()}


def main(args):
    utils.init_distributed_mode_ds(args)
    print(args)
    utils.set_seed(args.seed)

    wandb_run = None
    if args.use_wandb and utils.is_main_process():
        import wandb  # fail loudly if tracking was requested but not installed
        id_file = (Path(args.output_dir) / "wandb_run_id.txt"
                   if args.output_dir else None)
        run_id = None
        if args.resume == "auto" and id_file is not None and id_file.exists():
            run_id = id_file.read_text().strip() or None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name
            or (Path(args.output_dir).name if args.output_dir else None),
            id=run_id,
            resume="allow" if run_id else None,
            config={k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))},
            tags=[f"stage{args.train_stage}", args.dataset, "unified"],
        )
        if id_file is not None:
            id_file.write_text(wandb_run.id)

    joint = args.train_stage >= 2

    # ---- datasets ---------------------------------------------------------
    print("Creating datasets:")
    t2p_args = build_t2p_args(args)
    t2p_train = S2T_Dataset_Pose2D_KLVAE(
        path=train_label_paths[args.dataset], args=t2p_args, phase="train"
    )
    print("t2p train:", t2p_train)
    t2p_train_sampler = torch.utils.data.distributed.DistributedSampler(
        t2p_train, shuffle=True
    )
    t2p_train_loader = DataLoader(
        t2p_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=t2p_train.collate_fn,
        sampler=t2p_train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    has_distinct_dev = "How2Sign" not in args.dataset
    t2p_val_phase = "dev" if has_distinct_dev else "test"
    t2p_val = S2T_Dataset_Pose2D_KLVAE(
        path=(dev_label_paths if has_distinct_dev else test_label_paths)[
            args.dataset
        ],
        args=t2p_args,
        phase=t2p_val_phase,
    )
    t2p_val_loader = DataLoader(
        t2p_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=t2p_val.collate_fn,
        sampler=torch.utils.data.SequentialSampler(t2p_val),
        pin_memory=args.pin_mem,
    )

    slt_train_loader = slt_dev_loader = slt_test_loader = None
    if joint or args.eval:
        slt_train = S2T_Dataset(
            path=train_label_paths[args.dataset], args=args, phase="train"
        )
        print("slt train:", slt_train)
        slt_train_sampler = torch.utils.data.distributed.DistributedSampler(
            slt_train, shuffle=True
        )
        slt_train_loader = DataLoader(
            slt_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=slt_train.collate_fn,
            sampler=slt_train_sampler,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        slt_test = S2T_Dataset(
            path=test_label_paths[args.dataset], args=args, phase="test"
        )
        slt_test_loader = DataLoader(
            slt_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=slt_test.collate_fn,
            sampler=torch.utils.data.SequentialSampler(slt_test),
            pin_memory=args.pin_mem,
        )
        if has_distinct_dev:
            slt_dev = S2T_Dataset(
                path=dev_label_paths[args.dataset], args=args, phase="dev"
            )
            slt_dev_loader = DataLoader(
                slt_dev,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=slt_dev.collate_fn,
                sampler=torch.utils.data.SequentialSampler(slt_dev),
                pin_memory=args.pin_mem,
            )
        else:
            slt_dev_loader = slt_test_loader

    # ---- model ------------------------------------------------------------
    print("Creating unified model:")
    args.default_task = "t2p" if not joint else "slt"
    model = Uni_Sign_Unified(args=args)
    model.attach_diffusion(
        create_gaussian_diffusion(args.diffusion_steps, args.noise_schedule),
        create_gaussian_diffusion_ddim(
            args.diffusion_steps, args.ddim_steps, args.noise_schedule
        ),
        None,  # sampler needs the train diffusion; created just below
    )
    model.schedule_sampler = create_named_schedule_sampler(
        "uniform", model.train_diffusion
    )
    model.cuda()

    if args.slt_init:
        # an SLT checkpoint legitimately lacks the generation side + VAE
        load_partial(
            model, args.slt_init, "SLT init (mT5 + ST-GCN)",
            allowed_missing_prefixes=VAE_OWNED_PREFIXES
            + GEN_ADAPTER_PREFIXES + ("latent_pos_pe",),
        )
    if args.finetune != "":
        # a unified checkpoint must provide everything except VAE-owned keys
        load_partial(
            model, args.finetune, "unified checkpoint",
            allowed_missing_prefixes=VAE_OWNED_PREFIXES,
        )

    apply_stage_freezing(model, args.train_stage)
    model.train()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    steps_per_epoch = (
        min(len(t2p_train_loader), len(slt_train_loader))
        if joint else len(t2p_train_loader)
    )
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(
            args.warmup_epochs * steps_per_epoch
            / args.gradient_accumulation_steps
        ),
        num_training_steps=int(
            args.epochs * steps_per_epoch / args.gradient_accumulation_steps
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
            eval_model = _EvalShim(model_without_ddp, args)
            model_without_ddp.default_task = "slt"
            if has_distinct_dev:
                print("📄 SLT dev result")
                evaluate(args, slt_dev_loader, eval_model, model_without_ddp,
                         phase="dev")
            print("📄 SLT test result")
            evaluate(args, slt_test_loader, eval_model, model_without_ddp,
                     phase="test")
            print("📄 T2P val diffusion loss")
            evaluate_t2p_loss(args, t2p_val_loader, eval_model)
        return

    # ---- crash recovery: full-state resume (epoch granularity) ------------
    resume_dir = None
    if args.output_dir and args.resume == "auto":
        resume_dir = output_dir / "resume_state"

    start_epoch = 0
    best_metric = float("-inf")
    if resume_dir is not None and (resume_dir / "latest").exists():
        # Collective DeepSpeed load on every rank: module weights (override
        # whatever --slt_init/--finetune seeded), per-rank ZeRO optimizer
        # shards and the lr scheduler. Requires the same world size as the
        # interrupted run.
        load_path, client_state = model.load_checkpoint(
            str(resume_dir),
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(
                f"resume state at {resume_dir} exists but could not be "
                f"loaded — delete the directory to restart this stage"
            )
        start_epoch = int(client_state.get("epoch", 0))
        best_metric = float(client_state.get("best_metric", float("-inf")))
        print(f"Resumed from {load_path}: continuing at epoch {start_epoch} "
              f"(best selection metric so far {best_metric:.4f})")

    start_time = time.time()
    print(f"Start training stage {args.train_stage} for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            t2p_train_sampler.set_epoch(epoch)
            if joint:
                slt_train_sampler.set_epoch(epoch)

        if not joint:
            model_without_ddp.default_task = "t2p"
            train_stats = train_one_epoch(
                args, model, t2p_train_loader, optimizer, epoch
            )
        else:
            train_stats = train_joint_epoch(
                args, model, slt_train_loader, t2p_train_loader,
                optimizer, epoch, wandb_run=wandb_run,
            )

        if args.output_dir:
            checkpoint_name = (
                "last_checkpoint.pth" if args.save_last_only
                else f"checkpoint_{epoch}.pth"
            )
            utils.save_on_master(
                {"model": unified_state_dict(model_without_ddp)},
                output_dir / checkpoint_name,
            )

        # All ranks wait here: rank-0 validation runs on the raw module (no
        # DDP/DeepSpeed collectives), so the other ranks must not race ahead
        # into the next epoch's collectives meanwhile.
        if args.distributed:
            torch.distributed.barrier()

        if utils.is_main_process():
            eval_model = _EvalShim(model_without_ddp, args)
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "stage": args.train_stage,
                "n_parameters": n_parameters,
            }

            if not joint:
                # stage 1 selection: minimal T2P validation diffusion loss
                val_stats = evaluate_t2p_loss(args, t2p_val_loader, eval_model)
                selection = -val_stats["t2p_loss"]
                log_stats.update(val_stats)
                print(f"T2P val loss: {val_stats['t2p_loss']:.4f}")
            else:
                # stage 2/3 selection: SLT BLEU-4 (dev when it exists)
                model_without_ddp.default_task = "slt"
                phase = "dev" if has_distinct_dev else "test"
                slt_stats = evaluate(
                    args, slt_dev_loader, eval_model, model_without_ddp,
                    phase=phase,
                )
                val_stats = evaluate_t2p_loss(args, t2p_val_loader, eval_model)
                selection = slt_stats["bleu4"]
                log_stats.update(
                    {f"slt_{k}": v for k, v in slt_stats.items()}
                )
                log_stats.update(val_stats)
                print(f"SLT BLEU-4 ({phase}): {slt_stats['bleu4']:.2f}  "
                      f"T2P val loss: {val_stats['t2p_loss']:.4f}")

            if selection > best_metric:
                best_metric = selection
                if args.output_dir:
                    utils.save_on_master(
                        {"model": unified_state_dict(model_without_ddp)},
                        output_dir / "best_checkpoint.pth",
                    )
            print(f"Best selection metric so far: {best_metric:.4f}")

            if wandb_run is not None:
                wandb_run.log(log_stats)

            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            torch.distributed.barrier()

        if resume_dir is not None:
            # Full-state snapshot for crash recovery (collective call on all
            # ranks). Write the new tag first; only afterwards does rank 0
            # prune the previous one — an interruption mid-save leaves
            # `latest` pointing at the still-complete older snapshot.
            # best_metric is rank 0's (the eval rank); DeepSpeed persists
            # rank 0's client_state, so the stale copies on other ranks
            # never reach disk.
            tag = f"epoch{epoch}"
            model.save_checkpoint(
                str(resume_dir),
                tag=tag,
                client_state={"epoch": epoch + 1, "best_metric": best_metric},
            )
            if utils.is_main_process():
                for old in resume_dir.glob("epoch*"):
                    if old.is_dir() and old.name != tag:
                        shutil.rmtree(old, ignore_errors=True)

    total_time = time.time() - start_time
    print("Training time {}".format(
        str(datetime.timedelta(seconds=int(total_time)))
    ))
    if wandb_run is not None:
        wandb_run.finish()


def train_joint_epoch(args, model, slt_loader, t2p_loader, optimizer, epoch,
                      wandb_run=None):
    """One epoch of mixed-task training: each optimizer step consumes one SLT
    batch and one T2P batch; the weighted sum backpropagates once, so every
    trainable parameter participates every step (DeepSpeed-safe)."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Joint Epoch: [{}/{}]".format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    class _SizedZip:
        """zip() with a __len__, so MetricLogger.log_every can show ETA."""

        def __init__(self, a, b):
            self.a, self.b = a, b

        def __iter__(self):
            return zip(self.a, self.b)

        def __len__(self):
            return min(len(self.a), len(self.b))

    paired = _SizedZip(slt_loader, t2p_loader)

    nonfinite_streak = 0
    for step, (slt_batch, t2p_batch) in enumerate(
        metric_logger.log_every(paired, print_freq, header)
    ):
        slt_src, slt_tgt = slt_batch
        t2p_src, t2p_tgt = t2p_batch
        move_src_input_to_device(slt_src, model)
        move_src_input_to_device(t2p_src, model)

        # single wrapped forward covering both tasks (DDP-safe)
        joint_out = model(
            slt_src, slt_tgt, task="joint",
            t2p_src_input=t2p_src, t2p_tgt_input=t2p_tgt,
            slt_weight=args.slt_loss_weight,
            t2p_weight=args.t2p_loss_weight,
        )
        total_loss = joint_out["loss"]
        loss_value = total_loss.item()

        nonfinite_streak = handle_nonfinite_loss(
            args, loss_value, step, epoch, nonfinite_streak,
            slt_src, slt_tgt,
            extra={
                "t2p_src_input": t2p_src,
                "t2p_tgt_input": t2p_tgt,
                "slt_loss": joint_out["slt_loss"].item(),
                "t2p_loss": joint_out["t2p_loss"].item(),
            },
        )
        if nonfinite_streak:
            continue

        model.backward(total_loss)
        model.step()

        metric_logger.update(
            loss=loss_value,
            slt_loss=joint_out["slt_loss"].item(),
            t2p_loss=joint_out["t2p_loss"].item(),
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if wandb_run is not None and step % print_freq == 0:
            wandb_run.log({
                "step/loss": loss_value,
                "step/slt_loss": joint_out["slt_loss"].item(),
                "step/t2p_loss": joint_out["t2p_loss"].item(),
                "step/lr": optimizer.param_groups[0]["lr"],
                "step/global_step": epoch * len(paired) + step,
            })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parsed_args = get_unified_args_parser().parse_args()
    validate_args(parsed_args)
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
