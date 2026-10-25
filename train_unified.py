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
from pose2d_metrics import DTWMPJPE2DMetrics

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
    parser.add_argument('--merge_datasets', action='store_true',
                        help='Concatenate all OpenASL and How2Sign train samples '
                             'with natural proportions; evaluate each test separately. '
                             '--dataset chooses the checkpoint-selection dataset.')
    parser.add_argument("--val_split", default="test", choices=["dev", "test"],
                        help="Split evaluated after each epoch for both tasks. "
                             "Default test; checkpoints are then test-selected. "
                             "Joint stages select best_checkpoint by SLT BLEU-4; "
                             "best_slg_checkpoint uses minimum hand DTW-MPJPE2D.")
    parser.add_argument("--skip_slg_generation_val", action="store_true",
                        help="Skip the full generated-pose DTW pass (debug only). "
                             "Default: generate and score the entire selected split each epoch.")
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
                             "continue; 'off': start fresh while saving new "
                             "recovery snapshots. To "
                             "intentionally restart a stage from scratch, "
                             "delete <output_dir>/resume_state and "
                             "wandb_run_id.txt (or pass --resume off). "
                             "Resuming requires the SAME number of GPUs "
                             "(ZeRO optimizer shards are per-rank).")
    # Geometry is opt-in; the baseline keeps its original model/optimizer.
    from geosign_training import add_geosign_args
    add_geosign_args(parser)
    parser.add_argument('--smoke_steps', type=int, default=0,
                        help='Run this many training microbatches, then exit before '
                             'validation/checkpoint saving; 0 means full training.')
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
    if args.merge_datasets and (args.train_stage == 1 or args.val_split != 'test'):
        raise ValueError('Merged training supports joint Stage 2/3 with --val_split test')
    if args.use_hyperbolic:
        if args.train_stage == 1:
            raise ValueError('Geo-Sign is enabled only for joint Stage 2/3')
        if args.dtype != 'fp32' or args.zero_stage != 0:
            raise ValueError('The unchanged Geo-Sign optimizer path requires '
                             '--dtype fp32 --zero_stage 0')
    resolve_validation_split(args)


def resolve_validation_split(args):
    """Use the explicitly selected split; never silently substitute another."""
    split = args.val_split
    paths = {"dev": dev_label_paths, "test": test_label_paths}
    if split not in paths:
        raise ValueError(f"unknown validation split {split!r}")
    path = paths[split].get(args.dataset, "")
    if not path:
        raise ValueError(f"No {split} labels configured for {args.dataset}; "
                         "configure that split or select --val_split test")
    return split, path


def restore_best_metric(client_state, dataset, val_split):
    """Scores selected on dev must not compete with scores selected on test.

    Legacy snapshots used dev on OpenASL and test on How2Sign.
    We retain the optimizer/model state but restart selection after a split change.
    """
    previous_split = client_state.get(
        "validation_split", "test" if dataset == "How2Sign" else "dev"
    )
    if previous_split != val_split:
        print(f"Validation split changed {previous_split} -> {val_split}; "
              "resetting best selection metric (model/optimizer resume is retained).")
        return float("-inf")
    return float(client_state.get("best_metric", float("-inf")))


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


def slg_validation_protocol(args):
    protocol = {'metric': 'momadiff_accumulated_DTW_MPJPE2D_hand',
            'dataset': args.dataset, 'split': args.val_split, 'seed': args.seed,
            'max_frames': args.t2p_max_length, 'frame_sampling': args.frame_sampling,
            'ddim_steps': args.ddim_steps, 'mar_states': args.trans_infer_timesteps,
            'guidance': args.guidance_param, 'diffusion_steps': args.diffusion_steps,
            'noise_schedule': args.noise_schedule, 'clip_coordinates': False,
            'vae_checkpoint': args.vae_checkpoint,
            'vae_mean_path': args.vae_mean_path, 'vae_std_path': args.vae_std_path,
            'batch_size': args.batch_size,
            'latent_mean_path': args.latent_mean_path, 'latent_std_path': args.latent_std_path}
    if getattr(args, 'merge_datasets', False):
        protocol['training_datasets'] = ['OpenASL', 'How2Sign']
        protocol['sampling'] = 'concatenated_natural'
    return protocol


@torch.no_grad()
def evaluate_slg_generation(args, data_loader, model):
    """Full text->pose evaluation on the raw rank-0 model, using GT lengths.

    No SLT judge is involved. Sampling RNG is fixed and restored afterward,
    keeping epoch comparisons reproducible without changing training RNG.
    """
    metric = DTWMPJPE2DMetrics()
    logger = utils.MetricLogger(delimiter='  ')
    device = model.motion_mean.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    model.eval()
    seen = 0
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(args.seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(args.seed)
            for src, tgt in logger.log_every(data_loader, 10, f'SLG-{args.val_split}:'):
                parts, pred_lengths = model.generate_pose(
                    tgt['gt_sentence'], src['src_length_batch'],
                    timesteps=args.trans_infer_timesteps, cond_scale=args.guidance_param,
                    clip_coordinates=False,
                )
                metric.update_parts(parts, src, src['src_length_batch'].tolist(),
                                    pred_lengths.tolist(), args.dataset, src['name_batch'])
                seen += len(tgt['gt_sentence'])
    finally:
        model.train(was_training)
    results = metric.results()
    monitor = f'{args.dataset.lower()}_DTW_MPJPE2D_hand'
    if monitor not in results or not torch.isfinite(torch.tensor(results[monitor])):
        raise ValueError('SLG validation has no finite, evaluable bilateral hand score')
    results['slg_num_samples'] = seen
    for key, count in metric.counts.items():
        results[f'slg_valid_sequences_{key}'] = count
    print('SLG generation:', json.dumps(results), flush=True)
    return results


def build_merged_validation_loaders(args, primary_slt_loader, primary_t2p_loader):
    from datasets_merged import MERGED_DATASETS
    loaders = {}
    for name in MERGED_DATASETS:
        eval_args = copy.copy(args)
        eval_args.dataset = name
        if args.eval and args.output_dir:
            eval_args.output_dir = str(Path(args.output_dir) / name.lower())
            Path(eval_args.output_dir).mkdir(parents=True, exist_ok=True)
        if name == args.dataset:
            slt_loader, t2p_loader = primary_slt_loader, primary_t2p_loader
        else:
            slt_dataset = S2T_Dataset(test_label_paths[name], eval_args, phase='test')
            t2p_dataset = S2T_Dataset_Pose2D_KLVAE(
                test_label_paths[name], build_t2p_args(eval_args), phase='test')
            kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=args.pin_mem)
            slt_loader = DataLoader(slt_dataset, collate_fn=slt_dataset.collate_fn,
                                   sampler=torch.utils.data.SequentialSampler(slt_dataset), **kwargs)
            t2p_loader = DataLoader(t2p_dataset, collate_fn=t2p_dataset.collate_fn,
                                   sampler=torch.utils.data.SequentialSampler(t2p_dataset), **kwargs)
        loaders[name] = (eval_args, slt_loader, t2p_loader)
    return loaders


def evaluate_merged_validation(args, loaders, eval_model, raw_model):
    """Separate complete test evaluations; retain the explicit selection dataset."""
    results = {}
    raw_model.default_task = 'slt'
    for name, (eval_args, slt_loader, t2p_loader) in loaders.items():
        slt = evaluate(eval_args, slt_loader, eval_model, raw_model, phase='test')
        stats = {f'slt_{key}': value for key, value in slt.items()}
        stats.update(evaluate_t2p_loss(eval_args, t2p_loader, eval_model))
        if not args.skip_slg_generation_val:
            stats.update(evaluate_slg_generation(eval_args, t2p_loader, raw_model))
        results.update({f'test/{name.lower()}/{key}': value for key, value in stats.items()})
        if name == args.dataset:
            results.update(stats)
        print(f'{name} test BLEU-4: {slt["bleu4"]:.2f}; '
              f'checkpoint selection dataset: {args.dataset}')
    return results


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
    if args.merge_datasets:
        from datasets_merged import build_merged_training_dataset
        t2p_train = build_merged_training_dataset(args, generation=True)
    else:
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

    val_split, val_label_path = resolve_validation_split(args)
    print(f"Epoch validation split: {val_split}; labels: {val_label_path}")
    t2p_val = S2T_Dataset_Pose2D_KLVAE(
        path=val_label_path,
        args=t2p_args,
        phase=val_split,
    )
    t2p_val_loader = DataLoader(
        t2p_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=t2p_val.collate_fn,
        sampler=torch.utils.data.SequentialSampler(t2p_val),
        pin_memory=args.pin_mem,
    )

    slt_train_loader = slt_val_loader = slt_test_loader = None
    if joint or args.eval:
        if args.merge_datasets:
            slt_train = build_merged_training_dataset(args)
        else:
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
        if val_split == "dev":
            slt_dev = S2T_Dataset(
                path=val_label_path, args=args, phase="dev"
            )
            slt_val_loader = DataLoader(
                slt_dev,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=slt_dev.collate_fn,
                sampler=torch.utils.data.SequentialSampler(slt_dev),
                pin_memory=args.pin_mem,
            )
        else:
            slt_val_loader = slt_test_loader

    merged_val_loaders = None
    if args.merge_datasets:
        merged_val_loaders = build_merged_validation_loaders(
            args, slt_val_loader, t2p_val_loader)

    # ---- model ------------------------------------------------------------
    print("Creating unified model:")
    args.default_task = "t2p" if not joint else "slt"
    steps_per_epoch = (min(len(t2p_train_loader), len(slt_train_loader))
                       if joint else len(t2p_train_loader))
    args.total_steps = max(int(args.epochs * steps_per_epoch
                               / args.gradient_accumulation_steps), 1)
    geo_prefixes = ()
    model_class = Uni_Sign_Unified
    if args.use_hyperbolic:
        from models_unified_geosign import Uni_Sign_Unified_GeoSign, GEOSIGN_PREFIXES
        model_class = Uni_Sign_Unified_GeoSign
        geo_prefixes = GEOSIGN_PREFIXES
    model = model_class(args=args)
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
            + GEN_ADAPTER_PREFIXES + ("latent_pos_pe",) + geo_prefixes,
        )
    if args.finetune != "":
        # a unified checkpoint must provide everything except VAE-owned keys
        load_partial(
            model, args.finetune, "unified checkpoint",
            allowed_missing_prefixes=VAE_OWNED_PREFIXES + (
                geo_prefixes if args.allow_plain_geosign_init else ()),
        )

    if args.use_hyperbolic:
        # Geo-Sign's alpha schedule is relative to the current stage.
        # Carrying the larger OpenASL step counter into How2Sign would clamp
        # alpha to 1 and silently disable the geometric objective. A true
        # resume later restores the counter along with all optimizer states.
        model.global_step.zero_()

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
    hyp_optimizer = None
    if args.use_hyperbolic:
        from geosign_training import create_geosign_optimizers
        optimizer, hyp_optimizer = create_geosign_optimizers(args, model_without_ddp)
    else:
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
    if hyp_optimizer is not None:
        from geosign_training import patch_geosign_engine
        patch_geosign_engine(model)
    print(optimizer)

    output_dir = Path(args.output_dir)

    if args.eval:
        if utils.is_main_process():
            eval_model = _EvalShim(model_without_ddp, args)
            if args.merge_datasets:
                print(json.dumps(evaluate_merged_validation(
                    args, merged_val_loaders, eval_model, model_without_ddp), indent=2))
                return
            model_without_ddp.default_task = "slt"
            if val_split == "dev":
                print("📄 SLT dev result")
                evaluate(args, slt_val_loader, eval_model, model_without_ddp,
                         phase="dev")
            print("📄 SLT test result")
            evaluate(args, slt_test_loader, eval_model, model_without_ddp,
                     phase="test")
            print(f"📄 T2P {val_split} diffusion loss")
            evaluate_t2p_loss(args, t2p_val_loader, eval_model)
            if not args.skip_slg_generation_val:
                evaluate_slg_generation(args, t2p_val_loader, model_without_ddp)
        return

    # ---- crash recovery: full-state resume (epoch granularity) ------------
    resume_dir = None
    if args.output_dir:
        resume_dir = output_dir / "resume_state"

    start_epoch = 0
    best_metric = float("-inf")
    best_slg_metric = float("inf")
    slg_protocol = slg_validation_protocol(args)
    if args.resume == "auto" and resume_dir is not None and (resume_dir / "latest").exists():
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
        expected_sources = ['OpenASL', 'How2Sign'] if args.merge_datasets else [args.dataset]
        previous_sources = client_state.get('training_datasets', [args.dataset])
        if previous_sources != expected_sources:
            raise ValueError('Resume training datasets changed; use a fresh output directory '
                             'and --finetune for weights-only initialization')
        if hyp_optimizer is not None:
            from geosign_training import restore_geosign_optimizer
            restore_geosign_optimizer(client_state, hyp_optimizer)
        best_metric = restore_best_metric(client_state, args.dataset, val_split)
        if client_state.get('slg_validation_protocol') == slg_protocol:
            best_slg_metric = float(client_state.get('best_slg_metric', float('inf')))
        print(f"Resumed from {load_path}: continuing at epoch {start_epoch} "
              f"(best selection metric so far {best_metric:.4f})")
        if start_epoch >= args.epochs and best_metric == float("-inf"):
            print("No remaining epochs to validate on the new split. Existing "
                  "best_checkpoint.pth retains its original selection protocol; "
                  "use --eval to evaluate weights or extend --epochs to continue.")

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
                hyp_optimizer=hyp_optimizer,
            )

        if args.smoke_steps:
            print(f'Smoke completed: {args.smoke_steps} microbatches; '
                  'no epoch validation or training checkpoint was saved.')
            break

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
                "validation_split": val_split,
            }

            if not joint:
                # stage 1 selection: minimal T2P validation diffusion loss
                val_stats = evaluate_t2p_loss(args, t2p_val_loader, eval_model)
                selection = -val_stats["t2p_loss"]
                log_stats.update(val_stats)
                print(f"T2P {val_split} loss: {val_stats['t2p_loss']:.4f}")
            else:
                # stage 2/3 selection: SLT BLEU-4 on the requested split
                model_without_ddp.default_task = "slt"
                if args.merge_datasets:
                    merged_stats = evaluate_merged_validation(
                        args, merged_val_loaders, eval_model, model_without_ddp)
                    log_stats.update(merged_stats)
                    selection = merged_stats['slt_bleu4']
                else:
                    slt_stats = evaluate(
                        args, slt_val_loader, eval_model, model_without_ddp,
                        phase=val_split,
                    )
                    val_stats = evaluate_t2p_loss(args, t2p_val_loader, eval_model)
                    selection = slt_stats["bleu4"]
                    log_stats.update(
                        {f"slt_{k}": v for k, v in slt_stats.items()}
                    )
                    log_stats.update(val_stats)
                    print(f"SLT BLEU-4 ({val_split}): {slt_stats['bleu4']:.2f}  "
                          f"T2P {val_split} loss: {val_stats['t2p_loss']:.4f}")

            if not args.skip_slg_generation_val:
                slg_stats = (merged_stats if args.merge_datasets else
                             evaluate_slg_generation(args, t2p_val_loader, model_without_ddp))
                log_stats.update(slg_stats)
                monitor = f'{args.dataset.lower()}_DTW_MPJPE2D_hand'
                if slg_stats[monitor] < best_slg_metric:
                    best_slg_metric = slg_stats[monitor]
                    if args.output_dir:
                        utils.save_on_master(
                            {'model': unified_state_dict(model_without_ddp),
                             'epoch': epoch, 'dataset': args.dataset,
                             'validation_split': val_split, 'selection_metric': monitor,
                             'selection_value': best_slg_metric,
                             'slg_validation_protocol': slg_protocol},
                            output_dir / 'best_slg_checkpoint.pth',
                        )
                print(f'Best SLG {monitor}: {best_slg_metric:.6f}')

            # Split-specific W&B/log keys keep new test curves distinguishable
            # from dev values in an existing resumed run.
            log_stats.update({f"{val_split}/{key}": value
                              for key, value in list(log_stats.items())
                              if key.startswith(("slt_", "t2p_", "slg_",
                                                 f'{args.dataset.lower()}_DTW_'))})

            if selection > best_metric:
                best_metric = selection
                if args.output_dir:
                    utils.save_on_master(
                        {"model": unified_state_dict(model_without_ddp),
                         "validation_split": val_split, "dataset": args.dataset,
                         "epoch": epoch, "selection_value": selection,
                         "selection_metric": "slt_bleu4" if joint else "negative_t2p_loss"},
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
                client_state={"epoch": epoch + 1, "best_metric": best_metric,
                              'training_datasets': (['OpenASL', 'How2Sign']
                                                    if args.merge_datasets else [args.dataset]),
                              "validation_split": val_split,
                              "best_slg_metric": best_slg_metric,
                              "slg_validation_protocol": slg_protocol,
                              **({'hyp_optimizer': hyp_optimizer.state_dict()}
                                 if hyp_optimizer is not None else {})},
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
                      wandb_run=None, hyp_optimizer=None):
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
    if hyp_optimizer is not None:
        hyp_optimizer.zero_grad(set_to_none=True)

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
    gradient_skips = 0
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
            if hyp_optimizer is not None:
                from vendor.geosign.training_helpers import _discard_accumulated_step
                _discard_accumulated_step(model, hyp_optimizer)
            continue

        model.backward(total_loss)
        if hyp_optimizer is not None:
            from geosign_training import step_geosign
            if not step_geosign(model, model.module.module, hyp_optimizer):
                gradient_skips += 1
                print(f'Geo-Sign: skipped non-finite gradient update at step {step}')
                if gradient_skips > args.max_nonfinite_skips:
                    raise FloatingPointError('Too many non-finite geometric gradient updates')
                continue
        else:
            model.step()

        metric_logger.update(
            loss=loss_value,
            slt_loss=joint_out["slt_loss"].item(),
            t2p_loss=joint_out["t2p_loss"].item(),
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        geo_stats = {key: joint_out[key].item()
                     for key in ('ce_loss', 'margin_loss', 'alpha', 'curvature',
                                 'temperature', 'effective_margin', 'hyp_sim_mean')
                     if key in joint_out}
        if geo_stats:
            metric_logger.update(**geo_stats)

        if wandb_run is not None and step % print_freq == 0:
            wandb_run.log({
                "step/loss": loss_value,
                "step/slt_loss": joint_out["slt_loss"].item(),
                "step/t2p_loss": joint_out["t2p_loss"].item(),
                "step/lr": optimizer.param_groups[0]["lr"],
                "step/global_step": epoch * len(paired) + step,
                **{f'step/{key}': value for key, value in geo_stats.items()},
            })

        if args.smoke_steps and step + 1 >= args.smoke_steps:
            if hyp_optimizer is not None:
                print(f'Geo-Sign successful optimizer updates: '
                      f'{model.module.module.global_step.item()}')
            break

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
