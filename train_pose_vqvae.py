"""Train the multi-part 2D-pose VQ-VAE (SOKE DETO stage adapted to Uni-Sign data).

Reuses Uni-Sign's S2T_Dataset / collate_fn, so the tokenizer trains on exactly
the four normalized parts (body / left / right / face_all) the SLT model sees.
Supports joint training over several datasets (default: OpenASL + How2Sign),
mirroring SOKE's multilingual DETO training.

Usage (single GPU):
    # joint OpenASL + How2Sign (default)
    python train_pose_vqvae.py --output_dir out/pose_vqvae_asl

    # single dataset
    python train_pose_vqvae.py --datasets OpenASL --output_dir out/pose_vqvae_openasl

Evaluation / model selection:
- datasets with a dev split (e.g. OpenASL) are validated on dev;
- datasets without one (How2Sign) are validated on a deterministic holdout
  carved out of their training labels — the test split is never touched;
- validation always uses deterministic (evenly spaced) frame subsampling, so
  checkpoint ranking is reproducible; the best checkpoint minimizes the mean
  of the per-dataset masked MPJPE.
"""

import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from config import train_label_paths, dev_label_paths
from datasets import S2T_Dataset
from pose_vqvae import (DEFAULT_PART_CONFIGS, MultiPartPoseVQVAE,
                        MultiPartReconLoss, PART_NJOINTS)
from pose_vqvae.data import DeterministicS2T_Dataset

PARTS = list(PART_NJOINTS.keys())


def get_args():
    p = argparse.ArgumentParser('Multi-part 2D pose VQ-VAE training')
    p.add_argument('--datasets', default='OpenASL,How2Sign',
                   help='comma-separated list out of '
                        'OpenASL,How2Sign,CSL_Daily,WLASL')
    p.add_argument('--output_dir', default='out/pose_vqvae')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--max_length', type=int, default=256,
                   help='max frames per clip (Uni-Sign default)')
    p.add_argument('--holdout_size', type=int, default=1000,
                   help='val samples held out of train for datasets w/o dev split')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', default='')
    p.add_argument('--eval_every', type=int, default=5)
    # loss weights (SOKE deto.yaml defaults)
    p.add_argument('--lambda_feature', type=float, default=1.0)
    p.add_argument('--lambda_velocity', type=float, default=0.0)
    p.add_argument('--lambda_commit', type=float, default=0.02)
    p.add_argument('--recons_loss', default='l1_smooth',
                   choices=['l1', 'l2', 'l1_smooth'])
    # codebook overrides (defaults follow SOKE: body/face 96, hands 192)
    p.add_argument('--code_num_body', type=int, default=None)
    p.add_argument('--code_num_hand', type=int, default=None)
    p.add_argument('--code_num_face', type=int, default=None)
    return p.parse_args()


def build_part_configs(args):
    cfgs = {k: dict(v) for k, v in DEFAULT_PART_CONFIGS.items()}
    if args.code_num_body:
        cfgs['body']['code_num'] = args.code_num_body
    if args.code_num_hand:
        cfgs['left']['code_num'] = args.code_num_hand
        cfgs['right']['code_num'] = args.code_num_hand
    if args.code_num_face:
        cfgs['face_all']['code_num'] = args.code_num_face
    return cfgs


def build_loaders(args):
    """Returns (train_loader, {dataset_name: eval_loader}).

    Train: ConcatDataset over all requested datasets, random frame subsampling.
    Eval: per-dataset loaders with deterministic subsampling — dev split when
    the dataset has one, otherwise a seeded holdout from its train labels
    (those samples are removed from training).
    """
    names = [n.strip() for n in args.datasets.split(',') if n.strip()]
    collate = None
    train_sets, eval_loaders = [], {}

    for name in names:
        data_args = SimpleNamespace(dataset=name, rgb_support=False,
                                    max_length=args.max_length)
        train_set = S2T_Dataset(train_label_paths[name], data_args, phase='train')
        collate = collate or train_set.collate_fn

        dev_path = dev_label_paths[name]
        if dev_path:
            eval_set = DeterministicS2T_Dataset(dev_path, data_args, phase='dev')
        else:
            # No dev split (How2Sign): hold out a deterministic, seeded slice
            # of train for model selection; the test split stays untouched.
            det_train = DeterministicS2T_Dataset(
                train_label_paths[name], data_args, phase='train')
            n_val = min(args.holdout_size, max(1, len(train_set) // 20))
            g = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(len(train_set), generator=g).tolist()
            eval_set = Subset(det_train, perm[:n_val])
            train_set = Subset(train_set, perm[n_val:])
            print(f"[{name}] no dev split: holding out {n_val} train samples "
                  f"for validation")

        train_sets.append(train_set)
        eval_loaders[name] = DataLoader(
            eval_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
        print(f"[{name}] train {len(train_set)} | eval {len(eval_set)}")

    train_all = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    train_loader = DataLoader(train_all, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate)
    return train_loader, eval_loaders


def prepare_batch(src_input, device, multiple):
    """Extract part tensors + frame mask; cast float64 -> float32; pad T to a
    multiple of the VQ downsample rate by repeating the last frame."""
    parts = {k: src_input[k].to(device, dtype=torch.float32, non_blocking=True)
             for k in PARTS}
    mask = src_input['attention_mask'].to(device, non_blocking=True)

    T = mask.shape[1]
    pad = (-T) % multiple
    if pad:
        parts = {k: torch.cat([v, v[:, -1:].expand(-1, pad, -1, -1)], dim=1)
                 for k, v in parts.items()}
        mask = torch.cat([mask, mask.new_zeros(mask.shape[0], pad)], dim=1)
    return parts, mask


@torch.no_grad()
def evaluate(model, eval_loaders, device):
    """Masked mean joint position error (normalized [-1,1] units) per part,
    computed per dataset. Selection metric: mean of per-dataset means."""
    model.eval()
    results = {}
    dataset_means = []

    for ds_name, loader in eval_loaders.items():
        err = {p: 0.0 for p in PARTS}
        cnt = {p: 0.0 for p in PARTS}
        perp = {p: 0.0 for p in PARTS}
        n_batches = 0

        for src_input, _ in loader:
            parts, mask = prepare_batch(src_input, device, model.downsample_rate)
            recons, _, _, stats = model(parts, frame_mask=mask)
            for p in PARTS:
                gt = parts[p]
                valid = (gt[..., 2] > 0).float() * mask.float().unsqueeze(-1)
                dist = torch.linalg.norm(recons[p] - gt[..., :2], dim=-1)  # [B,T,V]
                err[p] += (dist * valid).sum().item()
                cnt[p] += valid.sum().item()
                perp[p] += float(stats[p]['perplexity'])
            n_batches += 1

        part_mpjpe = {p: err[p] / max(cnt[p], 1.0) for p in PARTS}
        for p in PARTS:
            results[f'{ds_name}/{p}/mpjpe'] = part_mpjpe[p]
            results[f'{ds_name}/{p}/perplexity'] = perp[p] / max(n_batches, 1)
        ds_mean = float(np.mean(list(part_mpjpe.values())))
        results[f'{ds_name}/mpjpe_mean'] = ds_mean
        dataset_means.append(ds_mean)

    results['mpjpe_mean'] = float(np.mean(dataset_means))
    return results


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    part_configs = build_part_configs(args)
    model = MultiPartPoseVQVAE(part_configs).to(device)
    criterion = MultiPartReconLoss(lambda_feature=args.lambda_feature,
                                   lambda_velocity=args.lambda_velocity,
                                   lambda_commit=args.lambda_commit,
                                   recons_loss=args.recons_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)

    start_epoch, best = 0, float('inf')
    if args.resume:
        # Quantizer EMA state (code_sum/code_count/initted) is stored in
        # buffers, so load_state_dict restores it and no codebook re-init
        # happens on the first resumed batch.
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt.get('best', best)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_loader, eval_loaders = build_loaders(args)
    print(f"train batches/epoch: {len(train_loader)} "
          f"| eval sets: {list(eval_loaders)}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for it, (src_input, _) in enumerate(train_loader):
            parts, mask = prepare_batch(src_input, device, model.downsample_rate)

            recons, commit, _, stats = model(parts, frame_mask=mask)
            loss, logs = criterion(recons, parts, mask, commit)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += logs['total']
            if it % 50 == 0:
                perp_str = ' '.join(
                    f"{p}:{float(stats[p]['perplexity']):.1f}" for p in PARTS)
                print(f"epoch {epoch:3d} it {it:5d}/{len(train_loader)} "
                      f"loss {logs['total']:.4f} commit {logs['commit']:.4f} "
                      f"| perplexity {perp_str}")

        scheduler.step()
        print(f"epoch {epoch:3d} done | mean loss {running / max(len(train_loader), 1):.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

        # Evaluate BEFORE checkpointing so 'best' saved in checkpoint_last.pth
        # is never stale (resuming from it must not clobber a better best.pth).
        do_eval = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        improved = False
        if do_eval:
            results = evaluate(model, eval_loaders, device)
            for ds_name in eval_loaders:
                msg = ' '.join(f"{p}:{results[f'{ds_name}/{p}/mpjpe']:.4f}"
                               for p in PARTS)
                print(f"[eval] epoch {epoch:3d} {ds_name}: mean "
                      f"{results[f'{ds_name}/mpjpe_mean']:.4f} | {msg}")
            print(f"[eval] epoch {epoch:3d} overall mpjpe(norm) "
                  f"{results['mpjpe_mean']:.4f}")
            if results['mpjpe_mean'] < best:
                best = results['mpjpe_mean']
                improved = True

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch,
                 'best': best, 'part_configs': part_configs, 'args': vars(args)}
        torch.save(state, os.path.join(args.output_dir, 'checkpoint_last.pth'))
        if improved:
            torch.save(state, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f"[eval] new best ({best:.4f}) saved")

    print(f"Training finished. Best masked MPJPE (normalized units): {best:.4f}")


if __name__ == '__main__':
    main()
