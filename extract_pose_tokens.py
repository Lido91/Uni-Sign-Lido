"""Tokenize Uni-Sign 2D pose clips with a trained multi-part VQ-VAE.

Analog of SOKE's scripts/get_motion_code.py: run after train_pose_vqvae.py to
dump one pickle of discrete tokens per clip, ready for autoregressive
(text-to-sign) generator training.

Usage:
    python extract_pose_tokens.py --dataset OpenASL --split train \
        --checkpoint out/pose_vqvae_openasl/checkpoint_best.pth \
        --out_dir data/OpenASL/pose-tokens

Output per clip: <out_dir>/<clip-name>.pkl containing
    {
        'body':      np.int64 [T'],   # T' = ceil(valid_frames / 4)
        'left':      np.int64 [T'],
        'right':     np.int64 [T'],
        'face_all':  np.int64 [T'],
        'num_frames': int,            # valid frames actually encoded
        'downsample_rate': int,
    }

Clips longer than --max_length frames are subsampled with evenly spaced
(deterministic) indices, so extraction is reproducible; raise --max_length if
you want (nearly) full clips tokenized.
"""

import argparse
import os
import pickle
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from config import train_label_paths, dev_label_paths, test_label_paths
from pose_vqvae import MultiPartPoseVQVAE, PART_NJOINTS
from pose_vqvae.data import DeterministicS2T_Dataset

PARTS = list(PART_NJOINTS.keys())
LABEL_PATHS = {'train': train_label_paths, 'dev': dev_label_paths,
               'test': test_label_paths}


def get_args():
    p = argparse.ArgumentParser('Extract 2D pose VQ tokens')
    p.add_argument('--dataset', default='OpenASL',
                   choices=['OpenASL', 'How2Sign', 'CSL_Daily', 'WLASL'])
    p.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = MultiPartPoseVQVAE(ckpt.get('part_configs'))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    rate = model.downsample_rate

    label_path = LABEL_PATHS[args.split][args.dataset]
    if not label_path:
        raise ValueError(f"No {args.split} labels for {args.dataset}")
    data_args = SimpleNamespace(dataset=args.dataset, rgb_support=False,
                                max_length=args.max_length)
    dataset = DeterministicS2T_Dataset(label_path, data_args, phase=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    done, skipped = 0, 0
    for src_input, _ in loader:
        names = src_input['name_batch']
        lengths = src_input['src_length_batch']
        parts = {k: src_input[k].to(device, dtype=torch.float32) for k in PARTS}

        for i, name in enumerate(names):
            out_path = os.path.join(
                args.out_dir, os.path.splitext(name)[0] + '.pkl')
            if os.path.exists(out_path) and not args.overwrite:
                skipped += 1
                continue

            L = int(lengths[i])
            pad = (-L) % rate
            sample = {}
            for part in PARTS:
                x = parts[part][i:i + 1, :L]
                if pad:
                    x = torch.cat([x, x[:, -1:].expand(-1, pad, -1, -1)], dim=1)
                codes = model.vaes[part].encode(x)  # [1, T']
                sample[part] = codes[0].cpu().numpy().astype('int64')
            sample['num_frames'] = L
            sample['downsample_rate'] = rate

            with open(out_path, 'wb') as f:
                pickle.dump(sample, f)
            done += 1

        if (done + skipped) % 2000 < args.batch_size:
            print(f"processed {done + skipped}/{len(dataset)} "
                  f"(written {done}, skipped {skipped})")

    print(f"Finished: {done} written, {skipped} skipped -> {args.out_dir}")


if __name__ == '__main__':
    main()
