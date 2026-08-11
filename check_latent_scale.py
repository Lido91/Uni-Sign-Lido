"""Diagnostic: measure the frozen KL-VAE latent scale on real batches.

The diffusion process assumes roughly unit-scale data. Run this once to see
whether the posterior-sampled latents need a scaling factor (Stable
Diffusion's 0.18215 trick):

    python check_latent_scale.py --dataset OpenASL \
        --vae_checkpoint .../kl_vae_h2s_openasl_pose2d/net_best_hand.pth

Reads ~20 batches from the train split, prints per-dim mean/std/absmax.
Interpretation: per-dim std in [0.5, 2] -> fine as-is; far outside -> add a
latent scaling factor before the diffusion head (and rerun stage 1).
"""

import os

import torch
from torch.utils.data import DataLoader

from config import train_label_paths
from datasets_pose2d_klvae import S2T_Dataset_Pose2D_KLVAE
from models_unified import Uni_Sign_Unified
from train_unified import get_unified_args_parser


def main():
    parser = get_unified_args_parser()
    parser.add_argument("--save_latent_stats", default="",
                        help="Directory to write latent_mean.npy / "
                             "latent_std.npy (per-dim, for "
                             "--latent_mean_path/--latent_std_path).")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Batches to analyse (use ~200 when saving).")
    args = parser.parse_args()
    args.default_task = "t2p"
    # measure the RAW latent scale even if stats already exist
    args.latent_mean_path = ""
    args.latent_std_path = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Uni_Sign_Unified(args=args).to(device).eval()

    data = S2T_Dataset_Pose2D_KLVAE(
        path=train_label_paths[args.dataset], args=args, phase="train"
    )
    print(data)
    loader = DataLoader(
        data,
        batch_size=8,
        num_workers=args.num_workers,
        collate_fn=data.collate_fn,
        shuffle=False,
    )

    rows = []
    with torch.no_grad():
        for index, (src_input, _tgt_input) in enumerate(loader):
            for key, value in src_input.items():
                if isinstance(value, torch.Tensor):
                    src_input[key] = value.float().to(device)
            latents, latent_lens = model.encode_gt_latents(src_input)
            for b, length in enumerate(latent_lens.tolist()):
                rows.append(latents[b, :length].reshape(-1, latents.shape[-1]))
            if index >= args.num_batches:
                break

    z = torch.cat(rows)
    per_dim_std = z.std(dim=0)
    print(f"tokens analysed:  {z.shape[0]}")
    print(f"global mean:      {z.mean().item():.4f}")
    print(f"global std:       {z.std().item():.4f}")
    print(f"per-dim std:      min={per_dim_std.min().item():.4f}  "
          f"median={per_dim_std.median().item():.4f}  "
          f"max={per_dim_std.max().item():.4f}")
    print(f"absmax:           {z.abs().max().item():.2f}")
    print(f"implied initial diffusion loss (SSE/token) ~= "
          f"{(z.var(dim=0).sum()).item():.1f}")

    if args.save_latent_stats:
        import numpy as np
        import os as _os
        _os.makedirs(args.save_latent_stats, exist_ok=True)
        mean_path = _os.path.join(args.save_latent_stats, "latent_mean.npy")
        std_path = _os.path.join(args.save_latent_stats, "latent_std.npy")
        np.save(mean_path, z.mean(dim=0).cpu().numpy())
        np.save(std_path, per_dim_std.clamp(min=1e-3).cpu().numpy())
        print(f"saved {mean_path} and {std_path}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
