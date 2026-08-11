"""Multi-part (decoupled) VQ-VAE over Uni-Sign 2D pose parts.

SOKE's DETO tokenizer decouples SMPL-X motion into body / left-hand /
right-hand streams, each with its own VQ-VAE and codebook. Uni-Sign's loader
(``datasets.load_part_kp``) already decomposes RTMPose 133 keypoints into FOUR
parts, so this manager runs one :class:`PoseVQVae2D` per part:

    body      [B, T,  9, 3]   bbox-normalized to [-1, 1]
    left      [B, T, 21, 3]   wrist-relative, scaled by body bbox
    right     [B, T, 21, 3]   wrist-relative, scaled by body bbox
    face_all  [B, T, 18, 3]   anchor-relative, scaled by body bbox

Interfaces mirror SOKE's ``VQArchitectureManager`` but operate on part dicts
instead of slices of a flat feature vector (the parts here live in different
local coordinate frames, so there is no meaningful flat concatenation).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqvae_2d import PoseVQVae2D

# Joints per part, matching datasets.load_part_kp output.
PART_NJOINTS = {
    'body': 9,
    'left': 21,
    'right': 21,
    'face_all': 18,
}

# Codebook sizes follow SOKE's DETO ratios (body re96, hands hand192);
# everything else matches configs/vq/*.yaml in the SOKE repo.
_COMMON = dict(
    quantizer='ema_reset',
    code_dim=512,
    output_emb_width=512,
    down_t=2,
    stride_t=2,
    width=512,
    depth=3,
    dilation_growth_rate=3,
    norm=None,
    activation='relu',
)

DEFAULT_PART_CONFIGS = {
    'body': dict(_COMMON, code_num=96),
    'left': dict(_COMMON, code_num=192),
    'right': dict(_COMMON, code_num=192),
    'face_all': dict(_COMMON, code_num=96),
}


class MultiPartPoseVQVAE(nn.Module):

    def __init__(self, part_configs: Optional[Dict[str, dict]] = None):
        super().__init__()
        part_configs = part_configs or DEFAULT_PART_CONFIGS

        self.vaes = nn.ModuleDict()
        for part, cfg in part_configs.items():
            if part not in PART_NJOINTS:
                raise ValueError(
                    f"Unknown part '{part}'. Valid parts: {list(PART_NJOINTS)}")
            self.vaes[part] = PoseVQVae2D(njoints=PART_NJOINTS[part], **cfg)

        rates = {p: v.downsample_rate for p, v in self.vaes.items()}
        if len(set(rates.values())) != 1:
            raise ValueError(f"Parts must share one downsample rate, got {rates}")
        self.downsample_rate = next(iter(rates.values()))
        self._frozen = False

        print(f"{'=' * 60}\nMultiPartPoseVQVAE initialized")
        for part, vae in self.vaes.items():
            print(f"  {part:9s}: {vae.njoints:2d} joints | "
                  f"codebook {vae.code_num:4d} x {vae.code_dim} | "
                  f"1 token / {vae.downsample_rate} frames")
        print('=' * 60)

    @property
    def parts(self):
        return list(self.vaes.keys())

    def latent_mask_from_frames(self, frame_mask: torch.Tensor) -> torch.Tensor:
        """[B, T] frame mask -> [B, T // rate] latent-step mask. A latent step
        is valid when at least one of its frames is real (non-padding)."""
        B, T = frame_mask.shape
        r = self.downsample_rate
        assert T % r == 0, f"T={T} must be a multiple of downsample rate {r}"
        return frame_mask.reshape(B, T // r, r).amax(dim=-1)

    def forward(self, features: Dict[str, torch.Tensor],
                frame_mask: Optional[torch.Tensor] = None):
        """features: dict of [B, T, V, 3] per part (T multiple of downsample_rate).
        frame_mask: optional [B, T], 1 for real frames, 0 for padding; padded
        latent steps are then excluded from codebook statistics/commit loss.

        Returns (recons, commit_loss, perplexity, per_part_stats) where recons
        is a dict of [B, T, V, 2].
        """
        latent_mask = None
        if frame_mask is not None:
            latent_mask = self.latent_mask_from_frames(frame_mask)

        recons, stats = {}, {}
        commit_total, perp_total = 0.0, 0.0
        for part, vae in self.vaes.items():
            recon, commit, perp = vae(features[part], latent_mask=latent_mask)
            recons[part] = recon
            stats[part] = {'commit': commit, 'perplexity': perp}
            commit_total = commit_total + commit
            perp_total = perp_total + perp
        return recons, commit_total, perp_total, stats

    @torch.no_grad()
    def encode(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """dict of [B, T, V, 3] -> dict of token indices [B, T // rate]"""
        return {part: vae.encode(features[part]) for part, vae in self.vaes.items()}

    @torch.no_grad()
    def decode(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """dict of [T'] / [B, T'] tokens -> dict of poses [B, T'*rate, V, 2]"""
        return {part: vae.decode(tokens[part]) for part, vae in self.vaes.items()}

    def get_codebook_sizes(self) -> Dict[str, int]:
        return {part: vae.code_num for part, vae in self.vaes.items()}

    def freeze(self):
        """Freeze all part VQ-VAEs (for downstream LM training, as in SOKE).

        Persistent: a later ``.train()`` on this module or any parent keeps the
        VAEs in eval mode, so EMA codebook updates (gated on ``self.training``
        inside the quantizers) can never run on a frozen tokenizer.
        """
        self._frozen = True
        for vae in self.vaes.values():
            vae.eval()
            for p in vae.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, '_frozen', False):
            for vae in self.vaes.values():
                vae.eval()
        return self


class MultiPartReconLoss(nn.Module):
    """Confidence-masked reconstruction loss for 2D pose parts.

    SOKE's DETO loss is plain smooth-L1 on the full feature vector; 2D pose
    needs masking instead, because the loader zeroes any joint with detection
    confidence <= 0.3 and the collate function pads time by repeating the last
    frame. A joint contributes to the loss only when
        confidence > 0  AND  its frame is inside the true sequence length.
    """

    def __init__(self,
                 lambda_feature: float = 1.0,
                 lambda_velocity: float = 0.0,
                 lambda_commit: float = 0.02,
                 recons_loss: str = 'l1_smooth',
                 part_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.lambda_feature = lambda_feature
        self.lambda_velocity = lambda_velocity
        self.lambda_commit = lambda_commit
        self.part_weights = part_weights or {}
        if recons_loss == 'l1':
            self._recon = lambda a, b: F.l1_loss(a, b, reduction='none')
        elif recons_loss == 'l2':
            self._recon = lambda a, b: F.mse_loss(a, b, reduction='none')
        elif recons_loss == 'l1_smooth':
            self._recon = lambda a, b: F.smooth_l1_loss(a, b, reduction='none')
        else:
            raise ValueError(f"Unknown recons_loss '{recons_loss}'")

    def forward(self,
                recons: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                frame_mask: torch.Tensor,
                commit_loss: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        recons:     dict of [B, T, V, 2] reconstructions
        targets:    dict of [B, T, V, 3] loader outputs (x, y, confidence)
        frame_mask: [B, T] 1 for real frames, 0 for collate/temporal padding
        """
        device = frame_mask.device
        total = torch.zeros((), device=device)
        logs = {}

        for part, pred in recons.items():
            gt = targets[part]
            T = min(pred.shape[1], gt.shape[1], frame_mask.shape[1])
            pred, gt = pred[:, :T], gt[:, :T]
            gt_xy, conf = gt[..., :2], gt[..., 2]

            # [B, T, V]: joint observed and frame not padding
            valid = (conf > 0).float() * frame_mask[:, :T].float().unsqueeze(-1)
            denom = valid.sum() * 2 + 1e-6

            feat = (self._recon(pred, gt_xy).sum(-1) * valid).sum() / denom

            part_loss = self.lambda_feature * feat
            logs[f'{part}/recon'] = feat.item()

            if self.lambda_velocity > 0 and T > 1:
                v_pred = pred[:, 1:] - pred[:, :-1]
                v_gt = gt_xy[:, 1:] - gt_xy[:, :-1]
                v_valid = valid[:, 1:] * valid[:, :-1]
                v_denom = v_valid.sum() * 2 + 1e-6
                vel = (self._recon(v_pred, v_gt).sum(-1) * v_valid).sum() / v_denom
                part_loss = part_loss + self.lambda_velocity * vel
                logs[f'{part}/velocity'] = vel.item()

            total = total + self.part_weights.get(part, 1.0) * part_loss

        total = total + self.lambda_commit * commit_loss
        logs['commit'] = float(commit_loss)
        logs['total'] = total.item()
        return total, logs
