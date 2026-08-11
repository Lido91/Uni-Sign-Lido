"""Trainable encode-only port of the MoMADiff KL-VAE.

This module is intentionally independent from the original Uni-Sign pose/ST-GCN
implementation.  It loads only ``encoder`` and ``quant_conv`` from a trained
MoMADiff AutoencoderKL checkpoint.  The decoder is not needed for sign-to-text
training.

Input:
    normalized motion ``[B, T, input_emb_width]`` (default 133; the joint
    How2Sign+OpenASL pose2d VAE uses 138 = 69 joints x [x, y])
Output:
    posterior moments ``[B, T // 4, 1024]`` (mean | log-variance)

Unlike the offline latent baseline, the returned encoder remains trainable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import torch
from torch import nn


FEATURE_DIM = 133
LATENT_DIM = 512
MOMENT_DIM = LATENT_DIM * 2
DOWNSAMPLE_FACTOR = 4

ARCH_PARAMS = {
    "input_emb_width": FEATURE_DIM,
    "output_emb_width": LATENT_DIM,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "norm": None,
}


class _ResConv1DBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_state: int,
        dilation: int = 1,
        activation: str = "relu",
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        if norm is not None:
            raise NotImplementedError("the How2Sign KL-VAE uses norm=None")
        if activation != "relu":
            raise NotImplementedError("the How2Sign KL-VAE uses ReLU")
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, dilation, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return x + residual


class _Resnet1D(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_depth: int,
        dilation_growth_rate: int = 1,
        reverse_dilation: bool = True,
        activation: str = "relu",
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        blocks = [
            _ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks.reverse()
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width: int,
        output_emb_width: int,
        down_t: int,
        stride_t: int,
        width: int,
        depth: int,
        dilation_growth_rate: int,
        activation: str = "relu",
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        blocks: list[nn.Module] = [
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU(),
        ]
        for _ in range(down_t):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                    _Resnet1D(
                        width,
                        depth,
                        dilation_growth_rate,
                        activation=activation,
                        norm=norm,
                    ),
                )
            )
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TrainableKlVaeEncoder(nn.Module):
    """MoMADiff KL-VAE encoder and posterior-moment projection."""

    def __init__(self, **overrides: object) -> None:
        super().__init__()
        params = {**ARCH_PARAMS, **overrides}
        output_emb_width = int(params["output_emb_width"])
        self.input_dim = int(params["input_emb_width"])
        self.encoder = _Encoder(
            self.input_dim,
            output_emb_width,
            int(params["down_t"]),
            int(params["stride_t"]),
            int(params["width"]),
            int(params["depth"]),
            int(params["dilation_growth_rate"]),
            activation=str(params["activation"]),
            norm=params["norm"],
        )
        self.quant_conv = nn.Conv1d(output_emb_width, 2 * output_emb_width, 1)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[-1] != self.input_dim:
            raise ValueError(
                f"VAE input must have shape [B,T,{self.input_dim}], "
                f"got {tuple(motion.shape)}"
            )
        # follow the module weight dtype so fp32 and bf16/fp16 training both work
        features = self.encoder(
            motion.permute(0, 2, 1).to(self.quant_conv.weight.dtype)
        )
        moments = self.quant_conv(features)
        return moments.permute(0, 2, 1)


def posterior_mean(moments: torch.Tensor) -> torch.Tensor:
    """Return the deterministic 512D mean used by sign-to-text training."""
    if moments.ndim != 3 or moments.shape[-1] != MOMENT_DIM:
        raise ValueError(
            f"posterior moments must have shape [B,T',{MOMENT_DIM}], "
            f"got {tuple(moments.shape)}"
        )
    mean, _ = torch.chunk(moments, 2, dim=-1)
    return mean


def _extract_autoencoder_state(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    if "net" in checkpoint:
        state = checkpoint["net"]
        if isinstance(state, Mapping):
            return dict(state)
    if "state_dict" in checkpoint:
        lightning_state = checkpoint["state_dict"]
        if isinstance(lightning_state, Mapping):
            state = {
                key[len("net.") :]: value
                for key, value in lightning_state.items()
                if isinstance(key, str) and key.startswith("net.")
            }
            if state:
                return state
    raise ValueError(
        "unsupported KL-VAE checkpoint: expected {'net': ...} or Lightning "
        "{'state_dict': {'net.*': ...}}"
    )


def load_trainable_klvae_encoder(
    checkpoint_path: Union[str, Path],
    **overrides: object,
) -> TrainableKlVaeEncoder:
    """Load encoder weights without freezing parameters or switching to eval.

    ``overrides`` are forwarded to :class:`TrainableKlVaeEncoder`, e.g.
    ``input_emb_width=138`` for the joint How2Sign+OpenASL pose2d VAE.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KL-VAE checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"KL-VAE checkpoint must contain a mapping: {path}")

    state = _extract_autoencoder_state(checkpoint)
    encoder_state = {
        key: value
        for key, value in state.items()
        if key.startswith(("encoder.", "quant_conv."))
    }
    if not encoder_state:
        raise ValueError(f"checkpoint contains no encoder/quant_conv weights: {path}")

    model = TrainableKlVaeEncoder(**overrides)
    model.load_state_dict(encoder_state, strict=True)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model
