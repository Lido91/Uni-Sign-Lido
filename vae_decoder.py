"""Frozen decode-only port of the MoMADiff KL-VAE (counterpart of vae_encoder).

Loads only ``decoder`` and ``post_quant_conv`` from a trained MoMADiff
AutoencoderKL checkpoint (``{'net': state_dict}`` or Lightning format) and is
permanently frozen: the unified model uses it purely to render generated
latents back into motion features.

Input:
    latents ``[B, L, 512]``
Output:
    motion features ``[B, 4*L, input_emb_width]`` (138 for the joint
    How2Sign+OpenASL pose2d VAE) in the z-scored representation space.

Module attribute names mirror MoMADiff ``models/klvae/autoencoder.py``
(``decoder.model.*`` / ``post_quant_conv.*``) so the checkpoint loads with
``strict=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn

from vae_encoder import (
    ARCH_PARAMS,
    _extract_autoencoder_state,
    _Resnet1D,
)


class _Decoder(nn.Module):
    """Mirror of MoMADiff models/klvae/encdec.py Decoder."""

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
        norm=None,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU(),
        ]
        for _ in range(down_t):
            blocks.append(
                nn.Sequential(
                    _Resnet1D(
                        width,
                        depth,
                        dilation_growth_rate,
                        reverse_dilation=True,
                        activation=activation,
                        norm=norm,
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(width, width, 3, 1, 1),
                )
            )
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FrozenKlVaeDecoder(nn.Module):
    """MoMADiff KL-VAE ``post_quant_conv`` + decoder, decode-only."""

    def __init__(self, **overrides: object) -> None:
        super().__init__()
        params = {**ARCH_PARAMS, **overrides}
        output_emb_width = int(params["output_emb_width"])
        self.latent_dim = output_emb_width
        self.output_dim = int(params["input_emb_width"])
        self.decoder = _Decoder(
            self.output_dim,
            output_emb_width,
            int(params["down_t"]),
            int(params["stride_t"]),
            int(params["width"]),
            int(params["depth"]),
            int(params["dilation_growth_rate"]),
            activation=str(params["activation"]),
            norm=params["norm"],
        )
        self.post_quant_conv = nn.Conv1d(output_emb_width, output_emb_width, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"decoder input must have shape [B,L,{self.latent_dim}], "
                f"got {tuple(z.shape)}"
            )
        x = z.permute(0, 2, 1).to(self.post_quant_conv.weight.dtype)
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)


def posterior_sample(moments: torch.Tensor) -> torch.Tensor:
    """Reparameterized sample from posterior moments ``[B,L,2*latent]``.

    Mirrors MoMADiff DiagonalGaussianDistribution.sample():
    mean + exp(0.5 * clamp(logvar, -30, 20)) * eps.
    """
    if moments.ndim != 3 or moments.shape[-1] % 2 != 0:
        raise ValueError(
            f"posterior moments must have shape [B,L,2*latent], "
            f"got {tuple(moments.shape)}"
        )
    mean, logvar = torch.chunk(moments, 2, dim=-1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


def load_frozen_klvae_decoder(
    checkpoint_path: Union[str, Path],
    **overrides: object,
) -> FrozenKlVaeDecoder:
    """Load decoder weights, then freeze and switch to eval permanently."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KL-VAE checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _extract_autoencoder_state(checkpoint)
    decoder_state = {
        key: value
        for key, value in state.items()
        if key.startswith(("decoder.", "post_quant_conv."))
    }
    if not decoder_state:
        raise ValueError(
            f"checkpoint contains no decoder/post_quant_conv weights: {path}"
        )

    model = FrozenKlVaeDecoder(**overrides)
    model.load_state_dict(decoder_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
