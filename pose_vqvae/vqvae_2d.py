"""Single-part 1D-conv VQ-VAE over 2D pose sequences.

Adapted from SOKE's decoupled tokenizer (mGPT/archs/mgpt_vq.py, itself based on
T2M-GPT). SOKE runs this stack on flat 133-D SMPL-X motion features sliced into
body / left-hand / right-hand streams. Here the identical
Encoder -> Quantizer -> Decoder stack runs on one part of Uni-Sign's RTMPose
COCO-WholeBody 2D keypoints, as produced by ``datasets.load_part_kp``:
a ``[B, T, V, 3]`` sequence of (x, y, confidence) per joint, coordinates already
normalized to [-1, 1] and joints with confidence <= 0.3 zeroed out.

Adaptation differences vs. the SOKE original:
- input is ``[B, T, V, 3]``; the confidence channel is fed to the encoder so it
  can distinguish "joint at origin" from "joint filtered out"
- the decoder reconstructs coordinates only: ``[B, T, V, 2]``
- ``decode`` accepts ``[T']`` or ``[B, T']`` token tensors (the SOKE version
  hard-coded batch size 1)

Temporal behavior is unchanged: with ``down_t=2, stride_t=2`` one token covers
4 frames, so ``T`` should be a multiple of ``downsample_rate`` (pad by repeating
the last frame, mirroring Uni-Sign's collate padding).
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .resnet import Resnet1D
from .quantize import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class PoseVQVae2D(nn.Module):

    def __init__(self,
                 njoints: int,
                 quantizer: str = "ema_reset",
                 code_num: int = 512,
                 code_dim: int = 512,
                 output_emb_width: int = 512,
                 down_t: int = 2,
                 stride_t: int = 2,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 norm=None,
                 activation: str = "relu",
                 in_channels: int = 3,
                 out_channels: int = 2,
                 **kwargs) -> None:
        super().__init__()

        self.njoints = njoints
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.code_num = code_num
        self.code_dim = code_dim
        self.nfeats_in = njoints * in_channels
        self.nfeats_out = njoints * out_channels
        self.downsample_rate = stride_t ** down_t

        self.encoder = Encoder(self.nfeats_in,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(self.nfeats_out,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)
        else:
            raise ValueError(f"Unknown quantizer '{quantizer}'")

    def preprocess(self, x: Tensor) -> Tensor:
        # (B, T, V, C) -> (B, V*C, T)
        B, T = x.shape[:2]
        return x.reshape(B, T, -1).permute(0, 2, 1).contiguous()

    def postprocess(self, x: Tensor) -> Tensor:
        # (B, V*2, T) -> (B, T, V, 2)
        B, _, T = x.shape
        return x.permute(0, 2, 1).reshape(B, T, self.njoints, self.out_channels)

    def forward(self, features: Tensor,
                latent_mask: Union[Tensor, None] = None
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """features: [B, T, V, 3] -> (recon [B, T, V, 2], commit_loss, perplexity)

        latent_mask: optional [B, T // downsample_rate], 1 for latent steps that
        cover at least one real (non-padding) frame. Padded steps are excluded
        from codebook init/EMA updates, commitment loss, and perplexity.
        """
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, perplexity = self.quantizer(x_encoder, mask=latent_mask)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    @torch.no_grad()
    def encode(self, features: Tensor) -> Tensor:
        """features: [B, T, V, 3] -> token indices [B, T // downsample_rate]"""
        N = features.shape[0]
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)  # (B, code_dim, T')
        x_encoder = x_encoder.permute(0, 2, 1).contiguous()  # (B, T', C)
        x_encoder = x_encoder.view(-1, x_encoder.shape[-1])  # (B*T', C)
        code_idx = self.quantizer.quantize(x_encoder)
        return code_idx.view(N, -1)

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        """codes: [T'] or [B, T'] token indices -> poses [B, T'*rate, V, 2]"""
        if codes.dim() == 1:
            codes = codes[None]
        x_d = self.quantizer.dequantize(codes)  # (B, T', code_dim)
        x_d = x_d.permute(0, 2, 1).contiguous()  # (B, code_dim, T')
        x_decoder = self.decoder(x_d)
        return self.postprocess(x_decoder)


class Encoder(nn.Module):
    """Unchanged SOKE temporal conv encoder (input width differs per part)."""

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """Unchanged SOKE temporal conv decoder; ``input_emb_width`` is the
    reconstruction width, here njoints * 2 (xy only, no confidence)."""

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm),
                nn.Upsample(scale_factor=stride_t, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
