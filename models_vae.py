"""Uni-Sign mT5 path with a trainable KL-VAE encoder instead of ST-GCN.

This is an additive model.  The original ``models.Uni_Sign`` implementation is
not imported, subclassed, or modified.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from vae_encoder import (
    DOWNSAMPLE_FACTOR,
    LATENT_DIM,
    load_trainable_klvae_encoder,
    posterior_mean,
)


def build_latent_attention_mask(
    frame_lengths: torch.Tensor,
    latent_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert valid frame lengths into a ``[B,T/4]`` VAE-token mask."""
    latent_lengths = torch.div(
        frame_lengths.to(device=device, dtype=torch.long),
        DOWNSAMPLE_FACTOR,
        rounding_mode="floor",
    )
    latent_lengths = latent_lengths.clamp(max=latent_steps)
    positions = torch.arange(latent_steps, device=device).unsqueeze(0)
    return (positions < latent_lengths.unsqueeze(1)).long()


class Uni_Sign_VAE(nn.Module):
    """Original Uni-Sign generative objective with a different motion encoder."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.lang = "English"

        self.vae_encoder = load_trainable_klvae_encoder(args.vae_checkpoint)
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(args.mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(
            args.mt5_path, legacy=False
        )
        mt5_hidden = int(self.mt5_model.config.d_model)

        # This is the direct replacement for Uni-Sign's Linear(1024, 768)
        # pose_proj.  It is not a LLaVA two-layer projector.
        self.pose_proj = nn.Linear(LATENT_DIM, mt5_hidden)
        nn.init.trunc_normal_(self.pose_proj.weight, std=0.02)
        if self.pose_proj.bias is not None:
            nn.init.zeros_(self.pose_proj.bias)

    def encode_motion(
        self, src_input: dict[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion = src_input["motion"]
        frame_lengths = src_input["src_length_batch"]
        if not isinstance(motion, torch.Tensor):
            raise TypeError("src_input['motion'] must be a tensor")
        if not isinstance(frame_lengths, torch.Tensor):
            raise TypeError("src_input['src_length_batch'] must be a tensor")

        moments = self.vae_encoder(motion)
        motion_features = posterior_mean(moments)
        motion_embeds = self.pose_proj(motion_features)
        latent_mask = build_latent_attention_mask(
            frame_lengths,
            latent_steps=motion_embeds.shape[1],
            device=motion_embeds.device,
        )
        return motion_embeds, latent_mask

    def forward(
        self,
        src_input: dict[str, object],
        tgt_input: dict[str, list[str]],
    ) -> dict[str, torch.Tensor]:
        motion_embeds, latent_mask = self.encode_motion(src_input)
        captions = tgt_input["gt_sentence"]

        prefix_token = self.mt5_tokenizer(
            [self.args.prompt] * len(captions),
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(motion_embeds.device)
        prefix_embeds = self.mt5_model.encoder.embed_tokens(
            prefix_token["input_ids"]
        )
        inputs_embeds = torch.cat([prefix_embeds, motion_embeds], dim=1)
        attention_mask = torch.cat(
            [prefix_token["attention_mask"], latent_mask], dim=1
        )

        target = self.mt5_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_target_length,
        )
        labels = target["input_ids"]
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100
        labels = labels.to(inputs_embeds.device)

        output = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        flat_logits = output.logits.reshape(-1, output.logits.shape[-1])
        flat_labels = labels.reshape(-1)
        loss = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing,
            ignore_index=-100,
        )(flat_logits, flat_labels)
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        pre_compute_item: dict[str, torch.Tensor],
        max_new_tokens: int,
        num_beams: int,
    ) -> torch.Tensor:
        return self.mt5_model.generate(
            inputs_embeds=pre_compute_item["inputs_embeds"],
            attention_mask=pre_compute_item["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
