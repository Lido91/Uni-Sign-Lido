"""Uni-Sign mT5 path for cached three-part SOKE VQ embeddings."""

from __future__ import annotations

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from datasets_vq_embeddings import (
    VQ_EMBEDDING_DIM,
    VQ_NUM_PARTS,
    VQ_PART_NAMES,
)


class ThreePartVQAdapter(nn.Module):
    """Fuse body/left/right 512D codebook vectors into one mT5 token."""

    def __init__(
        self,
        output_dim: int,
        part_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_dim < 1 or part_hidden_dim < 1:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("adapter dropout must be in [0, 1)")

        self.part_projections = nn.ModuleDict(
            {
                name: nn.Linear(VQ_EMBEDDING_DIM, part_hidden_dim)
                for name in VQ_PART_NAMES
            }
        )
        self.part_bias = nn.Parameter(
            torch.zeros(VQ_NUM_PARTS, part_hidden_dim)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fusion_projection = nn.Linear(
            VQ_NUM_PARTS * part_hidden_dim, output_dim
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 4:
            raise ValueError(
                f"VQ embeddings must be [B,T,3,512], got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[2:] != (VQ_NUM_PARTS, VQ_EMBEDDING_DIM):
            raise ValueError(
                f"VQ embeddings must end in [{VQ_NUM_PARTS},{VQ_EMBEDDING_DIM}], "
                f"got {tuple(embeddings.shape)}"
            )

        projected_parts = []
        for index, name in enumerate(VQ_PART_NAMES):
            projected = self.part_projections[name](embeddings[:, :, index])
            projected = self.activation(projected + self.part_bias[index])
            projected_parts.append(projected)
        fused = torch.cat(projected_parts, dim=-1)
        return self.fusion_projection(self.dropout(fused))


class Uni_Sign_VQ_Embedding(nn.Module):
    """Translate cached SOKE VQ embeddings with the original mT5 objective."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.lang = "English"
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(args.mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(
            args.mt5_path, legacy=False
        )
        self.text_tokenizer = self.mt5_tokenizer
        mt5_hidden = int(self.mt5_model.config.d_model)
        self.vq_adapter = ThreePartVQAdapter(
            output_dim=mt5_hidden,
            part_hidden_dim=args.vq_part_hidden_dim,
            dropout=args.vq_adapter_dropout,
        )

    def encode_motion(
        self, src_input: dict[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = src_input.get("vq_embeddings")
        attention_mask = src_input.get("attention_mask")
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("src_input['vq_embeddings'] must be a tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("src_input['attention_mask'] must be a tensor")
        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "VQ attention mask must match [B,T]: "
                f"mask={tuple(attention_mask.shape)}, "
                f"embeddings={tuple(embeddings.shape)}"
            )
        return self.vq_adapter(embeddings), attention_mask.long()

    def forward(
        self,
        src_input: dict[str, object],
        tgt_input: dict[str, list[str]],
    ) -> dict[str, torch.Tensor]:
        motion_embeds, motion_mask = self.encode_motion(src_input)
        captions = tgt_input["gt_sentence"]

        prompt_tokens = self.mt5_tokenizer(
            [self.args.prompt] * len(captions),
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(motion_embeds.device)
        prompt_embeds = self.mt5_model.encoder.embed_tokens(
            prompt_tokens["input_ids"]
        )
        inputs_embeds = torch.cat([prompt_embeds, motion_embeds], dim=1)
        attention_mask = torch.cat(
            [prompt_tokens["attention_mask"], motion_mask], dim=1
        )

        targets = self.mt5_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_target_length,
        )
        labels = targets["input_ids"]
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100
        labels = labels.to(inputs_embeds.device)

        output = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        logits = output.logits.reshape(-1, output.logits.shape[-1])
        flat_labels = labels.reshape(-1)
        if getattr(self.args, "compute_fp32_loss", False):
            logits = logits.float()
        loss = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing,
            ignore_index=-100,
        )(logits, flat_labels)
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
