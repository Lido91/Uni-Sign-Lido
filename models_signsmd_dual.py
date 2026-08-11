"""Dual-stream SignSMD fusion (experiment H-dual, option B of the fusion
design — see doc/signsmd_dual_stream_fusion_plan.md).

Additive subclass of the ORIGINAL ``models.Uni_Sign``.  Where early fusion
(``models_signsmd.Uni_Sign_SignSMD``) concatenates SMD text tokens with the
pose embeddings at the mT5 encoder INPUT, this model encodes the two
modalities in SEPARATE passes of the SAME encoder and fuses only the
resulting hidden states:

    pose stream:  [translate prompt][ST-GCN pose embeddings]  -> H_P
    SMD stream:   ["Motion description: "][SMD text tokens]   -> H_S
    fusion:       [H_P ; scale * H_S]  -> mT5 decoder

The pose stream is byte-identical to the baseline encoder input, so the
pretrained prompt-pose adjacency is preserved and any baseline mT5
Uni-Sign checkpoint warm-starts it unchanged.  All cross-modal interaction
moves to decoder cross-attention, which has no positional bias over the
encoder axis, so concatenating two independently encoded segments is
well-formed.

Fusion variants (--fusion_gate):

    none    H_S is scaled by the fixed hyperparameter --fusion_alpha
            (default 1.0 = plain concat).  ZERO new parameters: strict=True
            checkpoint round-trip with the baseline in both directions.
    global  H_S is scaled by sigmoid(a) with a single learnable logit a.
    sample  per-sample scale sigmoid(W[pool(H_P); pool(H_S)]) (masked mean
            pooling) — SMD quality varies per clip.

Both gate variants add parameters (smd_gate_*) and therefore load baseline
warm starts with strict=False; they are initialized so the gate opens at
~0.1, starting the model near the pose-only baseline.

--smd_dropout p zeroes the SMD segment's ATTENTION MASK per sample during
training (not the hidden states), blocking the SMD->translation shortcut.
--eval_mask_smd applies the same masking to every sample at evaluation
time: a diagnostic measuring whether the pose stream alone degraded
relative to the pose-only baseline.
"""

import math

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from models import Uni_Sign

# sigmoid(GATE_INIT_LOGIT) ~= 0.1 -> gated variants start near pose-only
GATE_INIT_LOGIT = math.log(0.1 / 0.9)

# plain vocabulary words, no new special tokens (tokenizes to short mT5
# pieces, no <unk>)
SMD_STREAM_PREFIX = "Motion description: "


class Uni_Sign_SignSMD_Dual(Uni_Sign):
    def __init__(self, args):
        super().__init__(args)
        self.signsmd_max_tokens = int(getattr(args, "signsmd_max_tokens", 512))
        if self.signsmd_max_tokens < 1:
            raise ValueError("--signsmd_max_tokens must be >= 1")
        self.fusion_alpha = float(getattr(args, "fusion_alpha", 1.0))
        self.fusion_gate_mode = getattr(args, "fusion_gate", "none")
        self.smd_dropout = float(getattr(args, "smd_dropout", 0.0))
        self.mask_smd_at_eval = bool(getattr(args, "eval_mask_smd", False))

        # gate parameters are created AFTER super().__init__'s
        # self.apply(self._init_weights), so this init is final
        if self.fusion_gate_mode == "global":
            self.smd_gate_logit = nn.Parameter(torch.tensor(GATE_INIT_LOGIT))
        elif self.fusion_gate_mode == "sample":
            hidden = self.get_llm_hidden_size()
            self.smd_gate_proj = nn.Linear(2 * hidden, 1)
            nn.init.constant_(self.smd_gate_proj.weight, 0.0)
            nn.init.constant_(self.smd_gate_proj.bias, GATE_INIT_LOGIT)
        elif self.fusion_gate_mode != "none":
            raise ValueError(
                f"--fusion_gate must be none|global|sample, "
                f"got {self.fusion_gate_mode!r}"
            )

    def _embed_tokens(self, texts, max_length=None):
        token = self.mt5_tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(next(self.parameters()).device)
        embeds = self.mt5_model.encoder.embed_tokens(token["input_ids"])
        return embeds, token["attention_mask"]

    def _encode_stream(self, inputs_embeds, attention_mask):
        return self.mt5_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

    @staticmethod
    def _masked_mean(hidden, mask):
        mask = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _smd_scale(self, h_pose, pose_mask, h_smd, smd_mask):
        if self.fusion_gate_mode == "global":
            return torch.sigmoid(self.smd_gate_logit)
        if self.fusion_gate_mode == "sample":
            pooled = torch.cat(
                [
                    self._masked_mean(h_pose, pose_mask),
                    self._masked_mean(h_smd, smd_mask),
                ],
                dim=-1,
            )
            # [B,1] -> [B,1,1], broadcast over the SMD segment
            return torch.sigmoid(self.smd_gate_proj(pooled)).unsqueeze(-1)
        return self.fusion_alpha

    def forward(self, src_input, tgt_input):
        descriptions = src_input.get("signsmd_text")
        if descriptions is None:
            raise KeyError(
                "src_input has no 'signsmd_text'; use "
                "datasets_signsmd.S2T_Dataset_SignSMD as the dataset"
            )

        # --- pose stream: exactly the baseline encoder input -------------
        pose_embeds = self.encode_pose(src_input)
        prefix_embeds, prefix_mask = self._embed_tokens(
            [f"Translate sign language video to {self.lang}: "]
            * len(tgt_input["gt_sentence"])
        )
        pose_stream_embeds = torch.cat([prefix_embeds, pose_embeds], dim=1)
        pose_mask = torch.cat(
            [prefix_mask, src_input["attention_mask"]], dim=1
        )

        # --- SMD stream: same encoder parameters, second pass ------------
        smd_prefix_embeds, smd_prefix_mask = self._embed_tokens(
            [SMD_STREAM_PREFIX] * len(descriptions)
        )
        desc_embeds, desc_mask = self._embed_tokens(
            list(descriptions), max_length=self.signsmd_max_tokens
        )
        smd_stream_embeds = torch.cat([smd_prefix_embeds, desc_embeds], dim=1)
        smd_mask = torch.cat([smd_prefix_mask, desc_mask], dim=1)

        h_pose = self._encode_stream(pose_stream_embeds, pose_mask)
        h_smd = self._encode_stream(smd_stream_embeds, smd_mask)

        # --- fusion ------------------------------------------------------
        h_smd = h_smd * self._smd_scale(h_pose, pose_mask, h_smd, smd_mask)

        if self.training and self.smd_dropout > 0:
            keep = (
                torch.rand(smd_mask.shape[0], device=smd_mask.device)
                >= self.smd_dropout
            )
            smd_mask = smd_mask * keep.to(smd_mask.dtype).unsqueeze(1)
        if self.mask_smd_at_eval and not self.training:
            smd_mask = torch.zeros_like(smd_mask)

        fused_hidden = torch.cat([h_pose, h_smd], dim=1)
        fused_mask = torch.cat([pose_mask, smd_mask], dim=1)

        # --- decoder + loss: exactly as in the baseline ------------------
        tgt_input_tokenizer = self.mt5_tokenizer(
            tgt_input["gt_sentence"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
        )
        labels = tgt_input_tokenizer["input_ids"]
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100

        out = self.mt5_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
            attention_mask=fused_mask,
            labels=labels.to(fused_hidden.device),
            return_dict=True,
        )

        label = labels.reshape(-1)
        out_logits = out["logits"]
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        if getattr(self.args, "compute_fp32_loss", False):
            logits = logits.float()
        loss_fct = torch.nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing, ignore_index=-100
        )
        loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))

        return {
            # use for inference (generate() below)
            "encoder_outputs": fused_hidden,
            "attention_mask": fused_mask,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams):
        # the inherited implementation passes inputs_embeds, which would
        # re-encode; here generation must start from the fused hidden states
        encoder_outputs = BaseModelOutput(
            last_hidden_state=pre_compute_item["encoder_outputs"]
        )
        out = self.mt5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=pre_compute_item["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return out
