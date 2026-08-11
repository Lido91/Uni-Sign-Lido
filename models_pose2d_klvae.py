"""Uni-Sign mT5 model whose ST-GCN pose encoder is replaced by the joint
How2Sign+OpenASL pose2d KL-VAE.

This is an additive model: ``models.Uni_Sign`` is not imported, subclassed, or
modified.  It consumes the ORIGINAL ``datasets.S2T_Dataset`` batches (the four
part tensors body/left/right/face_all, each ``[B,T,V,3]`` with x,y,conf) and:

1. flattens the xy coordinates in MoMADiff's exact part order
   (body 9 | left 21 | right 21 | face_all 18 -> 69 joints x 2 = 138D),
   dropping the confidence channel (the VAE never models it; low-confidence
   joints are already zero-filled by ``datasets.load_part_kp``);
2. z-scores with the Mean/Std persisted by the trained KL-VAE
   (``kl_vae_h2s_openasl_pose2d/meta/{mean,std}.npy``);
3. runs the pretrained, PERMANENTLY FROZEN KL-VAE encoder (under
   ``torch.no_grad()``) PER SAMPLE at each sample's true length truncated to
   a multiple of 4 (grouped by equal length), so collate padding never leaks
   through the non-causal temporal convolutions:
   ``[T_i,138] -> [T_i/4,512]`` posterior mean, then latent-pads the batch;
4. projects with ONE linear head into the mT5 embedding space
   (``Linear(512, 768)``) — no ST-GCN, no extra MLP;
5. keeps the original Uni-Sign prompt, seq2seq cross-entropy, and beam
   generation, with the frame attention mask downsampled x4 to latent steps.
"""

import numpy as np
import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from config import mt5_path
from vae_encoder import (
    DOWNSAMPLE_FACTOR,
    LATENT_DIM,
    load_trainable_klvae_encoder,
    posterior_mean,
)

# Same order as MoMADiff utils/pose2d_repr.py (body | lhand | rhand | face).
POSE2D_PART_ORDER = ("body", "left", "right", "face_all")
POSE2D_FEATURE_DIM = 138  # 69 joints x [x, y]


def _load_stat(path, name):
    stat = np.load(path).astype(np.float32).reshape(-1)
    if stat.shape != (POSE2D_FEATURE_DIM,):
        raise ValueError(
            f"{name} at {path} must have shape ({POSE2D_FEATURE_DIM},), "
            f"got {stat.shape}"
        )
    return torch.from_numpy(stat)


class Uni_Sign_Pose2D_KLVAE(nn.Module):
    """ST-GCN-free Uni-Sign: frozen KL-VAE encoder + one linear head into mT5."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lang = "Chinese" if "CSL" in args.dataset else "English"

        mean = _load_stat(args.vae_mean_path, "vae mean")
        std = _load_stat(args.vae_std_path, "vae std").clamp(min=1e-10)
        self.register_buffer("motion_mean", mean)
        self.register_buffer("motion_std", std)

        # The KL-VAE is a permanently frozen feature extractor: SLT training
        # only updates the head and mT5, so the encoder stays numerically
        # identical to the MoMADiff checkpoint.
        self.vae_encoder = load_trainable_klvae_encoder(
            args.vae_checkpoint, input_emb_width=POSE2D_FEATURE_DIM
        )
        self.vae_encoder.eval()
        for parameter in self.vae_encoder.parameters():
            parameter.requires_grad = False

        llm_path = getattr(args, "llm_path", "") or mt5_path
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(llm_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(llm_path, legacy=False)
        self.text_tokenizer = self.mt5_tokenizer

        # The single head replacing ST-GCN's Linear(256*4, 768) pose_proj.
        self.pose_proj = nn.Linear(
            LATENT_DIM, int(self.mt5_model.config.d_model)
        )
        nn.init.trunc_normal_(self.pose_proj.weight, std=0.02)
        nn.init.zeros_(self.pose_proj.bias)

    def train(self, mode=True):
        super().train(mode)
        # keep the frozen VAE in eval mode regardless of the outer state
        self.vae_encoder.eval()
        return self

    def encode_pose(self, src_input):
        parts = []
        for part in POSE2D_PART_ORDER:
            xy = src_input[part][..., :2]  # B,T,V,2
            parts.append(xy.reshape(xy.shape[0], xy.shape[1], -1))
        motion = torch.cat(parts, dim=-1)  # B,T,138
        motion = (motion - self.motion_mean.to(motion.dtype)) \
            / self.motion_std.to(motion.dtype)

        batch_size, num_frames, _ = motion.shape
        frame_lengths = src_input["src_length_batch"].to(
            device=motion.device, dtype=torch.long
        ).clamp(min=1, max=num_frames)

        # The VAE is a NON-CAUSAL temporal CNN with a large receptive field:
        # encoding the collate-padded batch would leak repeat-last padding
        # into valid latents (a mask cannot undo that, and a sample's
        # representation would depend on the other samples' lengths). So each
        # sample is encoded at its own length truncated to a multiple of 4,
        # batching samples of equal truncated length, then latents are padded.
        unit = DOWNSAMPLE_FACTOR
        trunc_lengths = [
            max((int(length) // unit) * unit, unit)
            for length in frame_lengths.tolist()
        ]
        groups = {}
        for index, length in enumerate(trunc_lengths):
            groups.setdefault(length, []).append(index)

        max_steps = max(trunc_lengths) // unit
        latents = motion.new_zeros(batch_size, max_steps, LATENT_DIM)
        for length, indices in groups.items():
            chunk = motion[indices, :length]
            if chunk.shape[1] < length:
                # only when every sample in the batch is shorter than 4
                # frames: repeat each sample's own last frame up to one unit
                pad = chunk[:, -1:].expand(-1, length - chunk.shape[1], -1)
                chunk = torch.cat([chunk, pad], dim=1)
            with torch.no_grad():
                feats = posterior_mean(self.vae_encoder(chunk))
            latents[indices, : feats.shape[1]] = feats.to(latents.dtype)

        inputs_embeds = self.pose_proj(latents)
        latent_lengths = torch.tensor(
            [length // unit for length in trunc_lengths],
            device=motion.device,
        )
        positions = torch.arange(max_steps, device=motion.device).unsqueeze(0)
        attention_mask = (positions < latent_lengths.unsqueeze(1)).long()
        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):
        inputs_embeds, pose_mask = self.encode_pose(src_input)

        prefix_token = self.mt5_tokenizer(
            [f"Translate sign language video to {self.lang}: "]
            * len(tgt_input["gt_sentence"]),
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(inputs_embeds.device)

        prefix_embeds = self.mt5_model.encoder.embed_tokens(
            prefix_token["input_ids"]
        )
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [prefix_token["attention_mask"], pose_mask], dim=1
        )

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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(inputs_embeds.device),
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
            # use for inference
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams):
        return self.mt5_model.generate(
            inputs_embeds=pre_compute_item["inputs_embeds"],
            attention_mask=pre_compute_item["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
