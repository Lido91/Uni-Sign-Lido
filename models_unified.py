"""Janus/MotionGPT3-style unified sign model: ONE mT5 for both directions.

Understanding (SLT):  pose parts -> ST-GCN -> mT5 encoder+decoder -> text.
    Inherited unchanged from ``models.Uni_Sign``.

Generation (T2P):     text -> the SAME mT5 encoder (bidirectional, so
    MoMADiff's MAR masked prediction needs no attention surgery) over
    [t2p prompt+caption embeds ; latent slots] -> per-token conditioning
    vector -> LDM diffusion head samples 512-D KL-VAE latents -> frozen
    KL-VAE decoder -> [T,138] -> de-normalize -> body/left/right/face_all.

The KL-VAE (encoder AND decoder) is permanently frozen and its weights come
only from ``--vae_checkpoint`` (never stored in this model's checkpoints).
CFG follows JanusFlow's released implementation: the unconditional pass
zeroes the caption tokens' attention mask; cond vectors are mixed as
``uncond + (cond - uncond) * scale`` (MoMADiff's mixing point).
"""

import numpy as np
import torch
from torch import nn

from models import Uni_Sign
from diffusion_head import LDM
from mask_generation import (
    MaskGeneration,
    cosine_schedule,
    lengths_to_mask,
    uniform,
)
from vae_decoder import load_frozen_klvae_decoder, posterior_sample
from vae_encoder import DOWNSAMPLE_FACTOR, LATENT_DIM, load_trainable_klvae_encoder

# Same order/layout as MoMADiff utils/pose2d_repr.py.
POSE2D_PART_ORDER = ("body", "left", "right", "face_all")
POSE2D_PART_JOINTS = {"body": 9, "left": 21, "right": 21, "face_all": 18}
POSE2D_FEATURE_DIM = 138  # 69 joints x [x, y]

T2P_PROMPT = "Translate English to American Sign Language: "

MAX_LATENT_STEPS = 512  # 400-frame clips -> 100 latent steps; ample margin


def _sinusoidal_table(length, dim):
    table = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
    )
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table


def _load_stat(path, name):
    stat = np.load(path).astype(np.float32).reshape(-1)
    if stat.shape != (POSE2D_FEATURE_DIM,):
        raise ValueError(
            f"{name} at {path} must have shape ({POSE2D_FEATURE_DIM},), "
            f"got {stat.shape}"
        )
    return torch.from_numpy(stat)


class Uni_Sign_Unified(Uni_Sign):
    """Uni_Sign (SLT) + MoMADiff-style masked latent generation (T2P)."""

    def __init__(self, args):
        super().__init__(args)
        self.default_task = getattr(args, "default_task", "slt")

        # ---- frozen KL-VAE codec (never in checkpoints) -------------------
        self.vae_encoder = load_trainable_klvae_encoder(
            args.vae_checkpoint, input_emb_width=POSE2D_FEATURE_DIM
        )
        self.vae_decoder = load_frozen_klvae_decoder(
            args.vae_checkpoint, input_emb_width=POSE2D_FEATURE_DIM
        )
        self.vae_encoder.eval()
        for parameter in self.vae_encoder.parameters():
            parameter.requires_grad = False

        mean = _load_stat(args.vae_mean_path, "vae mean")
        std = _load_stat(args.vae_std_path, "vae std").clamp(min=1e-10)
        self.register_buffer("motion_mean", mean)
        self.register_buffer("motion_std", std)

        # Optional per-dim latent standardization (diffusion assumes roughly
        # unit-scale data; the raw posterior latents measured ~0.31 median
        # per-dim std with a 10x spread across dims). Stats come from
        # check_latent_scale.py --save_latent_stats; without them this is the
        # identity.
        latent_mean_path = getattr(args, "latent_mean_path", "")
        latent_std_path = getattr(args, "latent_std_path", "")
        if latent_mean_path and latent_std_path:
            lmean = torch.from_numpy(
                np.load(latent_mean_path).astype(np.float32).reshape(-1)
            )
            lstd = torch.from_numpy(
                np.load(latent_std_path).astype(np.float32).reshape(-1)
            ).clamp(min=1e-3)
            if lmean.shape != (LATENT_DIM,) or lstd.shape != (LATENT_DIM,):
                raise ValueError(
                    f"latent stats must have shape ({LATENT_DIM},)"
                )
        else:
            lmean = torch.zeros(LATENT_DIM)
            lstd = torch.ones(LATENT_DIM)
        self.register_buffer("latent_mean", lmean)
        self.register_buffer("latent_std", lstd)

        # ---- trainable generation adapters --------------------------------
        llm_hidden = self.get_llm_hidden_size()
        self.latent_in_proj = nn.Linear(LATENT_DIM, llm_hidden)
        self.mask_emb = nn.Parameter(torch.zeros(llm_hidden))
        self.register_buffer(
            "latent_pos_pe", _sinusoidal_table(MAX_LATENT_STEPS, llm_hidden)
        )
        self.cond_out_head = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.LayerNorm(llm_hidden),
            nn.Linear(llm_hidden, LATENT_DIM),
        )
        self.diff_head = LDM(
            input_dim=LATENT_DIM,
            cond_dim=LATENT_DIM,
            ff_size=getattr(args, "diff_ff_size", 1024),
            output_dim=LATENT_DIM,
            num_layers=getattr(args, "diff_layers", 4),
        )
        self.cond_drop_prob = float(getattr(args, "cond_drop_prob", 0.1))

        nn.init.trunc_normal_(self.mask_emb, std=0.02)
        for module in (self.latent_in_proj, *self.cond_out_head):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

        # Diffusion processes and the timestep sampler are plain python
        # objects owned by the entrypoint; attach via attach_diffusion().
        self.train_diffusion = None
        self.eval_diffusion = None
        self.schedule_sampler = None

    # ---------------------------------------------------------------- misc
    def attach_diffusion(self, train_diffusion, eval_diffusion, schedule_sampler):
        self.train_diffusion = train_diffusion
        self.eval_diffusion = eval_diffusion
        self.schedule_sampler = schedule_sampler

    def train(self, mode=True):
        super().train(mode)
        # the frozen codec stays in eval mode regardless of the outer state
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        # stage-frozen modules (set by the entrypoint) must ALSO stay in eval
        # mode: requires_grad=False alone would still let BatchNorm update
        # running statistics and dropout fire
        for name in getattr(self, "frozen_eval_modules", ()):
            getattr(self, name).eval()
        return self

    # ------------------------------------------------------ representation
    def parts_to_motion(self, src_input):
        """Flatten part dict xy to the z-scored [B,T,138] VAE representation."""
        parts = []
        for part in POSE2D_PART_ORDER:
            xy = src_input[part][..., :2]  # B,T,V,2
            parts.append(xy.reshape(xy.shape[0], xy.shape[1], -1))
        motion = torch.cat(parts, dim=-1)  # B,T,138
        return (motion - self.motion_mean.to(motion.dtype)) \
            / self.motion_std.to(motion.dtype)

    def motion_to_parts(self, motion, conf_value=1.0, clip_coordinates=True):
        """De-normalized ``[B,T,138]`` -> Uni-Sign part dict with conf channel.

        The layout follows pose2d_repr: a decoded sequence plus conf=1 obeys
        the exact schema the ST-GCN encoder consumes (back-translation).
        """
        joints = motion.reshape(*motion.shape[:-1], POSE2D_FEATURE_DIM // 2, 2)
        if clip_coordinates:
            joints = joints.clamp(-1.0, 1.0)
        conf = torch.full_like(joints[..., :1], conf_value)
        packed = torch.cat([joints, conf], dim=-1)  # [B,T,69,3]
        parts = {}
        start = 0
        for part in POSE2D_PART_ORDER:
            count = POSE2D_PART_JOINTS[part]
            parts[part] = packed[..., start:start + count, :]
            start += count
        return parts

    @torch.no_grad()
    def encode_gt_latents(self, src_input, sample=True):
        """Frozen-VAE latents at each sample's true length (grouped).

        Returns (latents [B,L_max,512], latent_lens LongTensor [B]).
        """
        motion = self.parts_to_motion(src_input)
        batch_size, num_frames, _ = motion.shape
        frame_lengths = src_input["src_length_batch"].to(
            device=motion.device, dtype=torch.long
        ).clamp(min=1, max=num_frames)

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
                pad = chunk[:, -1:].expand(-1, length - chunk.shape[1], -1)
                chunk = torch.cat([chunk, pad], dim=1)
            moments = self.vae_encoder(chunk)
            if sample:
                feats = posterior_sample(moments)
            else:
                feats, _ = torch.chunk(moments, 2, dim=-1)
            feats = (feats - self.latent_mean.to(feats.dtype)) \
                / self.latent_std.to(feats.dtype)
            latents[indices, : feats.shape[1]] = feats.to(latents.dtype)

        latent_lens = torch.tensor(
            [length // unit for length in trunc_lengths], device=motion.device
        )
        return latents, latent_lens

    # ----------------------------------------------------------- T2P train
    def _embed_t2p_text(self, sentences, device):
        prompt_token = self.mt5_tokenizer(
            [T2P_PROMPT + sentence for sentence in sentences],
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        text_embeds = self.mt5_model.encoder.embed_tokens(
            prompt_token["input_ids"]
        )
        return text_embeds, prompt_token["attention_mask"]

    def _encoder_cond(self, text_embeds, text_mask, latents, masked, non_pad):
        """One mT5-encoder pass over [text ; latent slots] -> cond [B,L,512].

        masked: bool [B,L], True where the slot carries mask_emb.
        non_pad: bool [B,L], True at valid latent steps.
        """
        num_steps = latents.shape[1]
        slots = self.latent_in_proj(latents)
        slots = torch.where(
            masked.unsqueeze(-1), self.mask_emb.to(slots.dtype), slots
        )
        slots = slots * non_pad.unsqueeze(-1).to(slots.dtype)
        slots = slots + self.latent_pos_pe[:num_steps].to(slots.dtype)

        encoder_out = self.mt5_model.encoder(
            inputs_embeds=torch.cat([text_embeds, slots], dim=1),
            attention_mask=torch.cat(
                [text_mask, non_pad.long()], dim=1
            ),
        ).last_hidden_state
        return self.cond_out_head(encoder_out[:, text_embeds.shape[1]:])

    def forward_t2p(self, src_input, tgt_input):
        if self.train_diffusion is None or self.schedule_sampler is None:
            raise RuntimeError(
                "call attach_diffusion() before the T2P forward"
            )
        latents, latent_lens = self.encode_gt_latents(src_input, sample=True)
        batch_size, num_steps, _ = latents.shape
        device = latents.device

        non_pad = lengths_to_mask(latent_lens, num_steps)  # [B,L] True=valid

        # MoMADiff cosine random masking, made per-sample-length aware:
        # ranking random scores among VALID positions only guarantees every
        # sample keeps >= 1 masked (supervised) token — the upstream variant
        # ranked over the padded batch length, so a short sample could end up
        # with all its candidates on padding and contribute no signal.
        rand_time = uniform((batch_size,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (
            latent_lens.float() * rand_mask_probs
        ).round().clamp(min=1)
        scores = torch.rand((batch_size, num_steps), device=device)
        scores = scores.masked_fill(~non_pad, 2.0)  # pads rank last, never chosen
        ranks = scores.argsort(dim=-1).argsort(dim=-1)
        mask = ranks < num_token_masked.unsqueeze(-1)

        text_embeds, text_mask = self._embed_t2p_text(
            tgt_input["gt_sentence"], device
        )
        # per-sample CFG dropout: zero the whole text row's attention mask
        if self.training and self.cond_drop_prob > 0:
            drop = (
                torch.rand(batch_size, device=device) < self.cond_drop_prob
            )
            text_mask = text_mask * (~drop).long().unsqueeze(-1)

        cond = self._encoder_cond(
            text_embeds, text_mask, latents, mask, non_pad
        )

        latents_flat = latents.reshape(-1, LATENT_DIM)
        cond_flat = cond.reshape(-1, LATENT_DIM)
        mask_flat = mask.reshape(-1)
        t, _ = self.schedule_sampler.sample(latents_flat.shape[0], device)
        losses = self.train_diffusion.training_losses(
            self.diff_head,
            latents_flat,
            t,
            model_kwargs={
                "y": {
                    "mask": torch.ones(1, 1, device=device),
                    "text_embed": cond_flat,
                }
            },
        )
        # loss_strategy='masked': only freshly-masked tokens count. Normalize
        # by the masked-token COUNT (not the padded batch length) so short
        # sequences are not down-weighted.
        loss = (losses["loss"] * mask_flat).sum() \
            / mask_flat.sum().clamp(min=1)
        return {"loss": loss}

    # ------------------------------------------------------- T2P inference
    @torch.no_grad()
    def generate_pose(
        self,
        sentences,
        frame_lens,
        timesteps=9,
        cond_scale=3.0,
        clip_coordinates=True,
    ):
        """Text -> pose parts via MAR masked generation + DDIM diffusion.

        sentences: list[str]; frame_lens: LongTensor [B] target frame counts
        (GT-length oracle, MoMADiff protocol).
        clip_coordinates=False preserves raw decoded coordinates for MoMADiff
        DTW evaluation; the default retains the existing SLT-input clipping.
        Returns (parts dict with conf=1, gen_frame_lens LongTensor [B]).
        """
        if self.eval_diffusion is None:
            raise RuntimeError("call attach_diffusion() before generate_pose")
        was_training = self.training
        self.eval()

        device = self.motion_mean.device
        unit = DOWNSAMPLE_FACTOR
        frame_lens = frame_lens.to(device=device, dtype=torch.long)
        m_lens = torch.div(frame_lens, unit, rounding_mode="floor").clamp(min=1)
        batch_size = len(sentences)
        num_steps = int(m_lens.max())
        non_pad = lengths_to_mask(m_lens, num_steps)

        text_embeds, text_mask = self._embed_t2p_text(sentences, device)
        uncond_mask = torch.zeros_like(text_mask)

        # raw 512-D latent buffer; still-masked positions are re-slotted with
        # mask_emb on every pass (MoMADiff keeps embs in code_dim space too)
        embs = torch.zeros(
            batch_size, num_steps, LATENT_DIM, device=device,
            dtype=self.motion_mean.dtype,
        )
        mask_gen = MaskGeneration(
            m_lens,
            num_steps,
            timesteps,
            noise_schedule=cosine_schedule,
            device=device,
            drop_last=False,
        )
        current_masked = next(mask_gen)
        assert int(current_masked.sum()) == int(m_lens.sum()), \
            "the first mask should cover every valid step"
        for next_masked in mask_gen:
            diff_mask = current_masked ^ next_masked
            if diff_mask.sum() == 0:
                current_masked = next_masked
                continue

            cond = self._encoder_cond(
                text_embeds, text_mask, embs, current_masked, non_pad
            )
            if cond_scale != 1:
                uncond = self._encoder_cond(
                    text_embeds, uncond_mask, embs, current_masked, non_pad
                )
                cond = uncond + (cond - uncond) * cond_scale

            cond_rows = cond[diff_mask]
            # true DDIM (eta=0) over the respaced schedule — p_sample_loop
            # would be ancestral sampling even on a "ddim"-respaced process
            pred = self.eval_diffusion.ddim_sample_loop(
                self.diff_head,
                (cond_rows.shape[0], LATENT_DIM),
                clip_denoised=False,
                model_kwargs={"y": {"text_embed": cond_rows}},
                progress=False,
                eta=0.0,
            )
            embs[diff_mask] = pred.to(embs.dtype)
            current_masked = next_masked
        assert int(current_masked.sum()) == 0, "the last mask should be empty"

        # de-standardize back to the raw latent space before decoding
        raw_latents = embs * self.latent_std.to(embs.dtype) \
            + self.latent_mean.to(embs.dtype)

        # decode grouped by true latent length (padding would pollute the
        # non-causal decoder convolutions, exactly like the encoder)
        gen_frame_lens = m_lens * unit
        max_frames = int(gen_frame_lens.max())
        motion = embs.new_zeros(batch_size, max_frames, POSE2D_FEATURE_DIM)
        groups = {}
        for index, length in enumerate(m_lens.tolist()):
            groups.setdefault(length, []).append(index)
        for length, indices in groups.items():
            decoded = self.vae_decoder(raw_latents[indices, :length])
            motion[indices, : decoded.shape[1]] = decoded.to(motion.dtype)

        motion = motion * self.motion_std.to(motion.dtype) \
            + self.motion_mean.to(motion.dtype)
        parts = self.motion_to_parts(motion, clip_coordinates=clip_coordinates)

        if was_training:
            self.train()
        return parts, gen_frame_lens

    # ------------------------------------------------------------ dispatch
    def forward(self, src_input, tgt_input, task=None,
                t2p_src_input=None, t2p_tgt_input=None,
                slt_weight=1.0, t2p_weight=1.0):
        task = task or self.default_task
        if task == "slt":
            return super().forward(src_input, tgt_input)
        if task == "t2p":
            return self.forward_t2p(src_input, tgt_input)
        if task == "joint":
            # Both objectives inside ONE wrapped forward call: torch DDP
            # tolerates a parameter appearing several times within a single
            # forward's autograd graph, but two separate wrapped forwards
            # sharing one backward trip the "marked ready twice" reducer
            # check on every shared mT5 parameter.
            slt_out = super().forward(src_input, tgt_input)
            t2p_out = self.forward_t2p(t2p_src_input, t2p_tgt_input)
            loss = slt_weight * slt_out["loss"] + t2p_weight * t2p_out["loss"]
            return {
                "loss": loss,
                "slt_loss": slt_out["loss"].detach(),
                "t2p_loss": t2p_out["loss"].detach(),
            }
        raise ValueError(
            f"unknown task {task!r} (expected 'slt', 't2p' or 'joint')"
        )
