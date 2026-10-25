"""Time-aligned, part-paired SignSMD injection for Uni-Sign.

Each structured event phrase is tokenized independently, mean-pooled through
the shared mT5 token embedding table, projected to the 256-d pose-part space,
and scattered over its inclusive [start, end] pose-step span.  Overlapping
events of the same part are averaged.  The four resulting tensors are injected
only into their matching body/left/right/face_all ST-GCN streams after spatial
GCN + body-anchor addition and before temporal GCN.

The final convolution of the shared fusion gate is zero-initialized.  Thus a
baseline checkpoint loaded with strict=False produces the exact baseline pose
embeddings at initialization, while the final gate convolution still receives
gradient (unlike an all-zero ReLU gate, which would be permanently dead).
"""

import torch
from torch import nn

from models import Uni_Sign
from signsmd2d.realization_parts import PARTS


class Uni_Sign_SignSMD_Paired(Uni_Sign):
    def __init__(self, args):
        super().__init__(args)
        self.signsmd_part_max_tokens = int(
            getattr(args, "signsmd_part_max_tokens", 128)
        )
        self.text_event_dropout = float(
            getattr(args, "text_event_dropout", 0.3)
        )
        self.mask_text_at_eval = bool(getattr(args, "eval_mask_text", False))
        if self.signsmd_part_max_tokens < 1:
            raise ValueError("--signsmd_part_max_tokens must be >= 1")
        if not 0.0 <= self.text_event_dropout <= 1.0:
            raise ValueError("--text_event_dropout must be in [0, 1]")

        llm_hidden = self.get_llm_hidden_size()
        self.signsmd_text_proj = nn.Linear(llm_hidden, self.pose_part_dim)
        self.signsmd_text_gate = nn.Sequential(
            nn.Conv1d(2 * self.pose_part_dim, self.pose_part_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.pose_part_dim, 1, 1),
            nn.Tanh(),
        )

        # These modules are created after Uni_Sign.__init__ has run its global
        # initializer, so initialize them explicitly.  Only the FINAL gate
        # projection is zero: identity at step 0 with a live gradient path.
        self.signsmd_text_proj.apply(self._init_weights)
        final_gate = self.signsmd_text_gate[2]
        nn.init.constant_(final_gate.weight, 0.0)
        nn.init.constant_(final_gate.bias, 0.0)

    def _validate_part_text_batch(self, src_input, batch_size):
        part_texts = src_input.get("signsmd_part_texts")
        if not isinstance(part_texts, dict) or set(part_texts) != set(PARTS):
            raise KeyError(
                "src_input['signsmd_part_texts'] must contain exactly "
                f"{list(PARTS)}; use the paired SignSMD dataset contract"
            )
        for part in PARTS:
            if len(part_texts[part]) != batch_size:
                raise ValueError(
                    f"{part} text batch has {len(part_texts[part])} items, "
                    f"expected {batch_size}"
                )

    def _tokenize_event_vectors(self, phrases, device):
        token = self.mt5_tokenizer(
            phrases,
            padding="longest",
            truncation=True,
            max_length=self.signsmd_part_max_tokens,
            return_tensors="pt",
        ).to(device)
        embeds = self.mt5_model.encoder.embed_tokens(token["input_ids"])
        mask = token["attention_mask"].to(embeds.dtype).unsqueeze(-1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled.to(self.signsmd_text_proj.weight.dtype)
        return self.signsmd_text_proj(pooled)

    def build_part_text_features(self, src_input):
        """Return dict part -> [B,T,C] aligned text features.

        Each EVENT phrase has its own token cap. No body/hand/face phrase can
        truncate a phrase belonging to another part, and no whole-clip text is
        concatenated before tokenization.
        """
        pose_mask = src_input["attention_mask"]
        batch_size, num_steps = pose_mask.shape
        self._validate_part_text_batch(src_input, batch_size)

        event_batch = src_input.get("signsmd_events")
        if not isinstance(event_batch, (list, tuple)):
            raise KeyError(
                "src_input has no paired 'signsmd_events'; use "
                "datasets_signsmd.S2T_Dataset_SignSMD with --fusion paired"
            )
        if len(event_batch) != batch_size:
            raise ValueError(
                f"event batch has {len(event_batch)} items, expected {batch_size}"
            )

        if self.mask_text_at_eval and not self.training:
            keep_sample = torch.zeros(
                batch_size, dtype=torch.bool, device=pose_mask.device
            )
        elif self.training and self.text_event_dropout > 0.0:
            keep_sample = (
                torch.rand(batch_size, device=pose_mask.device)
                >= self.text_event_dropout
            )
        else:
            keep_sample = torch.ones(
                batch_size, dtype=torch.bool, device=pose_mask.device
            )

        flat_events = []
        for batch_index, events in enumerate(event_batch):
            if not isinstance(events, (list, tuple)):
                raise ValueError(
                    f"events for batch item {batch_index} must be a list"
                )
            if not bool(keep_sample[batch_index]):
                continue
            true_steps = int(pose_mask[batch_index].sum().item())
            for event in events:
                part = event.get("part")
                start = event.get("start")
                end = event.get("end")
                phrase = event.get("phrase")
                if part not in PARTS:
                    raise ValueError(f"invalid paired event part {part!r}")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start <= end < true_steps
                ):
                    raise ValueError(
                        f"paired event span [{start}, {end}] outside "
                        f"sample length {true_steps}"
                    )
                if not isinstance(phrase, str) or not phrase.strip():
                    raise ValueError("paired event phrase must be non-empty")
                flat_events.append((batch_index, part, start, end, phrase))

        dtype = self.signsmd_text_proj.weight.dtype
        zero_anchor = self.signsmd_text_proj.weight.sum() * 0.0
        if not flat_events:
            return {
                part: torch.zeros(
                    batch_size,
                    num_steps,
                    self.pose_part_dim,
                    device=pose_mask.device,
                    dtype=dtype,
                ) + zero_anchor
                for part in PARTS
            }

        event_vectors = self._tokenize_event_vectors(
            [event[4] for event in flat_events], pose_mask.device
        )
        num_events = len(flat_events)
        features = {}
        for part in PARTS:
            weights = event_vectors.new_zeros(
                batch_size, num_steps, num_events
            )
            for event_index, (batch_index, event_part, start, end, _) in enumerate(
                flat_events
            ):
                if event_part == part:
                    weights[batch_index, start:end + 1, event_index] = 1.0
            counts = weights.sum(dim=-1, keepdim=True)
            weights = weights / counts.clamp(min=1.0)
            features[part] = torch.einsum(
                "btn,nc->btc", weights, event_vectors
            )
        return features

    def _inject_part_text(self, part, gcn_feat, text_feat):
        if text_feat.ndim != 3:
            raise ValueError(
                f"{part} text feature must be [B,T,C], got {text_feat.shape}"
            )
        batch_size, channels, num_steps, _ = gcn_feat.shape
        if text_feat.shape != (batch_size, num_steps, channels):
            raise ValueError(
                f"{part} text feature has shape {text_feat.shape}, expected "
                f"{(batch_size, num_steps, channels)}"
            )
        text_ct = text_feat.transpose(1, 2).to(gcn_feat.dtype)
        pose_ct = gcn_feat.mean(dim=-1)
        gate = self.signsmd_text_gate(
            torch.cat([pose_ct, text_ct], dim=1)
        )
        return gcn_feat + gate.unsqueeze(-1) * text_ct.unsqueeze(-1)

    def forward(self, src_input, tgt_input):
        part_text_features = self.build_part_text_features(src_input)
        inputs_embeds = self.encode_pose(
            src_input, part_text_features=part_text_features
        )

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
            [prefix_token["attention_mask"], src_input["attention_mask"]],
            dim=1,
        )

        tgt_token = self.mt5_tokenizer(
            tgt_input["gt_sentence"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
        )
        labels = tgt_token["input_ids"]
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
        loss_fct = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing, ignore_index=-100
        )
        loss = loss_fct(
            logits, label.to(out_logits.device, non_blocking=True)
        )
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "loss": loss,
        }
