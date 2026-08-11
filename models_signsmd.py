"""Baseline Uni-Sign + SignSMD text fusion (option A of the fusion design).

Additive subclass of the ORIGINAL ``models.Uni_Sign`` (ST-GCN + mT5
baseline).  The four-part ST-GCN pose encoder, the mT5 objective, and beam
generation are inherited unchanged; the ONLY change is the mT5 encoder
input, which gains a per-sample structured description segment:

    [SignSMD text tokens][translate prompt][ST-GCN pose embeddings]

Segment order rationale: the description goes FIRST so that

1. the pretrained adjacency [translate prompt][pose] from existing baseline
   checkpoints is preserved (this path adds ZERO new parameters, so any
   baseline mT5 Uni-Sign checkpoint loads with strict=True and starts from
   exactly the baseline behavior when the text adds nothing);
2. the description segment can be truncated independently
   (``--signsmd_max_tokens``, tail-truncated) without ever cutting the
   prompt.  Interval-aware compaction should preferably happen at
   compile time in the SignSMD realization; the truncation here is only a
   hard safety net.

The description is real discrete text read by mT5 through its own
embedding table — not a pooled sentence vector — so the fused model keeps
the auditability of the SignSMD representation (unlike MMSLT-style [CLS]
feature fusion).
"""

import torch

from models import Uni_Sign


class Uni_Sign_SignSMD(Uni_Sign):
    def __init__(self, args):
        super().__init__(args)
        self.signsmd_max_tokens = int(getattr(args, "signsmd_max_tokens", 512))
        if self.signsmd_max_tokens < 1:
            raise ValueError("--signsmd_max_tokens must be >= 1")

    def forward(self, src_input, tgt_input):
        descriptions = src_input.get("signsmd_text")
        if descriptions is None:
            raise KeyError(
                "src_input has no 'signsmd_text'; use "
                "datasets_signsmd.S2T_Dataset_SignSMD as the dataset"
            )

        inputs_embeds = self.encode_pose(src_input)

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

        desc_token = self.mt5_tokenizer(
            list(descriptions),
            padding="longest",
            truncation=True,
            max_length=self.signsmd_max_tokens,
            return_tensors="pt",
        ).to(inputs_embeds.device)
        desc_embeds = self.mt5_model.encoder.embed_tokens(
            desc_token["input_ids"]
        )

        inputs_embeds = torch.cat(
            [desc_embeds, prefix_embeds, inputs_embeds], dim=1
        )
        attention_mask = torch.cat(
            [
                desc_token["attention_mask"],
                prefix_token["attention_mask"],
                src_input["attention_mask"],
            ],
            dim=1,
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
