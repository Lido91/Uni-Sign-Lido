"""Uni-Sign with a decoder-only Qwen LLM (LoRA) in place of mT5.

The ST-GCN pose encoder is inherited unchanged from models.Uni_Sign; only the
language-model side differs. Because Qwen is decoder-only, pose frames cannot go
into a separate encoder: each batch row is packed as one embedding sequence

    training:   [prompt embeds; pose embeds; target embeds]  (right-padded)
    generation: [pad; prompt embeds; pose embeds]            (left-padded)

with labels set to -100 everywhere except the target tokens. The Qwen base
weights are frozen; only the LoRA adapters, the pose encoder, part_para and
pose_proj are trainable (and therefore the only tensors saved in checkpoints,
see models.get_requires_grad_dict — load such checkpoints with strict=False).
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from models import Uni_Sign
from config import qwen_path

TARGET_MAX_TOKENS = 50  # same target truncation length as the mT5 pipeline


class Uni_Sign_Qwen(Uni_Sign):
    def _llm_path(self):
        return getattr(self.args, 'llm_path', '') or qwen_path

    def get_llm_hidden_size(self):
        # read from config only, so pose_proj can be built (and init-ed by
        # self.apply(_init_weights)) before the LLM weights are loaded
        return AutoConfig.from_pretrained(self._llm_path()).hidden_size

    def _build_llm(self):
        path = self._llm_path()
        dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
        load_dtype = dtype_map.get(getattr(self.args, 'dtype', 'bf16'), torch.bfloat16)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=load_dtype)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(path)

        for p in self.qwen_model.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_cfg)

        if getattr(self.args, 'llm_grad_ckpt', False):
            self.qwen_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

        self.text_tokenizer = self.qwen_tokenizer

    def _build_sequences(self, src_input, tgt_input):
        """Pack per-sample [prompt; pose; target] rows, right-padded.

        Returns (inputs_embeds, attention_mask, labels, prefix_len) where
        prefix_len[i] = prompt tokens + valid pose frames of sample i, i.e. the
        generation prefix length.
        """
        pose_embeds = self.encode_pose(src_input)  # [B, T, H]
        embed_layer = self.qwen_model.get_input_embeddings()
        pose_embeds = pose_embeds.to(embed_layer.weight.dtype)
        device = pose_embeds.device
        pose_mask = src_input['attention_mask']

        prompt_ids = self.qwen_tokenizer(
            f"Translate sign language video to {self.lang}: ",
            add_special_tokens=False, return_tensors='pt',
        ).input_ids.to(device)
        prompt_embeds = embed_layer(prompt_ids)[0]  # [P, H]

        tgt_tok = self.qwen_tokenizer(
            tgt_input['gt_sentence'],
            add_special_tokens=False, truncation=True, max_length=TARGET_MAX_TOKENS,
        )
        eos_id = self.qwen_tokenizer.eos_token_id

        seqs, labels, prefix_lens = [], [], []
        for i in range(pose_embeds.shape[0]):
            t_i = int(pose_mask[i].sum().item())
            tgt_ids = torch.tensor(tgt_tok['input_ids'][i] + [eos_id],
                                   dtype=torch.long, device=device)
            seq = torch.cat([prompt_embeds, pose_embeds[i, :t_i], embed_layer(tgt_ids)], dim=0)
            prefix = prompt_embeds.shape[0] + t_i
            lab = torch.cat([torch.full((prefix,), -100, dtype=torch.long, device=device),
                             tgt_ids])
            seqs.append(seq)
            labels.append(lab)
            prefix_lens.append(prefix)

        max_len = max(s.shape[0] for s in seqs)
        batch = len(seqs)
        inputs_embeds = pose_embeds.new_zeros(batch, max_len, seqs[0].shape[1])
        attention_mask = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        label_batch = torch.full((batch, max_len), -100, dtype=torch.long, device=device)
        for i, (seq, lab) in enumerate(zip(seqs, labels)):
            inputs_embeds[i, :seq.shape[0]] = seq
            attention_mask[i, :seq.shape[0]] = 1
            label_batch[i, :lab.shape[0]] = lab

        prefix_lens = torch.tensor(prefix_lens, dtype=torch.long, device=device)
        return inputs_embeds, attention_mask, label_batch, prefix_lens

    def forward(self, src_input, tgt_input):
        inputs_embeds, attention_mask, labels, prefix_lens = \
            self._build_sequences(src_input, tgt_input)

        out = self.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # causal LM: predict token t+1 from position t
        logits = out['logits'][:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        logits = logits.reshape(-1, logits.shape[-1])
        if getattr(self.args, 'compute_fp32_loss', False):
            logits = logits.float()
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing,
                                       ignore_index=-100)
        loss = loss_fct(logits, shift_labels.reshape(-1))

        return {
            # used by generate(): the [prompt; pose] prefix of each row
            'inputs_embeds': inputs_embeds,
            'prefix_len': prefix_lens,
            'loss': loss,
        }

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams):
        inputs_embeds = pre_compute_item['inputs_embeds']
        prefix_lens = pre_compute_item['prefix_len']
        batch, _, hidden = inputs_embeds.shape

        # decoder-only generation needs left padding
        max_prefix = int(prefix_lens.max().item())
        gen_embeds = inputs_embeds.new_zeros(batch, max_prefix, hidden)
        gen_mask = torch.zeros(batch, max_prefix, dtype=torch.long,
                               device=inputs_embeds.device)
        for i in range(batch):
            length = int(prefix_lens[i].item())
            gen_embeds[i, max_prefix - length:] = inputs_embeds[i, :length]
            gen_mask[i, max_prefix - length:] = 1

        out = self.qwen_model.generate(
            inputs_embeds=gen_embeds,
            attention_mask=gen_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=self.qwen_tokenizer.eos_token_id,
        )
        # with inputs_embeds (no input_ids), HF returns only the new tokens
        return out
