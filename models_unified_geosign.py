"""Use Geo-Sign's unchanged SLT forward inside the existing unified model.

The vendored model is a byte-for-byte copy. Its Uni_Sign constructor is never
called: ST-GCN, mT5 and the motion codec remain the single existing instances.
Only its original hyperbolic initialization block and SLT forward are reused.
"""
import torch
from torch import nn

from models_unified import Uni_Sign_Unified
from vendor.geosign import models as reference
from vendor.geosign.models import (
    GEOOPT_AVAILABLE, PoincareBall, HyperbolicProjection,
    HyperbolicContrastiveLoss, geoopt,
)

GEOSIGN_PREFIXES = (
    "hyp_", "manifold.", "geom_loss.", "loss_alpha_logit", "global_step",
    "text_pool_attn.",
)


class Uni_Sign_Unified_GeoSign(Uni_Sign_Unified):
    def __init__(self, args):
        if not GEOOPT_AVAILABLE:
            raise ImportError("Geo-Sign requires geoopt; see the existing Uni-Sign environment")
        super().__init__(args)
        self.use_hyp = True
        self.mt5_dim = self.mt5_model.config.d_model
        final_dim_gcn = self.pose_proj.in_features // len(self.modes)
        if self.use_hyp and GEOOPT_AVAILABLE:
            self.hyp_dim    = args.hyp_dim
            self.manifold   = PoincareBall(c=args.init_c, learnable=True)
            self.hyp_proj_body  = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_right = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_left  = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_face  = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_text  = HyperbolicProjection(self.mt5_dim,   self.hyp_dim, self.manifold)

            # For 'token' mode attention: keys = (W ⊗_c v) ⊕_c b.
            # W is a Euclidean matrix (Möbius matvec accepts any real matrix; the paper
            # treats it as such), initialised to identity so keys start equal to the
            # values. Only the bias b is a point on the ball (RiemannianAdam).
            self.hyp_attn_W = nn.Parameter(torch.eye(args.hyp_dim))
            self.hyp_attn_b = geoopt.ManifoldParameter(torch.zeros(args.hyp_dim), manifold=self.manifold)


            self.geom_loss        = HyperbolicContrastiveLoss(self.manifold, args.label_smoothing_hyp)
            self.loss_alpha_logit = nn.Parameter(torch.tensor(0.0))
            self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
            self.total_steps      = max(int(getattr(args, 'total_steps', 1)), 1)
            self.text_cmp_mode    = getattr(args, "hyp_text_cmp", "pooled")
            if self.text_cmp_mode not in {"pooled", "attn", "token"}:
                raise ValueError(f"--hyp_text_cmp must be pooled|attn|token, got {self.text_cmp_mode}")
            if self.text_cmp_mode == "attn":
                self.text_pool_attn = nn.Linear(self.mt5_dim, 1)
            self.hyp_text_emb_src = getattr(args, "hyp_text_emb_src", "token")
            if self.hyp_text_emb_src not in {"token", "decoder"}:
                raise ValueError(f"--hyp_text_emb_src must be 'token' or 'decoder', got {self.hyp_text_emb_src}")

        # Match Geo-Sign's initialization, without reinitializing the backbone.
        for name, module in self.named_children():
            if name.startswith(("hyp_", "geom_loss", "manifold", "text_pool_attn")):
                module.apply(reference.Uni_Sign._init_weights)

    def forward_slt_geosign(self, src_input, tgt_input):
        device = self.pose_proj.weight.device
        src = dict(src_input)
        tgt = dict(tgt_input)
        if "prefix_ids" not in src:
            prefix = self.mt5_tokenizer(
                [f"Translate sign language video to {self.lang}: "]
                * len(tgt["gt_sentence"]),
                padding="longest", truncation=True, return_tensors="pt",
            ).to(device)
            src["prefix_ids"] = prefix["input_ids"]
            src["prefix_mask"] = prefix["attention_mask"]
        if "labels_ids" not in tgt:
            target = self.mt5_tokenizer(
                tgt["gt_sentence"], padding=True, truncation=True,
                max_length=50, return_tensors="pt",
            ).to(device)
            tgt["labels_ids"] = target["input_ids"]
        # Execute the exact source implementation; no rewritten geometry/loss.
        return reference.Uni_Sign.forward(self, src, tgt)

    def forward(self, src_input, tgt_input, task=None,
                t2p_src_input=None, t2p_tgt_input=None,
                slt_weight=1.0, t2p_weight=1.0):
        task = task or self.default_task
        if task == "slt":
            return self.forward_slt_geosign(src_input, tgt_input)
        if task == "t2p":
            return self.forward_t2p(src_input, tgt_input)
        if task != "joint":
            raise ValueError(f"Unknown unified task: {task}")
        slt_out = self.forward_slt_geosign(src_input, tgt_input)
        t2p_out = self.forward_t2p(t2p_src_input, t2p_tgt_input)
        result = {
            "loss": slt_weight * slt_out["loss"] + t2p_weight * t2p_out["loss"],
            "slt_loss": slt_out["loss"].detach(),
            "t2p_loss": t2p_out["loss"].detach(),
        }
        for key in ("ce_loss", "margin_loss", "alpha", "curvature",
                    "temperature", "effective_margin", "hyp_sim_mean"):
            if key in slt_out:
                result[key] = slt_out[key].detach()
        return result
