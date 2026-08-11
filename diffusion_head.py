# Ported from MoMADiff models/diffusion/mlp.py: the per-token latent
# diffusion head (LDM = MLP denoiser with DiT-style AdaLayerNorm conditioning
# on token conditioning vector + timestep embedding). No CLIP dependency.
import numpy as np
import torch
import torch.nn as nn


class DiTAdaLayerNorm(nn.Module):
    def __init__(self, feature_dim, epsilon=1e-6):
        super(DiTAdaLayerNorm, self).__init__()
        self.epsilon = epsilon
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, feature_dim * 2))

    def __call__(self, x, condition):
        """
        Args:
            x: (batch_size, feature_dim)
            condition: (batch_size, feature_dim)
                Ps: condition = time_cond_embedding + class_cond_embedding
        """
        affine = condition @ self.weight  # shape: (batch_size, feature_dim * 2)
        gamma, beta = torch.split(affine, affine.shape[-1] // 2, dim=-1)
        _mean = torch.mean(x, dim=-1, keepdims=True)
        _var = torch.var(x, dim=-1, keepdims=True)
        # NOTE: deliberately fixed relative to upstream MoMADiff, whose
        # `x - mean / (var + eps)` mis-parenthesization breaks shift
        # invariance and explodes on low-variance activations. This head is
        # trained from scratch here, so no checkpoint compatibility is lost.
        x_layer_norm = gamma * ((x - _mean) / torch.sqrt(_var + self.epsilon)) + beta
        return x_layer_norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model; the table is indexed by TimestepEmbedder
        raise NotImplementedError


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class MLP_Denoiser(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP_Denoiser, self).__init__()

        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()

        # build projector
        self.input_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.condition_embedder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        # build middle layers
        for _ in range(num_layers - 1):
            self.ln_layers.append(DiTAdaLayerNorm(hidden_dim))
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        # build last layer
        self.ln_layers.append(DiTAdaLayerNorm(hidden_dim))
        self.mlp_layers.append(nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        ))

    def forward(self, x, cond):
        x = self.input_embedder(x)
        cond = self.condition_embedder(cond)
        for i in range(self.num_layers):
            x = self.ln_layers[i](x, cond)
            x = self.mlp_layers[i](x)
        return x


class LDM(nn.Module):
    """Per-token latent diffusion head.

    forward contract (guided-diffusion model interface):
        x: [N, input_dim] noisy latent rows
        timesteps: [N] int
        y: {'text_embed': [N, cond_dim] per-token conditioning}
    returns [N, output_dim] x0 prediction.
    """

    def __init__(self, input_dim=512, cond_dim=512, ff_size=1024,
                 output_dim=512, num_layers=4, dropout=0.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.sequence_pos_encoder = PositionalEncoding(self.cond_dim, self.dropout)
        self.mlp = MLP_Denoiser(self.input_dim, self.cond_dim, self.ff_size,
                                self.output_dim, num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.cond_dim, self.sequence_pos_encoder)

    def forward(self, x, timesteps, y):
        timestep_emb = self.embed_timestep(timesteps)
        cond = y['text_embed']
        cond = cond + timestep_emb
        output = self.mlp(x, cond)
        return output
