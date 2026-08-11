# Ported from SOKE: mGPT/archs/tools/quantize_cnn.py, with two changes:
#
# 1. Resume safety: EMA/init state (`code_sum`, `code_count`, `initted`) is
#    registered as buffers so it lands in state_dict. The SOKE original kept
#    them as plain attributes, so a resumed run re-entered init_codebook() on
#    its first batch and overwrote the trained codebook.
# 2. Masking: forward() takes an optional `mask` of valid latent steps
#    ([N, T], 1 = real frame content). Padded steps (last-frame repeats from
#    collate/temporal padding) are still quantized and decoded, but they are
#    excluded from codebook init, EMA updates, commitment loss, and perplexity.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _flat_valid(mask):
    """[N, T] mask -> flat bool index aligned with preprocess(); None if empty."""
    if mask is None:
        return None
    valid = mask.reshape(-1).bool()
    if not bool(valid.any()):
        return None
    return valid


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.register_buffer('codebook', torch.zeros(nb_code, code_dim))
        self.register_buffer('code_sum', torch.zeros(nb_code, code_dim))
        self.register_buffer('code_count', torch.zeros(nb_code))
        self.register_buffer('initted', torch.zeros(()))

    @property
    def init(self):
        return bool(self.initted)

    def reset_codebook(self):
        self.codebook.zero_()
        self.code_sum.zero_()
        self.code_count.zero_()
        self.initted.zero_()

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    @torch.no_grad()
    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code].detach().clone()
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.initted = torch.ones((), device=self.codebook.device)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + \
            torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x, mask=None):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)
        valid = _flat_valid(mask)
        x_stats = x[valid].detach() if valid is not None else x.detach()

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x_stats)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        stats_idx = code_idx[valid] if valid is not None else code_idx

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x_stats, stats_idx)
        else:
            perplexity = self.compute_perplexity(stats_idx)

        # Loss (valid steps only)
        if valid is not None:
            commit_loss = F.mse_loss(x[valid], x_d[valid].detach())
        else:
            commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask=None):
        N, width, T = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        valid = _flat_valid(mask)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if valid is not None:
            loss = torch.mean((z_q[valid] - z[valid].detach()) ** 2) + self.beta * \
                torch.mean((z_q[valid].detach() - z[valid]) ** 2)
            stats_idx = min_encoding_indices[valid]
        else:
            loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * \
                torch.mean((z_q.detach() - z) ** 2)
            stats_idx = min_encoding_indices

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        min_encodings = F.one_hot(stats_idx, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):
        assert z.shape[-1] == self.e_dim

        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim,)).contiguous()
        return z_q

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x


class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))
        self.register_buffer('initted', torch.zeros(()))
        self.code_count = None

    @property
    def init(self):
        return bool(self.initted)

    def reset_codebook(self):
        self.initted.zero_()
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook.data.copy_(out[:self.nb_code].detach())
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.initted = torch.ones((), device=self.codebook.device)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_count = code_count
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + \
            torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x, mask=None):
        N, width, T = x.shape
        x = self.preprocess(x)
        valid = _flat_valid(mask)
        x_stats = x[valid].detach() if valid is not None else x.detach()

        if self.training and not self.init:
            self.init_codebook(x_stats)
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        stats_idx = code_idx[valid] if valid is not None else code_idx

        if self.training:
            perplexity = self.update_codebook(x_stats, stats_idx)
        else:
            perplexity = self.compute_perplexity(stats_idx)

        if valid is not None:
            commit_loss = F.mse_loss(x[valid], x_d[valid].detach())
        else:
            commit_loss = F.mse_loss(x, x_d.detach())

        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()

        return x_d, commit_loss, perplexity


class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.register_buffer('codebook', torch.zeros(nb_code, code_dim))
        self.register_buffer('code_sum', torch.zeros(nb_code, code_dim))
        self.register_buffer('code_count', torch.zeros(nb_code))
        self.register_buffer('initted', torch.zeros(()))

    @property
    def init(self):
        return bool(self.initted)

    def reset_codebook(self):
        self.codebook.zero_()
        self.code_sum.zero_()
        self.code_count.zero_()
        self.initted.zero_()

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    @torch.no_grad()
    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code].detach().clone()
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.initted = torch.ones((), device=self.codebook.device)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + \
            torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x, mask=None):
        N, width, T = x.shape

        x = self.preprocess(x)
        valid = _flat_valid(mask)
        x_stats = x[valid].detach() if valid is not None else x.detach()

        if self.training and not self.init:
            self.init_codebook(x_stats)

        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        stats_idx = code_idx[valid] if valid is not None else code_idx

        if self.training:
            perplexity = self.update_codebook(x_stats, stats_idx)
        else:
            perplexity = self.compute_perplexity(stats_idx)

        if valid is not None:
            commit_loss = F.mse_loss(x[valid], x_d[valid].detach())
        else:
            commit_loss = F.mse_loss(x, x_d.detach())

        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()

        return x_d, commit_loss, perplexity
