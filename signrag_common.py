"""Shared pieces of the SignRAG-style retriever experiment
(doc/signrag_style_retriever_plan.md).

Text normalisation, cached ragged features, the alignment head phi, the
retrieval filters, the soft-alignment scoring kernel, the frozen text
caches (official mT5-base text tower; paraphrase model for semantic
positives) and the multi-positive InfoNCE.  Used by signrag_retrieval.py
(extract / retrieve / evaluate) and signrag_contrastive.py (Step 2).
"""
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import utils
from config import dev_label_paths, test_label_paths, train_label_paths

DEFAULT_OUT = "/data/hwu/Uni_sign_repo/out/signrag_retrieval"
LABEL_PATHS = {"train": train_label_paths, "dev": dev_label_paths, "test": test_label_paths}


# --------------------------------------------------------------------------- #
#  Text / clip keys
# --------------------------------------------------------------------------- #
_CJK = re.compile(r"[一-鿿]")
_CSL_NAME = re.compile(r"S\d{6}_P\d{4}_T\d{2}")


def norm_text(text):
    """Normalisation behind every 'identical text' statistic and filter."""
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def is_cjk(text):
    return bool(_CJK.search(text))


def text_tokens(text):
    """Tokens of norm_text: characters for Chinese (CSL-Daily), else words."""
    t = norm_text(text)
    return list(t.replace(" ", "")) if is_cjk(t) else t.split()


def bleu_text(text):
    """Text as fine_tuning.evaluate scores it: CSL-Daily is scored per character."""
    if not is_cjk(text):
        return text
    return " ".join(text.replace("，", ",").replace("？", "?").replace(" ", "").replace("\n", ""))


def video_id(name):
    # How2Sign and OpenASL clip names start with the 11-character YouTube ID;
    # a CSL-Daily clip (S<sentence>_P<signer>_T<take>) is its own video
    return name if _CSL_NAME.match(name) else name[:11]


def label_path(dataset, split):
    path = LABEL_PATHS[split][dataset]
    if not path:
        raise ValueError(f"{dataset} has no {split} split")
    return path


def same_video_default(dataset):
    # How2Sign test videos are disjoint from train, so same-video neighbours
    # never exist at test time; every OpenASL test video also occurs in train
    return dataset == "How2Sign"


def write_json(path, obj, **kwargs):
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)


# --------------------------------------------------------------------------- #
#  Ragged features
# --------------------------------------------------------------------------- #
def ragged_padded(X, offsets, ids, dim):
    """Rows of entries `ids` of a ragged [sum T, D] array -> zero-padded
    float16 tensor [n, L, D] plus a bool mask (True = valid)."""
    lengths = (offsets[ids + 1] - offsets[ids]).astype(np.int64)
    L = int(lengths.max())
    x = np.zeros((len(ids), L, dim), dtype=np.float16)
    for k, i in enumerate(ids):
        x[k, :lengths[k]] = X[offsets[i]:offsets[i + 1]]
    mask = np.arange(L)[None] < lengths[:, None]
    return torch.from_numpy(x), torch.from_numpy(mask)


def gather_padded(X, offsets, ids):
    """Same as ragged_padded for tensors already on a device. Padded
    positions repeat the entry's last row and are masked out."""
    start = offsets[ids]
    length = offsets[ids + 1] - start
    ar = torch.arange(int(length.max()), device=offsets.device)
    mask = ar[None] < length[:, None]
    idx = start[:, None] + torch.minimum(ar[None], length[:, None] - 1)
    return X[idx], mask


class FeatSplit:
    """Cached frame features of one split, as written by `extract`."""

    def __init__(self, split_dir, feature="X"):
        with open(os.path.join(split_dir, "meta.json")) as f:
            self.meta = json.load(f)
        if feature not in self.meta["features"]:
            raise KeyError(f"{split_dir} has no '{feature}' features "
                           f"(have {sorted(self.meta['features'])})")
        self.items = self.meta["items"]
        self.names = [it["name"] for it in self.items]
        self.texts = [it["text"] for it in self.items]
        self.videos = [it["video_id"] for it in self.items]
        self.offsets = np.load(os.path.join(split_dir, "offsets.npy"))
        self.dim = self.meta["features"][feature]
        self.X = np.memmap(os.path.join(split_dir, f"{feature}.f16"), dtype=np.float16,
                           mode="r", shape=(int(self.offsets[-1]), self.dim))

    def __len__(self):
        return len(self.items)

    @property
    def lengths(self):
        return np.diff(self.offsets)

    def padded(self, ids):
        return ragged_padded(self.X, self.offsets, np.asarray(ids), self.dim)


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


# --------------------------------------------------------------------------- #
#  Alignment head phi
# --------------------------------------------------------------------------- #
class Phi(nn.Module):
    """Alignment head. `linear` has the shape of Uni-Sign's pose_proj
    (1024 -> 768) and is initialised from it; the optional residual MLP has a
    zero-initialised output layer, so phi(X) == pose_proj(X) at init."""

    def __init__(self, in_dim=1024, out_dim=768, mlp_hidden=0):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "out_dim": out_dim, "mlp_hidden": mlp_hidden}
        self.linear = nn.Linear(in_dim, out_dim)
        self.mlp = None
        if mlp_hidden:
            self.mlp = nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.GELU(),
                                     nn.Linear(mlp_hidden, out_dim))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        y = self.linear(x)
        return y if self.mlp is None else y + self.mlp(x)


def pose_proj_phi(feat_dir, mlp_hidden=0):
    """phi initialised from the checkpoint's pose_proj (saved by `extract`)."""
    proj = torch.load(os.path.join(feat_dir, "pose_proj.pt"), map_location="cpu")
    out_dim, in_dim = proj["weight"].shape
    phi = Phi(in_dim, out_dim, mlp_hidden)
    with torch.no_grad():
        phi.linear.weight.copy_(proj["weight"])
        phi.linear.bias.copy_(proj["bias"])
    return phi


def identity_linear(dim):
    head = nn.Linear(dim, dim)
    with torch.no_grad():
        head.weight.copy_(torch.eye(dim))
        head.bias.zero_()
    return head


def load_phi_checkpoint(path):
    """phi.pth from signrag_contrastive.py -> (phi, text_head, checkpoint)."""
    ckpt = torch.load(path, map_location="cpu")
    phi = Phi(**ckpt["phi_cfg"])
    phi.load_state_dict(ckpt["phi"])
    head = identity_linear(ckpt["text_head"]["weight"].shape[0])
    head.load_state_dict(ckpt["text_head"])
    return phi, head, ckpt


# --------------------------------------------------------------------------- #
#  Scoring
# --------------------------------------------------------------------------- #
def soft_match_scores(q, q_mask, g, g_mask, tau=None):
    """SignRAG soft-alignment score for every (query, gallery) pair.

    q [b, Lq, D], g [n, Lg, D], rows L2-normalised; masks True = valid.
    E_ij = q_i . g_j,  P = softmax_j(E / tau),  s = mean_i sum_j P_ij E_ij.
    tau=None is the retrieval score (Eq. 5-8, no temperature); with tau it is
    the Stage-I sign->text score (Eq. 21-23).  Returns float32 [b, n].
    """
    E = torch.einsum("bqd,nld->bnql", q, g)
    logits = E if tau is None else E / tau
    P = logits.masked_fill(~g_mask[None, :, None, :], float("-inf")).softmax(-1)
    per_frame = (P * E).sum(-1, dtype=torch.float32)
    qm = q_mask[:, None, :].float()
    return (per_frame * qm).sum(-1) / qm.sum(-1).clamp(min=1)


def soft_match_pair_scores(f, f_mask, t, t_mask, tau, chunk=0):
    """Both Stage-I directions from one similarity tensor, for B paired
    clips/texts: s2t[a, b] = s(S_a, T_b) (frames attend over tokens, mean
    over frames) and t2s[a, b] = s(T_a, S_b) (tokens attend over frames,
    mean over tokens).

    The [Bs, Bt, M, L] tensor grows with B^2. chunk > 0 processes `chunk`
    texts at a time under activation checkpointing: identical scores and
    gradients, peak memory proportional to Bs * chunk instead of Bs * Bt."""
    if not chunk or chunk >= t.size(0):
        return _pair_scores(f, f_mask, t, t_mask, tau)
    from torch.utils.checkpoint import checkpoint

    parts = [checkpoint(_pair_scores, f, f_mask, t[s:s + chunk], t_mask[s:s + chunk], tau,
                        use_reentrant=False)
             for s in range(0, t.size(0), chunk)]
    return torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 0)


def _pair_scores(f, f_mask, t, t_mask, tau):
    E = torch.einsum("amd,bld->abml", f, t)                  # [Bs, Bt, M, L]
    neg = float("-inf")
    P = (E / tau).masked_fill(~t_mask[None, :, None, :], neg).softmax(-1)
    fm = f_mask[:, None, :].float()
    s2t = ((P * E).sum(-1, dtype=torch.float32) * fm).sum(-1) / fm.sum(-1).clamp(min=1)
    P = (E / tau).masked_fill(~f_mask[:, None, :, None], neg).softmax(2)
    tm = t_mask[None, :, :].float()
    t2s = ((P * E).sum(2, dtype=torch.float32) * tm).sum(-1) / tm.sum(-1).clamp(min=1)
    return s2t, t2s.t()


@torch.no_grad()
def bidirectional_recall(phi, head, texts, sem, sign_batches, objective="pooled", tau_attn=None,
                         theta=0.85, ks=(1, 5, 10), device="cuda"):
    """Within-split retrieval in both directions (the standard V2T / T2V
    protocol). sign_batches yields (X [b, T, 1024] on device, mask, texts).

    sign->text (r@K): gallery = the split's unique texts, hit = own text.
    text->sign (t2s_r@K): queries = those texts, gallery = the split's clips,
    hit = any clip with that text. sem_* variants count a paraphrase with
    cosine >= theta as a hit. Scores use the objective's own similarity."""
    zs, fs, tids = [], [], []
    for x, m, txt in sign_batches:
        tids.extend(texts.ids(txt))
        f = phi(x)
        if objective == "pooled":
            zs.append(F.normalize(masked_mean(f, m), dim=-1))
        else:
            fs.append((F.normalize(f, dim=-1).half(), m))
    tids = np.asarray(tids)
    u, col = np.unique(tids, return_inverse=True)
    if objective == "pooled":
        zt = F.normalize(head(torch.as_tensor(texts.pooled[u], device=device).float()), dim=-1)
        s2t = torch.cat(zs) @ zt.t()                                    # [N, U]
        t2s = s2t.t()
    else:
        tok = []
        for g in range(0, len(u), 256):
            t, tm = texts.padded_tokens(u[g:g + 256])
            tok.append((F.normalize(head(t.to(device).float()), dim=-1).half(), tm.to(device)))
        s2t = torch.cat([torch.cat([soft_match_scores(f, m, t, tm, tau_attn) for t, tm in tok], 1)
                         for f, m in fs])                               # [N, U]
        t2s = torch.cat([torch.cat([soft_match_scores(t, tm, f, m, tau_attn) for t, tm in tok], 0)
                         for f, m in fs], 1)                            # [U, N]
    K = max(ks)
    top_t = s2t.topk(min(K, len(u)), dim=-1).indices.cpu().numpy()     # text columns per clip
    top_c = t2s.topk(min(K, len(tids)), dim=-1).indices.cpu().numpy()  # clip rows per text
    S = sem.pooled.astype(np.float32)
    sem_st = (S[u[top_t]] * S[tids][:, None]).sum(-1) >= theta
    sem_ts = (S[tids[top_c]] * S[u][:, None]).sum(-1) >= theta
    out = {}
    for k in ks:
        out[f"r{k}"] = 100 * float((top_t[:, :k] == col[:, None]).any(1).mean())
        out[f"t2s_r{k}"] = 100 * float((tids[top_c[:, :k]] == u[:, None]).any(1).mean())
        out[f"sem_r{k}"] = 100 * float(sem_st[:, :k].any(1).mean())
        out[f"t2s_sem_r{k}"] = 100 * float(sem_ts[:, :k].any(1).mean())
    out["r5_mean"] = (out["r5"] + out["t2s_r5"]) / 2
    out["sem_r5_mean"] = (out["sem_r5"] + out["t2s_sem_r5"]) / 2
    return out


def multi_positive_nce(logits, pos, mode="any"):
    """InfoNCE with a set of positives per row (pos: bool, >= 1 True per row).
    any: -log sum_{pos} p   (MIL-NCE: retrieving any positive is a hit)
    all: mean_{pos} -log p  (SupCon: pull towards every positive)."""
    if mode == "any":
        return (logits.logsumexp(-1)
                - logits.masked_fill(~pos, float("-inf")).logsumexp(-1)).mean()
    logp = logits.log_softmax(-1)
    return -((logp * pos).sum(-1) / pos.sum(-1)).mean()


# --------------------------------------------------------------------------- #
#  Retrieval filters (§4.4)
# --------------------------------------------------------------------------- #
class Gallery:
    """Gallery entries plus the integer keys the filters compare."""

    def __init__(self, names, texts, videos):
        self.names, self.texts, self.videos = names, texts, videos
        self._text_id, self._video_id = {}, {}
        self.text_gid = np.array([self._text_id.setdefault(norm_text(t), len(self._text_id))
                                  for t in texts])
        self.video_gid = np.array([self._video_id.setdefault(v, len(self._video_id))
                                   for v in videos])
        self._index = {n: i for i, n in enumerate(names)}

    def __len__(self):
        return len(self.names)

    def query_keys(self, names, texts, videos, same_split):
        """(text key, video key, own gallery index) per query; -1 = absent."""
        qt = np.array([self._text_id.get(norm_text(t), -1) for t in texts])
        qv = np.array([self._video_id.get(v, -1) for v in videos])
        qs = np.array([self._index.get(n, -1) if same_split else -1 for n in names])
        return qt, qv, qs


class Exclusion:
    """[b, N] mask of gallery entries a query may not retrieve (True =
    excluded). Only training queries are filtered: self, identical
    normalised text and, optionally, the same source video."""

    def __init__(self, gallery, q_keys, active, same_video, device):
        self.active, self.same_video, self.device = active, same_video, device
        self.n = len(gallery)
        self.g_text = torch.as_tensor(gallery.text_gid, device=device)
        self.g_video = torch.as_tensor(gallery.video_gid, device=device)
        self.q_text, self.q_video, self.q_self = (torch.as_tensor(k, device=device)
                                                  for k in q_keys)

    def mask(self, rows):
        if not self.active:
            return torch.zeros(len(rows), self.n, dtype=torch.bool, device=self.device)
        m = self.g_text[None] == self.q_text[rows][:, None]
        if self.same_video:
            m |= self.g_video[None] == self.q_video[rows][:, None]
        own = self.q_self[rows]
        has = (own >= 0).nonzero(as_tuple=True)[0]
        m[has, own[has]] = True
        return m


# --------------------------------------------------------------------------- #
#  Frozen text caches (one per dataset, keyed by normalised text)
# --------------------------------------------------------------------------- #
def collect_texts(dataset):
    """Unique normalised texts of every labelled split in a fixed order
    (train, dev, test; first occurrence) with one raw representative each."""
    keys, raws, seen = [], [], set()
    for split in ("train", "dev", "test"):
        path = LABEL_PATHS[split][dataset]
        if not path:
            continue
        for item in utils.load_dataset_file(path).values():
            key = norm_text(item["text"])
            if key not in seen:
                seen.add(key)
                keys.append(key)
                raws.append(item["text"])
    return keys, raws


class TextCache:
    """kind 'mt5': official mT5-base encoder token states (ragged) and their
    masked mean -- the contrastive text tower.  kind 'sem': L2-normalised
    paraphrase-model embeddings -- only used to define semantic positives."""

    def __init__(self, cache_dir):
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.dir = os.path.abspath(cache_dir)
        self.keys = self.meta["keys"]
        self.key_to_id = {k: i for i, k in enumerate(self.keys)}
        self.pooled = np.load(os.path.join(cache_dir, "pooled.npy"))
        self.tokens = None
        if self.meta["kind"] == "mt5":
            self.tok_offsets = np.load(os.path.join(cache_dir, "tok_offsets.npy"))
            self.tokens = np.memmap(os.path.join(cache_dir, "tokens.f16"), dtype=np.float16,
                                    mode="r", shape=(int(self.tok_offsets[-1]), self.pooled.shape[1]))

    def ids(self, texts):
        return np.array([self.key_to_id[norm_text(t)] for t in texts])

    def padded_tokens(self, ids):
        return ragged_padded(self.tokens, self.tok_offsets, np.asarray(ids), self.pooled.shape[1])


def load_checkpoint_mt5_encoder(enc, ckpt_path):
    """Overwrite an MT5EncoderModel with the mT5 encoder of a Uni-Sign
    checkpoint ({'model': state_dict} with 'mt5_model.*' keys)."""
    sd = torch.load(ckpt_path, map_location="cpu")["model"]
    enc.load_state_dict({k[len("mt5_model."):]: v.float() for k, v in sd.items()
                         if k.startswith("mt5_model.encoder.") or k == "mt5_model.shared.weight"}, strict=True)
    return enc


@torch.no_grad()
def build_mt5_text_cache(dataset, cache_dir, mt5_path, layer=-1, batch_size=256, device="cuda",
                         encoder_ckpt=None):
    """encoder_ckpt=None: official mT5 weights; otherwise the translation-
    finetuned mT5 encoder of that Uni-Sign checkpoint (same tokenizer)."""
    from transformers import MT5EncoderModel, T5Tokenizer

    keys, raws = collect_texts(dataset)
    tok = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
    enc = MT5EncoderModel.from_pretrained(mt5_path, torch_dtype=torch.float32)
    if encoder_ckpt:
        enc = load_checkpoint_mt5_encoder(enc, encoder_ckpt)
    enc = enc.to(device).eval()
    dim = enc.config.d_model
    pooled = np.zeros((len(keys), dim), dtype=np.float16)
    tokens, lengths = [None] * len(keys), np.zeros(len(keys), dtype=np.int64)
    order = np.argsort([len(r) for r in raws], kind="stable")
    for s in range(0, len(order), batch_size):
        ids = order[s:s + batch_size]
        batch = tok([raws[i] for i in ids], padding="longest", truncation=True,
                    max_length=50, return_tensors="pt").to(device)
        out = enc(**batch, output_hidden_states=layer != -1)
        h = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
        m = batch["attention_mask"].bool()
        pooled[ids] = masked_mean(h, m).half().cpu().numpy()
        h, m = h.half().cpu().numpy(), m.cpu().numpy()
        for k, i in enumerate(ids):
            tokens[i] = h[k, m[k]]
            lengths[i] = m[k].sum()
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "tokens.f16"), "wb") as f:
        for t in tokens:
            f.write(t.tobytes())
    np.save(os.path.join(cache_dir, "tok_offsets.npy"), np.concatenate([[0], np.cumsum(lengths)]))
    np.save(os.path.join(cache_dir, "pooled.npy"), pooled)
    write_json(os.path.join(cache_dir, "meta.json"),
               {"kind": "mt5", "dataset": dataset, "model": os.path.abspath(mt5_path),
                "encoder_ckpt": os.path.abspath(encoder_ckpt) if encoder_ckpt else None,
                "layer": layer, "keys": keys})
    return TextCache(cache_dir)


@torch.no_grad()
def build_sem_cache(dataset, cache_dir, model_name, batch_size=512, device="cuda"):
    """Mean-pooled, L2-normalised sentence embeddings (sentence-transformers
    convention for the paraphrase-* / all-* models)."""
    from transformers import AutoModel, AutoTokenizer

    keys, raws = collect_texts(dataset)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    emb = []
    for s in range(0, len(raws), batch_size):
        batch = tok(raws[s:s + batch_size], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(device)
        h = model(**batch).last_hidden_state
        emb.append(F.normalize(masked_mean(h, batch["attention_mask"].bool()), dim=-1).half().cpu())
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "pooled.npy"), torch.cat(emb).numpy())
    write_json(os.path.join(cache_dir, "meta.json"),
               {"kind": "sem", "dataset": dataset, "model": model_name, "keys": keys})
    return TextCache(cache_dir)


def text_cache_dir(cache_root, dataset, kind, name):
    tag = name if kind == "mt5" else name.replace("/", "__")
    return os.path.join(cache_root, dataset, f"{kind}_{tag}")


def get_mt5_cache(cache_root, dataset, mt5_path, layer, device="cuda", encoder_ckpt=None):
    tag = f"layer{layer}" if not encoder_ckpt else \
        f"ckpt-{os.path.splitext(os.path.basename(encoder_ckpt))[0]}_layer{layer}"
    path = text_cache_dir(cache_root, dataset, "mt5", tag)
    if os.path.isfile(os.path.join(path, "meta.json")):
        return TextCache(path)
    print(f"building mT5 text cache -> {path}")
    return build_mt5_text_cache(dataset, path, mt5_path, layer, device=device, encoder_ckpt=encoder_ckpt)


def get_sem_cache(cache_root, dataset, model_name, device="cuda"):
    path = text_cache_dir(cache_root, dataset, "sem", model_name)
    if os.path.isfile(os.path.join(path, "meta.json")):
        return TextCache(path)
    print(f"building semantic text cache -> {path}")
    return build_sem_cache(dataset, path, model_name, device=device)


# --------------------------------------------------------------------------- #
#  Semantic positives
# --------------------------------------------------------------------------- #
@torch.no_grad()
def semantic_neighbours(emb, threshold, max_pos, device, chunk=4096):
    """For each row of L2-normalised emb [U, d]: the rows with cosine >=
    threshold (self always first, at most max_pos, most similar first), as
    CSR (indptr [U + 1], indices) on `device`."""
    E = (emb if torch.is_tensor(emb) else torch.as_tensor(np.asarray(emb))).to(device).float()
    k = min(max_pos, E.size(0))
    counts, indices = [], []
    for s in range(0, E.size(0), chunk):
        sim = E[s:s + chunk] @ E.t()
        r = torch.arange(sim.size(0), device=device)
        sim[r, r + s] = 2.0
        val, idx = sim.topk(k, dim=-1)
        keep = val >= threshold
        counts.append(keep.sum(-1))
        indices.append(idx[keep])                    # row-major: per-row order kept
    counts = torch.cat(counts)
    indptr = torch.zeros(len(counts) + 1, dtype=torch.long, device=device)
    indptr[1:] = counts.cumsum(0)
    return indptr, torch.cat(indices)


def csr_rows_mask(indptr, indices, rows, n_cols):
    """Dense bool [len(rows), n_cols] with the CSR neighbours of `rows`."""
    start = indptr[rows]
    counts = indptr[rows + 1] - start
    r = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), counts)
    first = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    cols = indices[torch.repeat_interleave(start, counts) + torch.arange(len(r), device=rows.device) - first]
    m = torch.zeros(len(rows), n_cols, dtype=torch.bool, device=rows.device)
    m[r, cols] = True
    return m
