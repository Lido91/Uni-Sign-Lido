"""SignRAG-style retriever on Uni-Sign features (doc/signrag_style_retriever_plan.md).

    extract   frozen Uni-Sign checkpoint -> cached frame features
              X = ST-GCN parts + part_para (T x 1024); Z = X W^T + b is the
              baseline mT5 input and is derived on the fly
    retrieve  top-K training clips per query with
                frame : SignRAG soft-alignment sign->sign score (Eq. 5-8)
                pool  : cosine of masked-mean features (control)
                s2t   : sign->text with a trained phi (Step 3)
              plus random / TF-IDF-oracle baselines on identical queries and
              filters
    evaluate  retrieval-only BLEU-4 / ROUGE-L, token F1, exact match,
              same video, train-test gap, gap closed -> metrics.json/.tsv

    python signrag_retrieval.py extract  --dataset How2Sign --finetune CKPT --splits train test --feat_dir DIR
    python signrag_retrieval.py retrieve --feat_dir DIR --score frame pool --queries test train:3000 --run_dir RUN
    python signrag_retrieval.py evaluate --run_dir RUN
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import utils
from datasets import S2T_Dataset, load_part_kp
from signrag_common import (Exclusion, FeatSplit, Gallery, TextCache, bidirectional_recall, bleu_text, get_sem_cache,
                            is_cjk, label_path, load_phi_checkpoint, masked_mean, norm_text, pose_proj_phi,
                            same_video_default, soft_match_scores, text_tokens, video_id, write_json)


# =========================================================================== #
#  extract
# =========================================================================== #
class S2T_Dataset_Uniform(S2T_Dataset):
    """S2T_Dataset with repeatable frames: clips longer than --max_length are
    subsampled uniformly instead of with random.sample, so cached features do
    not depend on the RNG. Otherwise identical to S2T_Dataset.load_pose."""

    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", ".pkl")), "rb"))
        if "start" in pose.keys():
            assert pose["start"] < pose["end"]
            duration, start = pose["end"] - pose["start"], pose["start"]
        else:
            duration, start = len(pose["scores"]), 0
        if duration > self.max_length:
            tmp = np.linspace(0, duration - 1, self.max_length).round().astype(int)
        else:
            tmp = np.arange(duration)
        tmp = tmp + start
        skeletons = [pose["keypoints"][i] for i in tmp]
        confs = [pose["scores"][i] for i in tmp]
        return load_part_kp(skeletons, confs, force_ok=True), {}


def build_unisign(dataset, finetune, max_length=256):
    from models import Uni_Sign

    uargs = utils.get_args_parser().parse_args(
        ["--dataset", dataset, "--max_length", str(max_length)])
    model = Uni_Sign(uargs)
    state = torch.load(finetune, map_location="cpu")["model"]
    model.load_state_dict(state, strict=True)
    return model, uargs


def encode_with_mt5(model, Z, pose_mask):
    """F2 features: mT5 encoder states at the pose positions of the baseline
    input [translate prompt][pose]."""
    prefix = model.mt5_tokenizer([f"Translate sign language video to {model.lang}: "] * Z.size(0),
                                 padding="longest", return_tensors="pt").to(Z.device)
    emb = model.mt5_model.encoder.embed_tokens(prefix["input_ids"])
    h = model.mt5_model.encoder(
        inputs_embeds=torch.cat([emb, Z.to(emb.dtype)], 1),
        attention_mask=torch.cat([prefix["attention_mask"], pose_mask], 1),
    ).last_hidden_state
    return h[:, prefix["input_ids"].size(1):].float()


@torch.no_grad()
def extract_split(model, uargs, split, args):
    out_dir = os.path.join(args.feat_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    ds = S2T_Dataset_Uniform(path=label_path(args.dataset, split), args=uargs, phase=split)
    if args.limit:
        ds.list = ds.list[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=ds.collate_fn, shuffle=False)
    proj = model.pose_proj
    keys = ["X"] + (["enc"] if args.save_enc else [])
    files = {k: open(os.path.join(out_dir, f"{k}.f16.tmp"), "wb") for k in keys}
    items, t0 = [], time.time()
    for step, (src, tgt) in enumerate(loader):
        for k, v in src.items():
            if torch.is_tensor(v):
                src[k] = v.cuda(non_blocking=True)
                if v.is_floating_point():
                    src[k] = src[k].float()
        model.pose_proj = nn.Identity()
        X = model.encode_pose(src)                      # [B, T, 1024]
        model.pose_proj = proj
        if step == 0:
            err = (proj(X) - model.encode_pose(src)).abs().max().item()
            print(f"[{split}] check pose_proj(X) == encode_pose: max |diff| = {err:.2e}")
            assert err < 1e-4, "X is not the input of pose_proj"
        feats = {"X": X.half().cpu().numpy()}
        if args.save_enc:
            feats["enc"] = encode_with_mt5(model, proj(X), src["attention_mask"]).half().cpu().numpy()
        lengths = src["src_length_batch"].tolist()
        for i, name in enumerate(src["name_batch"]):
            for k in keys:
                files[k].write(feats[k][i, :lengths[i]].tobytes())
            items.append({"name": name, "text": tgt["gt_sentence"][i],
                          "video_id": video_id(name), "length": lengths[i]})
        if step % 100 == 0:
            print(f"[{split}] {len(items)}/{len(ds)} clips  {time.time() - t0:.0f}s", flush=True)
    for k, f in files.items():
        f.close()
        os.replace(os.path.join(out_dir, f"{k}.f16.tmp"), os.path.join(out_dir, f"{k}.f16"))
    np.save(os.path.join(out_dir, "offsets.npy"),
            np.concatenate([[0], np.cumsum([it["length"] for it in items])]))
    # meta.json is written last: its presence marks a finished split
    write_json(os.path.join(out_dir, "meta.json"), {
        "dataset": args.dataset, "split": split, "finetune": os.path.abspath(args.finetune),
        "max_length": args.max_length, "sampling": "uniform",
        "features": {"X": int(X.shape[-1]), **({"enc": int(proj.out_features)} if args.save_enc else {})},
        "items": items,
    })
    print(f"[{split}] done: {len(items)} clips, {sum(it['length'] for it in items)} frames, "
          f"{time.time() - t0:.0f}s -> {out_dir}")


def cmd_extract(args):
    model, uargs = build_unisign(args.dataset, args.finetune, args.max_length)
    model.cuda().eval()
    os.makedirs(args.feat_dir, exist_ok=True)
    torch.save({"weight": model.pose_proj.weight.detach().float().cpu(),
                "bias": model.pose_proj.bias.detach().float().cpu(),
                "finetune": os.path.abspath(args.finetune)},
               os.path.join(args.feat_dir, "pose_proj.pt"))
    for split in args.splits:
        extract_split(model, uargs, split, args)


# =========================================================================== #
#  retrieve
# =========================================================================== #
def make_projector(args, device):
    """Frame projector for the frame/pool scores: identity for 'enc'
    features, pose_proj (Step 1) or a trained phi (Step 3) for 'X'."""
    if args.feature == "enc":
        if args.phi:
            raise ValueError("--phi applies to X features only")
        return nn.Identity()
    phi = load_phi_checkpoint(args.phi)[0] if args.phi else pose_proj_phi(args.feat_dir)
    return phi.to(device).eval()


def _frame_worker(rank, world, cfg, args, shard_path):
    """One GPU: exhaustive soft-alignment scores for its share of queries
    against the whole gallery, filters, top-K.

    kind  frames      : sign frames -> gallery sign frames (Eq. 5-8)
          text_tokens : sign frames -> gallery text tokens (fine phi, s2t)
          t2s_tokens  : query text tokens -> gallery sign frames (fine phi, t2s)
    """
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    gal = FeatSplit(cfg["gallery_dir"], args.feature)
    qs = FeatSplit(cfg["query_dir"], args.feature)
    gallery = Gallery(gal.names, gal.texts, gal.videos)
    excl = Exclusion(gallery, cfg["q_keys"], cfg["active"], cfg["same_video"], device)
    ids, kind = np.asarray(cfg["ids"]), cfg["kind"]

    with torch.no_grad():
        if kind == "frames":
            proj, head, tau = make_projector(args, device), None, args.frame_tau
        else:
            phi, head, ckpt = load_phi_checkpoint(args.phi)
            proj, head = phi.to(device).eval(), head.to(device).eval()
            texts = TextCache(ckpt["text_cache"])
            tau = ckpt["tau_attn"]
        clip_cols = None
        if kind == "text_tokens":                # gallery = unique train texts, scores expanded to clips
            uniq, inv = np.unique(texts.ids(gal.texts), return_inverse=True)
            g_len, g_proj = np.diff(texts.tok_offsets)[uniq], head
            g_get = lambda sel: texts.padded_tokens(uniq[sel])
            clip_cols = torch.as_tensor(inv, device=device)
        else:                                    # gallery = train clips
            g_len, g_proj, g_get = gal.lengths, proj, gal.padded
        if kind == "t2s_tokens":                 # queries = the query clips' texts
            q_tids = texts.ids([qs.texts[i] for i in ids])
            q_len, q_proj = np.diff(texts.tok_offsets)[q_tids], head
            q_get = lambda pos: texts.padded_tokens(q_tids[pos])
        else:
            q_len, q_proj = qs.lengths[ids], proj
            q_get = lambda pos: qs.padded(ids[pos])
        order = np.argsort(g_len, kind="stable")

        # gallery chunks, sorted by length so padding stays small
        chunks, free = [], torch.cuda.mem_get_info(device)[0]
        on_gpu = True
        for s in range(0, len(order), args.gallery_chunk):
            sel = order[s:s + args.gallery_chunk]
            x, m = g_get(sel)
            g = F.normalize(g_proj(x.to(device).float()), dim=-1).half()
            cols = torch.as_tensor(sel, device=device)
            m = m.to(device)
            if on_gpu and torch.cuda.memory_allocated(device) > 0.6 * free:
                on_gpu = False
                print(f"[gpu{rank}] gallery exceeds 60% of free memory; streaming from host")
            if not on_gpu:
                g, m = g.cpu().pin_memory(), m.cpu().pin_memory()
            chunks.append((cols, g, m))
        n_cols = len(order)

        q_order = np.argsort(q_len, kind="stable")
        batches = [q_order[s:s + args.query_batch] for s in range(0, len(q_order), args.query_batch)]
        out_pos, out_idx, out_val = [], [], []
        t0 = time.time()
        for bi, pos in enumerate(batches[rank::world]):
            x, qm = q_get(pos)
            q = F.normalize(q_proj(x.to(device).float()), dim=-1).half()
            qm = qm.to(device)
            scores = torch.full((len(pos), n_cols), float("-inf"), device=device)
            for cols, g, gm in chunks:
                scores[:, cols] = soft_match_scores(q, qm, g.to(device, non_blocking=True),
                                                    gm.to(device, non_blocking=True), tau)
            if clip_cols is not None:
                scores = scores[:, clip_cols]
            rows = torch.as_tensor(pos, device=device)
            scores.masked_fill_(excl.mask(rows), float("-inf"))
            val, idx = scores.topk(min(args.k, scores.size(1)), dim=-1)
            out_pos.append(pos)
            out_idx.append(idx.cpu().numpy())
            out_val.append(val.cpu().numpy())
            if rank == 0 and bi % 50 == 0:
                print(f"[gpu0] {bi}/{len(batches[rank::world])} query batches  {time.time() - t0:.0f}s",
                      flush=True)
    torch.save({"pos": np.concatenate(out_pos) if out_pos else np.zeros(0, int),
                "idx": np.concatenate(out_idx) if out_idx else np.zeros((0, args.k), int),
                "val": np.concatenate(out_val) if out_val else np.zeros((0, args.k))}, shard_path)


def run_sharded(cfg, args, n_queries):
    world = min(args.gpus or torch.cuda.device_count(), torch.cuda.device_count())
    shard_dir = os.path.join(args.run_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    paths = [os.path.join(shard_dir, f"{cfg['tag']}_{r}.pt") for r in range(world)]
    if world == 1:
        _frame_worker(0, 1, cfg, args, paths[0])
    else:
        torch.multiprocessing.spawn(_spawn_entry, args=(world, cfg, args, paths), nprocs=world, join=True)
    idx = np.zeros((n_queries, args.k), dtype=np.int64)
    val = np.full((n_queries, args.k), -np.inf)
    for p in paths:
        shard = torch.load(p)
        idx[shard["pos"]], val[shard["pos"]] = shard["idx"], shard["val"]
        os.remove(p)
    return idx, val


def _spawn_entry(rank, world, cfg, args, paths):
    _frame_worker(rank, world, cfg, args, paths[rank])


@torch.no_grad()
def pooled_vectors(feats, ids, proj, device, chunk=512):
    out = []
    for s in range(0, len(ids), chunk):
        x, m = feats.padded(ids[s:s + chunk])
        out.append(F.normalize(masked_mean(proj(x.to(device).float()), m.to(device)), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def topk_dense(score_fn, n_queries, excl, k, device, batch=1024):
    """score_fn(rows) -> [b, N] scores; filters; top-K over the gallery."""
    idx = np.zeros((n_queries, k), dtype=np.int64)
    val = np.full((n_queries, k), -np.inf)
    for s in range(0, n_queries, batch):
        rows = torch.arange(s, min(s + batch, n_queries), device=device)
        scores = score_fn(rows).float().masked_fill(excl.mask(rows), float("-inf"))
        v, i = scores.topk(min(k, scores.size(1)), dim=-1)
        idx[s:s + len(rows)], val[s:s + len(rows)] = i.cpu().numpy(), v.cpu().numpy()
    return idx, val


def tfidf_score_fn(gallery_texts, query_texts, device):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Chinese has no spaces: character uni/bigrams instead of the default word pattern
    cjk = dict(tokenizer=text_tokens, token_pattern=None, lowercase=False) if is_cjk(gallery_texts[0]) else {}
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, **cjk)
    G = vec.fit_transform(gallery_texts)
    Q = vec.transform(query_texts)

    def fn(rows):
        r = rows.cpu().numpy()
        return torch.as_tensor((Q[r] @ G.T).toarray(), device=device)
    return fn


def distinct_topk(idx, val, texts, k):
    """First k entries per row whose normalised texts differ."""
    out_i = np.zeros((len(idx), k), dtype=np.int64)
    out_v = np.full((len(idx), k), -np.inf)
    for r in range(len(idx)):
        seen, c = set(), 0
        for j, v in zip(idx[r], val[r]):
            if c == k or not np.isfinite(v):
                break
            key = norm_text(texts[j])
            if key not in seen:
                seen.add(key)
                out_i[r, c], out_v[r, c] = j, v
                c += 1
    return out_i, out_v


def save_results(run_dir, split, method, query_items, gallery, idx, val, excl_cfg):
    """retrieval_<split>_<method>.json (+ per-query CSV); asserts the
    filters held (smoke test S5)."""
    if excl_cfg.get("distinct_k"):
        idx, val = distinct_topk(idx, val, gallery.texts, excl_cfg["distinct_k"])
    results, rows = [], []
    for p, q in enumerate(query_items):
        top = [{"name": gallery.names[j], "text": gallery.texts[j], "video_id": gallery.videos[j],
                "score": round(float(v), 6)} for j, v in zip(idx[p], val[p]) if np.isfinite(v)]
        if excl_cfg["active"]:
            for t in top:
                bad = (t["name"] == q["name"] or norm_text(t["text"]) == norm_text(q["text"])
                       or (excl_cfg["same_video"] and t["video_id"] == q["video_id"]))
                assert not bad, f"filter violated for {q['name']} -> {t['name']}"
        results.append({"query": {k: q[k] for k in ("name", "text", "video_id")}, "topk": top})
        t1 = top[0] if top else {"name": "", "text": "", "video_id": "", "score": float("nan")}
        rows.append([q["name"], q["text"], t1["name"], t1["text"], t1["score"],
                     int(norm_text(t1["text"]) == norm_text(q["text"])),
                     int(t1["video_id"] == q["video_id"])])
    write_json(os.path.join(run_dir, f"retrieval_{split}_{method}.json"),
               {"split": split, "method": method, "results": results})
    with open(os.path.join(run_dir, f"per_query_{split}_{method}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "gt", "top1_name", "top1_text", "top1_score", "exact_match", "same_video"])
        w.writerows(rows)


def parse_queries(specs):
    out = []
    for spec in specs:
        split, _, n = spec.partition(":")
        out.append((split, int(n) if n else None))
    return out


def cmd_retrieve(args):
    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device("cuda", 0)
    gallery_dir = os.path.join(args.feat_dir, args.gallery_split)
    gal = FeatSplit(gallery_dir, args.feature)
    dataset = gal.meta["dataset"]
    same_video = {"auto": same_video_default(dataset), "on": True, "off": False}[args.same_video]
    gallery = Gallery(gal.names, gal.texts, gal.videos)
    phi_ckpt = load_phi_checkpoint(args.phi)[2] if args.phi else None
    if ({"s2t", "t2s"} & set(args.score) or args.pair_recall) and phi_ckpt is None:
        raise ValueError("--score s2t/t2s and --pair_recall need a trained --phi")
    fine = phi_ckpt is not None and phi_ckpt["objective"] == "fine"
    if phi_ckpt is not None and phi_ckpt.get("encoder") and gal.meta["finetune"] != phi_ckpt["encoder"]:
        raise ValueError(f"{args.phi} was trained together with its ST-GCN; re-extract features first: "
                         f"signrag_retrieval.py extract --finetune {phi_ckpt['encoder']} ...")
    write_json(os.path.join(args.run_dir, "retrieve_args.json"),
               {**vars(args), "dataset": dataset, "same_video_resolved": same_video}, indent=2)
    k_out = args.k
    if args.distinct_text:
        args.k = max(args.k, args.distinct_fetch)          # fetch more, keep k_out distinct texts
    print(f"{dataset}: gallery = {args.gallery_split} ({len(gal)} clips), same-video filter "
          f"for train queries: {same_video}")

    for qsplit, n in parse_queries(args.queries):
        query_dir = os.path.join(args.feat_dir, qsplit)
        qs = gal if qsplit == args.gallery_split else FeatSplit(query_dir, args.feature)
        ids = np.arange(len(qs))
        if n and n < len(qs):
            ids = np.sort(np.random.RandomState(args.seed).choice(len(qs), n, replace=False))
        same_split = qsplit == args.gallery_split
        q_items = [qs.items[i] for i in ids]
        q_keys = gallery.query_keys([q["name"] for q in q_items], [q["text"] for q in q_items],
                                    [q["video_id"] for q in q_items], same_split)
        excl_cfg = {"active": same_split and not args.no_filter, "same_video": same_video,
                    "distinct_k": k_out if args.distinct_text else 0}
        excl = Exclusion(gallery, q_keys, excl_cfg["active"], same_video, device)
        print(f"\n== queries: {qsplit} ({len(ids)}), filtered: {excl_cfg['active']}")

        for method in args.score:
            t0 = time.time()
            if method == "frame" or (method in ("s2t", "t2s") and fine):
                kind = {"frame": "frames", "s2t": "text_tokens", "t2s": "t2s_tokens"}[method]
                cfg = {"tag": f"{qsplit}_{method}", "kind": kind, "gallery_dir": gallery_dir,
                       "query_dir": query_dir, "ids": ids, "q_keys": q_keys, **excl_cfg}
                idx, val = run_sharded(cfg, args, len(ids))
            elif method == "pool":
                proj = make_projector(args, device)
                zg = pooled_vectors(gal, np.arange(len(gal)), proj, device)
                zq = pooled_vectors(qs, ids, proj, device)
                idx, val = topk_dense(lambda r: zq[r] @ zg.t(), len(ids), excl, args.k, device)
            elif method == "s2t":                          # pooled objective
                phi, head, ckpt = load_phi_checkpoint(args.phi)
                phi, head = phi.to(device).eval(), head.to(device).eval()
                texts = TextCache(ckpt["text_cache"])
                with torch.no_grad():
                    zt = F.normalize(head(torch.as_tensor(texts.pooled, device=device).float()), dim=-1)
                zt_clip = zt[torch.as_tensor(texts.ids(gal.texts), device=device)]
                zq = pooled_vectors(qs, ids, phi, device)
                idx, val = topk_dense(lambda r: zq[r] @ zt_clip.t(), len(ids), excl, args.k, device)
            elif method == "t2s":                          # pooled objective: sentence -> sign clips
                phi, head, ckpt = load_phi_checkpoint(args.phi)
                phi, head = phi.to(device).eval(), head.to(device).eval()
                texts = TextCache(ckpt["text_cache"])
                q_pool = texts.pooled[texts.ids([q["text"] for q in q_items])]
                with torch.no_grad():
                    zq = F.normalize(head(torch.as_tensor(q_pool, device=device).float()), dim=-1)
                zg = pooled_vectors(gal, np.arange(len(gal)), phi, device)
                idx, val = topk_dense(lambda r: zq[r] @ zg.t(), len(ids), excl, args.k, device)
            else:
                raise ValueError(method)
            save_results(args.run_dir, qsplit, method, q_items, gallery, idx, val, excl_cfg)
            print(f"   {method}: {time.time() - t0:.0f}s")

        if args.pair_recall:
            # within-split V2T / T2V recall (gallery = this query set itself)
            phi, head, ckpt = load_phi_checkpoint(args.phi)
            phi, head = phi.to(device).eval(), head.to(device).eval()
            ca = ckpt["args"]
            sem = get_sem_cache(ca["cache_root"], dataset, ca["sem_model"])

            def sign_batches():
                for s in range(0, len(ids), 64):
                    x, m = qs.padded(ids[s:s + 64])
                    yield x.to(device).float(), m.to(device), [qs.texts[i] for i in ids[s:s + 64]]
            pr = bidirectional_recall(phi, head, TextCache(ckpt["text_cache"]), sem, sign_batches(),
                                      ckpt["objective"], ckpt["tau_attn"], ckpt["pos_threshold"], device=device)
            write_json(os.path.join(args.run_dir, f"pair_recall_{qsplit}.json"),
                       {"split": qsplit, "n": len(ids), **pr}, indent=2)
            print(f"   pair recall: s2t R@1/5/10 {pr['r1']:.1f}/{pr['r5']:.1f}/{pr['r10']:.1f}, "
                  f"t2s R@1/5/10 {pr['t2s_r1']:.1f}/{pr['t2s_r5']:.1f}/{pr['t2s_r10']:.1f}")

        if args.baselines:
            for seed in range(args.random_seeds):
                gen = torch.Generator(device=device).manual_seed(seed)
                idx, val = topk_dense(lambda r: torch.rand(len(r), len(gal), generator=gen, device=device),
                                      len(ids), excl, args.k, device)
                save_results(args.run_dir, qsplit, f"random_s{seed}", q_items, gallery, idx, val, excl_cfg)
            fn = tfidf_score_fn(gal.texts, [q["text"] for q in q_items], device)
            idx, val = topk_dense(fn, len(ids), excl, args.k, device, batch=256)
            save_results(args.run_dir, qsplit, "tfidf_oracle", q_items, gallery, idx, val, excl_cfg)
            print(f"   baselines: random x{args.random_seeds}, tfidf_oracle")


# =========================================================================== #
#  evaluate
# =========================================================================== #
def _token_set(text, stop):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    toks = text_tokens(text)
    return {t for t in toks if not (stop and t in ENGLISH_STOP_WORDS)}


def _f1(gt, texts, stop):
    """SignRAG Eq. 20 token-overlap F1 of the union of `texts` against gt."""
    g = _token_set(gt, stop)
    r = set().union(*(_token_set(t, stop) for t in texts)) if texts else set()
    inter = len(g & r)
    if not inter:
        return 0.0
    p, rc = inter / len(r), inter / len(g)
    return 2 * p * rc / (p + rc)


def retrieval_metrics(results):
    from SLRT_metrics import translation_performance

    # the rouge package rejects inputs without word tokens; such texts are
    # replaced by a placeholder and counted
    fix = lambda t: t if norm_text(t) else "<empty>"
    refs = [r["query"]["text"] for r in results]
    hyps = [r["topk"][0]["text"] if r["topk"] else "" for r in results]
    n_fixed = sum(not norm_text(t) for t in refs + hyps)
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, rouge = translation_performance([bleu_text(fix(t)) for t in refs], [bleu_text(fix(t)) for t in hyps])
    top = lambda r, k: [t["text"] for t in r["topk"][:k]]
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "n": len(results), "bleu4": bleu["bleu4"], "rouge_l": rouge,
        "f1@1": 100 * mean([_f1(r["query"]["text"], top(r, 1), False) for r in results]),
        "f1@3": 100 * mean([_f1(r["query"]["text"], top(r, 3), False) for r in results]),
        "cf1@1": 100 * mean([_f1(r["query"]["text"], top(r, 1), True) for r in results]),
        "cf1@3": 100 * mean([_f1(r["query"]["text"], top(r, 3), True) for r in results]),
        "em@1": 100 * mean([bool(r["topk"]) and norm_text(r["topk"][0]["text"]) == norm_text(r["query"]["text"])
                            for r in results]),
        "same_video@1": 100 * mean([bool(r["topk"]) and r["topk"][0]["video_id"] == r["query"]["video_id"]
                                    for r in results]),
        "n_placeholder": n_fixed,
    }


METRIC_COLS = ["n", "bleu4", "rouge_l", "f1@1", "f1@3", "cf1@1", "cf1@3", "em@1", "same_video@1"]


def cmd_evaluate(args):
    per = {}
    for path in sorted(glob.glob(os.path.join(args.run_dir, "retrieval_*.json"))):
        with open(path) as f:
            data = json.load(f)
        per.setdefault(data["split"], {})[data["method"]] = retrieval_metrics(data["results"])
    if not per:
        raise FileNotFoundError(f"no retrieval_*.json in {args.run_dir}")

    for split, methods in per.items():
        seeds = [m for m in methods if m.startswith("random_s")]
        if seeds:
            rnd = {c: float(np.mean([methods[s][c] for s in seeds])) for c in METRIC_COLS}
            rnd.update({f"{c}_std": float(np.std([methods[s][c] for s in seeds])) for c in METRIC_COLS})
            rnd["n"] = methods[seeds[0]]["n"]
            for s in seeds:
                del methods[s]
            methods["random"] = rnd
        lo, hi = methods.get("random", {}).get("bleu4"), methods.get("tfidf_oracle", {}).get("bleu4")
        for name, m in methods.items():
            if lo is not None and hi is not None and hi > lo and name not in ("random", "tfidf_oracle"):
                m["gap_closed_bleu4"] = (m["bleu4"] - lo) / (hi - lo)
    if "train" in per:
        for split in per:
            if split == "train":
                continue
            for name, m in per[split].items():
                if name in per["train"]:
                    m["train_test_gap_bleu4"] = per["train"][name]["bleu4"] - m["bleu4"]

    pair = {}
    for path in sorted(glob.glob(os.path.join(args.run_dir, "pair_recall_*.json"))):
        with open(path) as f:
            d = json.load(f)
        pair[d["split"]] = d
    write_json(os.path.join(args.run_dir, "metrics.json"),
               {**per, **({"pair_recall": pair} if pair else {})}, indent=2)
    if pair:
        pcols = ["split", "n", "r1", "r5", "r10", "t2s_r1", "t2s_r5", "t2s_r10",
                 "sem_r5", "t2s_sem_r5", "r5_mean"]
        plines = ["\t".join(pcols)] + [
            "\t".join(f"{pair[s][c]:.2f}" if isinstance(pair[s][c], float) else str(pair[s][c]) for c in pcols)
            for s in sorted(pair)]
        with open(os.path.join(args.run_dir, "pair_recall.tsv"), "w") as f:
            f.write("\n".join(plines) + "\n")
        print("\n".join(plines) + "\n")
    cols = ["split", "method"] + METRIC_COLS + ["gap_closed_bleu4", "train_test_gap_bleu4"]
    lines = ["\t".join(cols)]
    for split in sorted(per):
        for name in sorted(per[split]):
            m = per[split][name]
            vals = [split, name] + [m.get(c, "-") for c in cols[2:]]
            lines.append("\t".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals))
    with open(os.path.join(args.run_dir, "metrics.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# =========================================================================== #
def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="cache frame features from a frozen Uni-Sign checkpoint")
    p.add_argument("--dataset", required=True, choices=["How2Sign", "OpenASL", "CSL_Daily"])
    p.add_argument("--finetune", required=True, help="Uni-Sign checkpoint ({'model': state_dict})")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--save_enc", action="store_true", help="also cache F2 (mT5 encoder states)")
    p.add_argument("--limit", type=int, default=0, help="first N clips only (smoke tests)")

    p = sub.add_parser("retrieve", help="top-K retrieval + baselines")
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gallery_split", default="train")
    p.add_argument("--queries", nargs="+", default=["test", "train:3000"],
                   help="split or split:N (seeded random sample)")
    p.add_argument("--score", nargs="+", default=["frame", "pool"], choices=["frame", "pool", "s2t", "t2s"],
                   help="frame/pool: sign->sign; s2t: sign->text; t2s: text->sign (s2t/t2s need --phi)")
    p.add_argument("--pair_recall", action="store_true",
                   help="also within-split V2T/T2V R@K for every query set (needs --phi)")
    p.add_argument("--feature", default="X", choices=["X", "enc"])
    p.add_argument("--phi", default="", help="trained phi.pth (Step 3); default: pose_proj")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--distinct_text", action="store_true",
                   help="top-K distinct texts: CSL-Daily repeats each sentence over signers, so the "
                        "top-K clips of a text score are often one sentence K times")
    p.add_argument("--distinct_fetch", type=int, default=64,
                   help="candidates fetched per query before de-duplication (--distinct_text)")
    p.add_argument("--frame_tau", type=float, default=None,
                   help="temperature for the frame score; default None = SignRAG Eq. 5-8 (none)")
    p.add_argument("--same_video", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--no_filter", action="store_true", help="disable all filters (smoke test S4)")
    p.add_argument("--no_baselines", dest="baselines", action="store_false")
    p.add_argument("--random_seeds", type=int, default=5)
    p.add_argument("--query_batch", type=int, default=8)
    p.add_argument("--gallery_chunk", type=int, default=128)
    p.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("evaluate", help="metrics for every retrieval_*.json in a run")
    p.add_argument("--run_dir", required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    {"extract": cmd_extract, "retrieve": cmd_retrieve, "evaluate": cmd_evaluate}[args.cmd](args)
