"""Stage 2 of doc/signrag_style_retriever_plan.md §11.3: a pose -> text retriever
trained for graded relevance instead of instance pairs.

S2dn (signrag_contrastive.py --objective fine) treats only the clip's own
sentence as a positive. RAG retrieves from the training sentences, which
contain the clip's own sentence for only 0.2% of CSL-Daily dev/test clips, so
the useful hit is the closest *other* sentence. The target of training clip i
is a distribution over candidate training sentences u:

    q_i(u) = softmax_u( TF-IDF(y_i, u) / T ),   u != y_i  (own sentence masked)

The sign->text scores are fitted to it with a soft cross-entropy (KL(q || p)
up to a constant). S2dn's in-batch exact-pair loss is kept with weight
--exact_weight:

    L = CE(q_i, softmax_u(s2t(i, u) / tau)) + lambda * L_exact

Candidates per batch: the batch's sentences plus the top --nbr_k TF-IDF
neighbours of each. The targets need the high-similarity region; the rest
are near zero. The reference is only used to build targets on the train split.

Model: the S2dn pieces on cached frame features with a frozen encoder: phi on
the X frames, the frozen official mT5 token cache with a linear text head, and
SignRAG soft alignment (fine objective). --init_phi starts from an S2dn phi.pth
(G1); without it phi starts from pose_proj (G2).

Selection: the real dev split retrieves from the unique training sentences, as
in RAG. The top-1 is scored as a translation (retrieval BLEU-4, char level for
CSL), plus F1@3 of the top-3. Epoch 0 is the initial model. The best
checkpoint is saved as phi.pth in the S2dn format, so
`signrag_retrieval.py retrieve --phi RUN/phi.pth --score s2t --distinct_text`
builds the hints. Test is scored once, with the best checkpoint.

    python graded_retriever.py --feat_dir $OUT/feats/E_csl --run_dir $OUT/step2/G1_E_csl \\
        --init_phi $OUT/step2/S2dn_E_csl/phi.pth
"""
import argparse
import contextlib
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import mt5_path
from signrag_common import (DEFAULT_OUT, FeatSplit, gather_padded, get_mt5_cache, identity_linear,
                            load_phi_checkpoint, multi_positive_nce, pose_proj_phi, soft_match_pair_scores,
                            soft_match_scores, write_json)
from signrag_retrieval import retrieval_metrics, tfidf_score_fn

SEM_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"      # S2dn default, kept for --pair_recall


@torch.no_grad()
def tfidf_matrix(keys, device, chunk=1024):
    """Dense [n, n] TF-IDF cosine between the gallery sentences (float16)."""
    fn = tfidf_score_fn(keys, keys, "cpu")
    R = torch.empty(len(keys), len(keys), dtype=torch.float16, device=device)
    for s in range(0, len(keys), chunk):
        rows = torch.arange(s, min(s + chunk, len(keys)))
        R[s:s + len(rows)] = fn(rows).to(device, torch.float16)
    return R


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.run_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(args.feat_dir, "dev", "meta.json")):
        raise SystemExit(f"{args.feat_dir} has no dev split; selection needs one (CSL-Daily first, plan §11.3)")
    train = FeatSplit(os.path.join(args.feat_dir, "train"))
    dev = FeatSplit(os.path.join(args.feat_dir, "dev"))
    test = FeatSplit(os.path.join(args.feat_dir, "test"))
    dataset = train.meta["dataset"]
    texts = get_mt5_cache(args.cache_root, dataset, args.mt5_path, args.text_layer)

    # gallery = unique training sentences; graded relevance between them by TF-IDF
    tids = texts.ids(train.texts)
    g_tids, clip_cols = np.unique(tids, return_inverse=True)
    raw_of = {}
    for t, raw in zip(tids, train.texts):
        raw_of.setdefault(t, raw)
    g_raw = [raw_of[t] for t in g_tids]
    n_g = len(g_tids)
    R = tfidf_matrix([texts.keys[t] for t in g_tids], device)
    R.fill_diagonal_(-1)                                          # own sentence: never a target
    nbr = R.topk(args.nbr_k, dim=1).indices                       # [n_g, K] nearest other sentences
    print(f"{dataset}: {len(train)} training clips, {n_g} unique sentences; dev {len(dev)}, test {len(test)}; "
          f"candidates per batch: batch sentences + top-{args.nbr_k} TF-IDF neighbours each")

    # ---- model pieces (S2dn, fine objective, frozen encoder) ------------------
    if args.init_phi:
        phi, head, ck = load_phi_checkpoint(args.init_phi)
        if ck["objective"] != "fine" or ck.get("encoder"):
            raise ValueError(f"{args.init_phi}: needs a fine-objective phi trained on frozen features")
        if os.path.abspath(ck["text_cache"]) != texts.dir:
            raise ValueError(f"{args.init_phi} used the text cache {ck['text_cache']}, not {texts.dir}")
        tau0, tau_attn0 = ck["tau"], ck["tau_attn"]
    else:
        phi, head = pose_proj_phi(args.feat_dir), identity_linear(texts.pooled.shape[1])
        tau0 = tau_attn0 = args.tau_init
    phi, head = phi.to(device), head.to(device)
    log_tau = nn.Parameter(torch.tensor(math.log(tau0), device=device))
    log_tau_attn = nn.Parameter(torch.tensor(math.log(tau_attn0), device=device))
    tau = lambda: log_tau.exp().clamp(min=0.01)
    tau_attn = lambda: log_tau_attn.exp().clamp(min=0.01)
    opt = torch.optim.AdamW([
        {"params": phi.parameters(), "lr": args.lr_phi, "weight_decay": args.weight_decay},
        {"params": head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": [log_tau, log_tau_attn], "lr": args.lr_head, "weight_decay": 0.0},
    ])

    x_dev = "cuda" if train.X.nbytes < 0.4 * torch.cuda.mem_get_info()[0] else "cpu"
    X = torch.from_numpy(np.array(train.X)).to(x_dev)
    off = torch.as_tensor(train.offsets, device=x_dev)
    t_tok = torch.from_numpy(np.array(texts.tokens)).to(device)
    t_off = torch.as_tensor(texts.tok_offsets, device=device)
    g_tids_t = torch.as_tensor(g_tids, device=device)
    cols = torch.as_tensor(clip_cols, device=device)
    print(f"cached X ({train.X.nbytes / 2**30:.1f} GiB) on {x_dev}; init: {args.init_phi or 'pose_proj'}")

    steps_per_epoch = len(train) // args.batch_size
    total = args.max_steps or args.epochs * steps_per_epoch
    warmup = min(100, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    @torch.no_grad()
    def rank(feats, k):
        """Top-k gallery sentences for every clip of a split (RAG retrieval, no filter)."""
        phi.eval(), head.eval()
        gal = []
        for s in range(0, n_g, 256):
            t, tm = texts.padded_tokens(g_tids[s:s + 256])
            gal.append((F.normalize(head(t.to(device).float()), dim=-1).half(), tm.to(device)))
        top = []
        for s in range(0, len(feats), 32):
            x, m = feats.padded(np.arange(s, min(s + 32, len(feats))))
            q = F.normalize(phi(x.to(device).float()), dim=-1).half()
            sc = torch.cat([soft_match_scores(q, m.to(device), g, gm, tau_attn()) for g, gm in gal], 1)
            top.append(sc.topk(k, dim=-1).indices.cpu())
        phi.train(), head.train()
        return torch.cat(top).numpy()

    def evaluate(feats):
        top = rank(feats, 3)
        results = [{"query": {"name": feats.names[i], "text": feats.texts[i], "video_id": feats.videos[i]},
                    "topk": [{"text": g_raw[j], "video_id": "-"} for j in top[i]]} for i in range(len(feats))]
        with contextlib.redirect_stdout(io.StringIO()):
            m = retrieval_metrics(results)
        return {k: m[k] for k in ("bleu4", "rouge_l", "f1@1", "f1@3", "em@1")}, results

    def save(tag, epoch, d):
        torch.save({"phi": phi.state_dict(), "phi_cfg": phi.cfg, "text_head": head.state_dict(),
                    "text_cache": texts.dir, "objective": "fine", "tau": float(tau()),
                    "tau_attn": float(tau_attn()), "positives": "graded", "pos_threshold": 0.85,
                    "epoch": epoch, "dev": d, "args": {**vars(args), "sem_model": SEM_MODEL}, "encoder": None},
                   os.path.join(args.run_dir, f"phi{tag}.pth"))

    write_json(os.path.join(args.run_dir, "args.json"), vars(args), indent=2)
    log = open(os.path.join(args.run_dir, "log.txt"), "a")
    d0, _ = evaluate(dev)
    save("_init", 0, d0)
    print(f"epoch 0 (init): dev {d0}", flush=True)
    log.write(json.dumps({"epoch": 0, **{f"dev_{k}": v for k, v in d0.items()}}) + "\n")
    best, bad, step, epoch = d0[args.select], 0, 0, 0
    save("", 0, d0)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        t0, parts = time.time(), []
        perm = torch.randperm(len(train), generator=gen, device=device)
        for s in range(0, steps_per_epoch * args.batch_size, args.batch_size):
            b = perm[s:s + args.batch_size]
            x, m = gather_padded(X, off, b.to(x_dev))
            x, m = x.to(device, non_blocking=True).float(), m.to(device)
            y = cols[b]                                           # gallery column of each clip's sentence
            U = torch.unique(torch.cat([y, nbr[y].flatten()]))   # sorted candidate sentences
            t, tm = gather_padded(t_tok, t_off, g_tids_t[U])
            fn = F.normalize(phi(x), dim=-1)
            tn = F.normalize(head(t.float()), dim=-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s2t, t2s = soft_match_pair_scores(fn, m, tn, tm, tau_attn(), args.sim_chunk)  # [B, |U|], [|U|, B]
            own = U[None] == y[:, None]
            # graded term over the candidates, own sentence excluded on both sides
            r = R[y][:, U].float().masked_fill(own, float("-inf"))
            q = (r / args.target_T).softmax(-1)
            valid = r.max(-1).values > 0                          # some candidate shares a token
            logp = (s2t / tau()).masked_fill(own, -1e4).log_softmax(-1)
            l_graded = -(q * logp).sum(-1)[valid].mean() if valid.any() else s2t.sum() * 0
            # S2dn's exact-pair loss, in batch and bidirectional
            bcol = torch.searchsorted(U, y)
            pos = y[:, None] == y[None, :]
            l_exact = 0.5 * (multi_positive_nce(s2t[:, bcol] / tau(), pos, args.pos_loss)
                             + multi_positive_nce(t2s[bcol] / tau(), pos, args.pos_loss))
            loss = l_graded + args.exact_weight * l_exact
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            parts.append([loss.item(), l_graded.item(), l_exact.item(), len(U)])
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        d, _ = evaluate(dev)
        p = np.mean(parts, 0)
        rec = {"epoch": epoch, "train_loss": p[0], "train_graded": p[1], "train_exact": p[2],
               "candidates": p[3], "tau": float(tau()), "tau_attn": float(tau_attn()),
               **{f"dev_{k}": v for k, v in d.items()}, "time_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch}: loss {p[0]:.4f} (graded {p[1]:.4f}, exact {p[2]:.4f}, |U| {p[3]:.0f}) "
              f"dev {d} ({rec['time_s']}s)", flush=True)
        if d[args.select] > best:
            best, bad = d[args.select], 0
            save("", epoch, d)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no dev {args.select} gain for {args.patience} epochs")
                break
        if args.max_steps and step >= args.max_steps:
            break

    # test once, with the selected checkpoint
    best_ck = torch.load(os.path.join(args.run_dir, "phi.pth"), map_location="cpu")
    phi.load_state_dict(best_ck["phi"])
    head.load_state_dict(best_ck["text_head"])
    with torch.no_grad():
        log_tau_attn.fill_(math.log(best_ck["tau_attn"]))
    out = {}
    for name, feats in (("dev", dev), ("test", test)):
        out[name], results = evaluate(feats)
        write_json(os.path.join(args.run_dir, f"retrieval_{name}_graded.json"),
                   {"split": name, "method": "graded", "results": results}, ensure_ascii=False)
    write_json(os.path.join(args.run_dir, "done.json"),
               {"best_epoch": best_ck["epoch"], "select": args.select, **out, "epochs_run": epoch}, indent=1)
    print(f"best epoch {best_ck['epoch']} (dev {args.select} {best:.2f}): dev {out['dev']}  test {out['test']}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feat_dir", required=True, help="output of `signrag_retrieval.py extract` (train/dev/test)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--init_phi", default="", help="S2dn phi.pth to start from (G1); default: pose_proj (G2)")
    p.add_argument("--cache_root", default=os.path.join(DEFAULT_OUT, "text_cache"))
    p.add_argument("--mt5_path", default=mt5_path, help="official mT5-base (frozen text tower)")
    p.add_argument("--text_layer", type=int, default=-1)
    p.add_argument("--nbr_k", type=int, default=8, help="TF-IDF neighbours per batch sentence added as candidates")
    p.add_argument("--target_T", type=float, default=0.1, help="temperature of the graded targets")
    p.add_argument("--exact_weight", type=float, default=0.25, help="lambda of S2dn's exact-pair loss")
    p.add_argument("--pos_loss", default="any", choices=["any", "all"])
    p.add_argument("--sim_chunk", type=int, default=32, help="texts per checkpointed chunk of [B, U, M, L]")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=0, help="stop after N steps (smoke tests)")
    p.add_argument("--lr_phi", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--tau_init", type=float, default=0.07, help="tau and tau_attn without --init_phi")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--select", default="bleu4", choices=["bleu4", "f1@1", "f1@3"],
                   help="dev retrieval metric (top-1 BLEU-4 or token F1) for model selection")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
