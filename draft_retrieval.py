"""Draft-guided retrieval inputs for RA-SFT ("translate, retrieve, refine").

A clip's *draft translation* is the TF-IDF query over the training sentences;
the top-K distinct sentences become its hint (rag_mode real). Test-time drafts
come from a translator that never saw the clip (no ground truth anywhere).
Train-time drafts must carry test-like errors, otherwise the hints are near
perfect in training and the model over-relies on them (cf. CSL_oracle_d with
weak test hints: 27.63 -> 24.60). Draft sources:

    preds      prediction files (sample: <name>, prediction: ...) for the split,
               e.g. cross-fitted fold translators (train) or the baseline (dev/test)
    synthetic  the clip's reference corrupted by token deletion / substitution, with
               --noise chosen so the draft BLEU matches the real test drafts
               (train split only; uses training labels only)

Training queries exclude themselves and every clip with the identical reference.

    python draft_retrieval.py --dataset CSL_Daily --split train --source synthetic --noise 0.45 --out_dir D
    python draft_retrieval.py --dataset CSL_Daily --split test --source preds --pred_file BASE/test_tmp_pres.txt --out_dir D
"""
import argparse
import contextlib
import io
import os
import random
import re
from collections import Counter

import numpy as np
import torch

import utils
from config import dev_label_paths, test_label_paths, train_label_paths
from signrag_common import bleu_text, is_cjk, norm_text, write_json
from signrag_retrieval import retrieval_metrics, tfidf_score_fn

LABELS = {"train": train_label_paths, "dev": dev_label_paths, "test": test_label_paths}


def read_preds(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"sample: (.*?), prediction: ?(.*)$", line.rstrip("\n"))
        out[m.group(1)] = m.group(2)
    return out


def units(text):
    """Characters (Chinese, punctuation removed) or words."""
    t = norm_text(text)
    return list(t.replace(" ", "")) if is_cjk(t) else t.split()


def join(us, cjk):
    return "".join(us) if cjk else " ".join(us)


def corrupt(text, noise, vocab, weights, rng):
    """Each unit is kept with prob 1 - noise, else deleted (1/2) or replaced by a
    frequency-sampled training unit (1/2)."""
    us, out = units(text), []
    for u in us:
        r = rng.random()
        if r >= noise:
            out.append(u)
        elif r < noise / 2:
            out.append(vocab[np.searchsorted(weights, rng.random())])
    return join(out, is_cjk(text))


def draft_bleu(refs, drafts):
    from SLRT_metrics import translation_performance
    with contextlib.redirect_stdout(io.StringIO()):
        b, _ = translation_performance([bleu_text(r) for r in refs], [bleu_text(d) if d else "<empty>" for d in drafts])
    return b["bleu4"]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    items = list(utils.load_dataset_file(LABELS[args.split][args.dataset]).values())
    train_items = list(utils.load_dataset_file(train_label_paths[args.dataset]).values())
    gallery = [it["text"] for it in train_items]                     # one entry per training clip
    g_names = [it["name"] for it in train_items]
    g_keys = np.array([norm_text(t) for t in gallery])

    if args.source == "preds":
        preds = read_preds(args.pred_file)
        items = [it for it in items if it["name"] in preds]
        drafts = [preds[it["name"]] for it in items]
        if args.dataset == "CSL_Daily":
            drafts = [d.replace(" ", "") for d in drafts]
    else:
        if args.split != "train":
            raise SystemExit("synthetic drafts are for the training split only")
        cnt = Counter(u for t in gallery for u in units(t))
        vocab = list(cnt)
        weights = np.cumsum([cnt[u] for u in vocab], dtype=float)
        weights /= weights[-1]
        rng = random.Random(args.seed)
        drafts = [corrupt(it["text"], args.noise, vocab, weights, rng) for it in items]
    refs = [it["text"] for it in items]
    print(f"{args.dataset} {args.split}: {len(items)} clips, draft BLEU-4 {draft_bleu(refs, drafts):.2f}")

    fn = tfidf_score_fn(gallery, [d if d else "." for d in drafts], "cpu")
    q_keys = [norm_text(r) for r in refs]
    g_index = {n: j for j, n in enumerate(g_names)}
    results = []
    for s in range(0, len(items), 256):
        rows = torch.arange(s, min(s + 256, len(items)))
        sc = fn(rows).numpy()
        for r, row in zip(rows.tolist(), sc):
            if args.split == "train":                                    # self and identical reference
                row[g_keys == q_keys[r]] = -np.inf
                j = g_index.get(items[r]["name"])
                if j is not None:
                    row[j] = -np.inf
            top, seen = [], set()
            for j in np.argsort(-row)[:args.fetch]:
                if not np.isfinite(row[j]) or g_keys[j] in seen:
                    continue
                seen.add(g_keys[j])
                top.append({"name": g_names[j], "text": gallery[j], "score": round(float(row[j]), 4), "video_id": "-"})
                if len(top) == args.k:
                    break
            results.append({"query": {"name": items[r]["name"], "text": refs[r], "video_id": ""},
                            "draft": drafts[r], "topk": top})
    with contextlib.redirect_stdout(io.StringIO()):
        m = retrieval_metrics(results)
    m.update({"draft_bleu4": draft_bleu(refs, drafts), "source": args.source, "noise": args.noise,
              "pred_file": args.pred_file})
    print(f"  retrieval BLEU-4 (top-1) {m['bleu4']:.2f}  F1@1 {m['f1@1']:.2f}  F1@3 {m['f1@3']:.2f}")
    tag = args.tag or args.source
    write_json(os.path.join(args.out_dir, f"retrieval_{args.split}_{tag}.json"),
               {"split": args.split, "method": f"draft_{tag}", "results": results}, ensure_ascii=False)
    write_json(os.path.join(args.out_dir, f"metrics_{args.split}_{tag}.json"), m, indent=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["CSL_Daily", "How2Sign"])
    p.add_argument("--split", required=True, choices=["train", "dev", "test"])
    p.add_argument("--source", required=True, choices=["preds", "synthetic"])
    p.add_argument("--pred_file", default="")
    p.add_argument("--noise", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--fetch", type=int, default=64)
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
