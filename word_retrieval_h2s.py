"""Word-level retrieval for How2Sign with an ASL isolated-sign lexicon (WLASL).

spot   Slide windows over a clip's pose sequence and run the released WLASL
       ISLR checkpoint (Uni-Sign, 2,000 ASL glosses) with beam search. Every
       window's top-B glosses and sequence scores are stored, so the same dump
       serves the test-time hints, calibration and train-split mining.
       The clip is normalised once (as a WLASL clip is), then cut into windows.

hints  Unconstrained word retrieval: a gloss is a candidate of a clip when it is
       the top-1 of some window; its evidence is the best such window score.
       Glosses with evidence >= theta are kept (at most --top_n, strongest
       first) and written in temporal order as "Words: w1 w2 ...".
       theta is calibrated on training clips only (max F1 against their
       subtitle content words). No translation of a test clip is used to
       select or rank words; the reference is used only for evaluation.
       Also written:
         - oracle hints (GT content words that have a WLASL gloss, sentence
           order): an **oracle upper bound**, labelled as such;
         - a train-split pseudo-dictionary (subtitle-constrained: training
           data only), word -> (clip, start, length, score).

Evaluation limit: How2Sign has no word-level timing, so retrieval is scored
at clip level against the reference's content words (lemma match), not
per segment.

    python word_retrieval_h2s.py spot --split test --shard 0 --num_shards 2 --out_dir SPOT
    python word_retrieval_h2s.py hints --spot_dir SPOT --out_dir HINTS
"""
import argparse
import glob
import json
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.append(os.environ.get("UNISIGN_PYLIB", "/data/hwu/Uni_sign_repo/pylib"))   # lemminflect
from lemminflect import getAllLemmas
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import utils
from config import pose_dirs, test_label_paths, train_label_paths
from datasets import load_part_kp
from signrag_common import norm_text, write_json

WLASL_LABELS = "data/WLASL/labels-2000.{}"
WLASL_CKPT = "/data/hwu/Uni_sign_repo/data/WLASL/wlasl_pose_only_islr.pth"
LABELS = {"train": train_label_paths["How2Sign"], "test": test_label_paths["How2Sign"]}
CHUNK = 1000


def wlasl_glosses():
    gl = set()
    for s in ("train", "dev", "test"):
        gl.update(v["text"] for v in utils.load_dataset_file(WLASL_LABELS.format(s)).values())
    return sorted(gl)


def lemmas(w):
    out = {w}
    for ls in getAllLemmas(w).values():
        out.update(ls)
    return out


def content_words(text):
    return [w for w in norm_text(text).split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]


class GlossMatcher:
    """Lemma-level match between WLASL glosses and subtitle words."""

    def __init__(self, glosses):
        self.key = {g: " ".join(norm_text(g).split()) for g in glosses}
        self.lemma2gloss = defaultdict(set)
        self.phrases = {}
        for g, k in self.key.items():
            if " " in k:
                self.phrases[g] = k
            elif k:
                for l in lemmas(k):
                    self.lemma2gloss[l].add(g)

    def glosses_of_word(self, w):
        return {g for l in lemmas(w) for g in self.lemma2gloss.get(l, ())}

    def gold(self, text):
        """(content words, set of glosses matching the text, ordered unique glosses)."""
        words = content_words(text)
        norm = " " + norm_text(text) + " "
        order = []
        for w in words:
            for g in sorted(self.glosses_of_word(w)):
                if g not in order:
                    order.append(g)
        order += [g for g, k in self.phrases.items() if f" {k} " in norm and g not in order]
        return words, set(order), order


# --------------------------------------------------------------------------- #
#  spot
# --------------------------------------------------------------------------- #
def cmd_spot(args):
    from signrag_retrieval import build_unisign

    os.makedirs(args.out_dir, exist_ok=True)
    glosses = wlasl_glosses()
    gid = {g: i for i, g in enumerate(glosses)}
    labels = utils.load_dataset_file(LABELS[args.split])
    keys = list(labels)
    if args.subset:
        keys = sorted(random.Random(args.seed).sample(keys, args.subset))
    keys = keys[args.shard::args.num_shards]
    model, _ = build_unisign("WLASL", args.ckpt)
    model = model.cuda().eval()
    tok = model.mt5_tokenizer
    prefix = tok([f"Translate sign language video to {model.lang}: "], return_tensors="pt").to("cuda")
    prefix_emb = model.mt5_model.encoder.embed_tokens(prefix["input_ids"])
    pose_dir = pose_dirs["How2Sign"].format(args.split)

    @torch.no_grad()
    def predict(kps, spans):
        ids, scs = [], []
        for i in range(0, len(spans), args.bs):
            chunk = spans[i:i + args.bs]
            L = max(l for _, l in chunk)
            src = {}
            for part, x in kps.items():
                src[part] = torch.stack([torch.cat([x[s:s + l], x[s + l - 1:s + l].expand(L - l, -1, -1)])
                                         for s, l in chunk]).float().cuda()
            mask = torch.zeros(len(chunk), L, dtype=torch.long, device="cuda")
            for j, (_, l) in enumerate(chunk):
                mask[j, :l] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.encode_pose(src)
                emb = torch.cat([prefix_emb.expand(len(chunk), -1, -1), z.to(prefix_emb.dtype)], 1)
                am = torch.cat([prefix["attention_mask"].expand(len(chunk), -1), mask], 1)
                g = model.mt5_model.generate(inputs_embeds=emb, attention_mask=am, max_new_tokens=10,
                                             num_beams=args.beams, num_return_sequences=args.beams,
                                             output_scores=True, return_dict_in_generate=True)
            txt = [t.strip() for t in tok.batch_decode(g.sequences, skip_special_tokens=True)]
            ids.append(np.array([gid.get(t, -1) for t in txt], dtype=np.int16).reshape(len(chunk), args.beams))
            scs.append(g.sequences_scores.float().cpu().numpy().astype(np.float16).reshape(len(chunk), args.beams))
        return np.concatenate(ids), np.concatenate(scs)

    for c0 in range(0, len(keys), CHUNK):
        path = os.path.join(args.out_dir, f"{args.split}_s{args.shard}of{args.num_shards}_c{c0 // CHUNK:03d}.npz")
        if os.path.isfile(path):
            continue
        names, n_frames, ptr, starts, lens, top_ids, top_sc = [], [], [0], [], [], [], []
        for key in keys[c0:c0 + CHUNK]:
            item = labels[key]
            pose = pickle.load(open(os.path.join(pose_dir, item["video_path"].replace(".mp4", ".pkl")), "rb"))
            n = len(pose["scores"])
            kps = load_part_kp([pose["keypoints"][i] for i in range(n)],
                               [pose["scores"][i] for i in range(n)], force_ok=True)
            spans = [(s, l) for l in args.lens if l <= n for s in range(0, n - l + 1, args.stride)] or [(0, n)]
            ids, scs = predict(kps, spans)
            names.append(item["name"])
            n_frames.append(n)
            starts += [s for s, _ in spans]
            lens += [l for _, l in spans]
            top_ids.append(ids)
            top_sc.append(scs)
            ptr.append(ptr[-1] + len(spans))
        np.savez(path, names=np.array(names), n_frames=np.array(n_frames), ptr=np.array(ptr),
                 starts=np.array(starts, dtype=np.int32), lens=np.array(lens, dtype=np.int16),
                 top_ids=np.concatenate(top_ids), top_scores=np.concatenate(top_sc))
        print(f"[{args.split} shard {args.shard}] {min(c0 + CHUNK, len(keys))}/{len(keys)} clips -> {path}", flush=True)


# --------------------------------------------------------------------------- #
#  hints
# --------------------------------------------------------------------------- #
def load_spot(spot_dir, split, use_lens=None, keep_stride=0):
    clips = {}
    for path in sorted(glob.glob(os.path.join(spot_dir, f"{split}_s*_c*.npz"))):
        z = np.load(path)
        for i, name in enumerate(z["names"]):
            a, b = z["ptr"][i], z["ptr"][i + 1]
            keep = np.isin(z["lens"][a:b], use_lens) if use_lens else np.ones(b - a, bool)
            if keep_stride:
                keep &= z["starts"][a:b] % keep_stride == 0
            clips[str(name)] = (z["starts"][a:b][keep], z["lens"][a:b][keep], z["top_ids"][a:b][keep],
                                z["top_scores"][a:b][keep].astype(np.float32))
    return clips


def candidates(clip):
    """gloss id -> (best top-1 window score, start of that window, #top-1 windows)."""
    starts, _, ids, sc = clip
    best = {}
    for w in range(len(starts)):
        g = int(ids[w, 0])
        if g < 0:
            continue
        s = float(sc[w, 0])
        if g not in best or s > best[g][0]:
            best[g] = (s, int(starts[w]), best.get(g, (0, 0, 0))[2] + 1)
        else:
            best[g] = (best[g][0], best[g][1], best[g][2] + 1)
    return best


def select(cands, theta, top_n):
    kept = sorted([(s, st, g) for g, (s, st, _) in cands.items() if s >= theta], reverse=True)[:top_n]
    return [g for _, _, g in sorted(kept, key=lambda x: x[1])]           # temporal order


def prf(pred, gold):
    inter = len(set(pred) & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if inter else 0.0)


def cmd_hints(args):
    os.makedirs(args.out_dir, exist_ok=True)
    glosses = wlasl_glosses()
    M = GlossMatcher(glosses)
    labels = {s: {v["name"]: v for v in utils.load_dataset_file(LABELS[s]).values()} for s in ("train", "test")}
    spot = {s: load_spot(args.spot_dir, s, args.use_lens, args.keep_stride) for s in ("train", "test")}
    print({s: len(v) for s, v in spot.items()}, "clips with window predictions")
    missing = {s: len(labels[s]) - len(spot[s]) for s in spot}
    if any(missing.values()) and not args.allow_partial:
        # a partial dump must not produce hint files that look complete
        raise SystemExit(f"incomplete spot dump {args.spot_dir}: missing clips {missing} (--allow_partial for subsets)")
    cands = {s: {n: candidates(c) for n, c in spot[s].items()} for s in spot}
    gold = {s: {n: M.gold(labels[s][n]["text"]) for n in spot[s]} for s in spot}

    # theta on training clips only (a fixed random subset when the full split is available)
    calib = sorted(cands["train"])
    if len(calib) > args.n_calib:
        calib = sorted(random.Random(0).sample(calib, args.n_calib))
    all_s = np.array([s for n in calib for (s, _, _) in cands["train"][n].values()])
    grid = np.unique(np.quantile(all_s, np.linspace(0, 0.99, 100))) if len(all_s) else np.array([0.0])
    best = (-1, None)
    curve = []
    for th in grid:
        P, R, F = zip(*[prf([glosses[g] for g in select(cands["train"][n], th, args.top_n)], gold["train"][n][1])
                        for n in calib])
        curve.append([float(th), 100 * np.mean(P), 100 * np.mean(R), 100 * np.mean(F)])
        if np.mean(F) > best[0]:
            best = (np.mean(F), float(th))
    theta = best[1]
    print(f"theta = {theta:.3f} (calibration F1 {100 * best[0]:.1f} on {len(calib)} train clips)")

    metrics = {"lexicon": "WLASL-2000 (ASL)", "checkpoint": WLASL_CKPT, "theta": theta, "top_n": args.top_n,
               "use_lens": args.use_lens, "keep_stride": args.keep_stride,
               "n_calib": len(calib), "calibration_curve": curve,
               "evaluation_limit": "no word-level timing in How2Sign: clip-level match of words to the "
                                   "reference's content words (lemmas), not per segment"}
    freq = Counter(g for n in labels["train"] for g in M.gold(labels["train"][n]["text"])[1])
    for split in ("train", "test"):
        names = sorted(spot[split])
        res, ores = [], []
        P, R, F, Rw, n_words, Pk = [], [], [], [], [], defaultdict(list)
        for n in names:
            sel = [glosses[g] for g in select(cands[split][n], theta, args.top_n)]
            words, gset, gorder = gold[split][n]
            res.append({"query": {"name": n}, "hint": "Words: " + " ".join(sel) if sel else "", "words": sel})
            ores.append({"query": {"name": n}, "hint": "Words: " + " ".join(gorder) if gorder else "",
                         "words": gorder})
            if words:
                p, r, f = prf(sel, gset)
                P.append(p)
                R.append(r)
                F.append(f)
                # recall over all reference content words (WLASL coverage included)
                hit = {w for w in words if M.glosses_of_word(w) & set(sel)}
                Rw.append(len(hit) / len(set(words)))
                ranked = [glosses[g] for g, _ in sorted(cands[split][n].items(), key=lambda x: -x[1][0])]
                for k in (1, 3, 5, 10):
                    Pk[f"P@{k}"].append(len(set(ranked[:k]) & gset) / k)
                    Pk[f"hit@{k}"].append(bool(set(ranked[:k]) & gset))
            n_words.append(len(sel))
        write_json(os.path.join(args.out_dir, f"hints_{split}.json"),
                   {"split": split, "oracle": False, "results": res})
        write_json(os.path.join(args.out_dir, f"oracle_gtwords_{split}.json"),
                   {"split": split, "oracle": True, "note": "GT content words with a WLASL gloss (upper bound)",
                    "results": ores})
        prior_n = max(1, round(float(np.mean(n_words))))
        prior = [g for g, _ in freq.most_common(prior_n)]
        prior_p = [prf(prior, gold[split][n][1])[0] for n in names if gold[split][n][0]]
        metrics[split] = {"n": len(names), "avg_words": float(np.mean(n_words)),
                          "precision_vs_wlasl_gold": 100 * float(np.mean(P)),
                          "recall_vs_wlasl_gold": 100 * float(np.mean(R)), "f1": 100 * float(np.mean(F)),
                          "recall_vs_all_content_words": 100 * float(np.mean(Rw)),
                          "wlasl_coverage_of_content_words": 100 * float(np.mean(
                              [len({w for w in g[0] if M.glosses_of_word(w)}) / len(set(g[0]))
                               for g in (gold[split][n] for n in names) if g[0]])),
                          "frequency_prior_precision": 100 * float(np.mean(prior_p)),
                          **{k: 100 * float(np.mean(v)) for k, v in Pk.items()}}
        print(split, {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics[split].items()})

    # representative test examples
    ex = []
    for n in random.Random(1).sample(sorted(spot["test"]), 10):
        c = cands["test"][n]
        top = sorted(c.items(), key=lambda x: -x[1][0])[:8]
        ex.append({"name": n, "text": labels["test"][n]["text"],
                   "selected": [glosses[g] for g in select(c, theta, args.top_n)],
                   "top_candidates": [[glosses[g], round(s, 3), st] for g, (s, st, _) in top]})
    write_json(os.path.join(args.out_dir, "examples_test.json"), ex, indent=1)

    # train pseudo-dictionary (subtitle-constrained, training data only)
    dic = defaultdict(list)
    gid = {g: i for i, g in enumerate(glosses)}
    for n, (starts, lens, ids, sc) in spot["train"].items():
        for g in gold["train"][n][1]:
            gi = gid[g]
            hit = np.where(ids[:, 0] == gi)[0]
            if len(hit):
                w = hit[np.argmax(sc[hit, 0])]
                dic[g].append([n, int(starts[w]), int(lens[w]), round(float(sc[w, 0]), 3)])
    write_json(os.path.join(args.out_dir, "dictionary_train.json"), dic)
    metrics["dictionary_train"] = {"glosses_with_entries": len(dic),
                                   "entries": int(sum(len(v) for v in dic.values())),
                                   "median_entries_per_gloss": float(np.median([len(v) for v in dic.values()])) if dic else 0}
    write_json(os.path.join(args.out_dir, "metrics.json"), metrics, indent=1)
    print("dictionary:", metrics["dictionary_train"])
    write_json(os.path.join(args.out_dir, "DONE"), {"clips": {s: len(v) for s, v in spot.items()}})


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("spot")
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--subset", type=int, default=0, help="random subset of N clips (validation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lens", type=int, nargs="+", default=[12, 16])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--ckpt", default=WLASL_CKPT)
    p.add_argument("--out_dir", required=True)
    p = sub.add_parser("hints")
    p.add_argument("--spot_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_n", type=int, default=8)
    p.add_argument("--n_calib", type=int, default=3000)
    p.add_argument("--use_lens", type=int, nargs="+", default=None, help="only windows of these lengths")
    p.add_argument("--keep_stride", type=int, default=0, help="only windows whose start is a multiple of N")
    p.add_argument("--allow_partial", action="store_true", help="accept a dump of a clip subset (validation)")
    a = parser.parse_args()
    {"spot": cmd_spot, "hints": cmd_hints}[a.cmd](a)
