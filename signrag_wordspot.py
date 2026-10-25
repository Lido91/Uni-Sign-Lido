"""Zero-shot word spotting with a trained fine-objective retriever (plan 1).

Every content word of the training vocabulary is scored against a clip in
the retriever's frame<->token space, with no extra training:

    word instance  = mean of the word's sub-word token states (mT5 text tower
                     + trained text head, L2) inside one training sentence
    clip-word score:
        maxsim : max over frames of cos(frame, instance), max over the word's
                 (at most --max_inst) instances
        proto  : max over frames of cos(frame, mean instance)
    *_z        : the same, z-scored per word over --n_ref reference training
                 clips (removes words that score high for every clip)

The top-N words of a clip are compared with the content words of its
reference translation (precision / recall / hit @ N), against a frequency
prior, the words of the clip's top-3 retrieved training sentences, random
words, and the vocabulary ceiling. Results go to --out_dir.

    python signrag_wordspot.py --feat_dir FEATS --phi STEP2/S2dn_E1_h2s/phi.pth \
        --retrieval RUNS/S3dn_E1_h2s/retrieval_test_s2t.json --out_dir OUT
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import T5Tokenizer

from signrag_common import FeatSplit, TextCache, collect_texts, load_phi_checkpoint, norm_text, write_json

KS = (1, 3, 5, 10)


def content_words(text):
    return [w for w in norm_text(text).split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]


def word_instances(texts, raws, tids, tok, head, device):
    """(word, normalised instance embedding) for every content word of the
    sentences `tids`, from their cached token states."""
    words, embs = [], []
    for tid in tids:
        ids = tok(raws[tid], truncation=True, max_length=50)["input_ids"]
        start, end = texts.tok_offsets[tid], texts.tok_offsets[tid + 1]
        if len(ids) != end - start:              # cache and tokenizer disagree: skip the sentence
            continue
        pieces = tok.convert_ids_to_tokens(ids)
        groups, cur = [], []
        for i, p in enumerate(pieces):
            if p == tok.eos_token:
                break
            if p.startswith("▁") and cur:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        states = None
        for g in groups:
            w = norm_text("".join(pieces[i] for i in g).replace("▁", ""))
            if " " in w or w in ENGLISH_STOP_WORDS or len(w) < 2:
                continue
            if states is None:
                with torch.no_grad():
                    states = F.normalize(head(torch.as_tensor(np.asarray(texts.tokens[start:end]),
                                                              device=device).float()), dim=-1)
            words.append(w)
            embs.append(F.normalize(states[g].mean(0), dim=-1))
    return words, torch.stack(embs)


@torch.no_grad()
def clip_scores(feats, ids, phi, inst, inst_word, proto, n_words, device, batch=8):
    """[len(ids), V] maxsim and proto scores."""
    out_m, out_p = [], []
    for s in range(0, len(ids), batch):
        x, m = feats.padded(ids[s:s + batch])
        f = F.normalize(phi(x.to(device).float()), dim=-1).half()
        m = m.to(device)
        best = torch.full((f.size(0), inst.size(0)), float("-inf"), device=device)
        for c in range(0, inst.size(0), 32768):                     # [b, M, n] chunks
            sim = torch.einsum("bmd,nd->bmn", f, inst[c:c + 32768])
            best[:, c:c + 32768] = sim.masked_fill(~m[:, :, None], float("-inf")).amax(1).float()
        wm = torch.full((f.size(0), n_words), float("-inf"), device=device)
        wm.scatter_reduce_(1, inst_word.expand(f.size(0), -1), best, reduce="amax", include_self=True)
        sim = torch.einsum("bmd,vd->bmv", f, proto)
        out_m.append(wm.cpu())
        out_p.append(sim.masked_fill(~m[:, :, None], float("-inf")).amax(1).float().cpu())
    return torch.cat(out_m), torch.cat(out_p)


def evaluate(preds, golds):
    """preds: list of ranked word lists; golds: list of content-word sets."""
    out = {}
    valid = [i for i, g in enumerate(golds) if g]
    for k in KS:
        p = [len(set(preds[i][:k]) & golds[i]) / k for i in valid]
        r = [len(set(preds[i][:k]) & golds[i]) / len(golds[i]) for i in valid]
        h = [bool(set(preds[i][:k]) & golds[i]) for i in valid]
        out[f"P@{k}"], out[f"R@{k}"], out[f"hit@{k}"] = (100 * float(np.mean(v)) for v in (p, r, h))
    return out


def main(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    phi, head, ckpt = load_phi_checkpoint(args.phi)
    phi, head = phi.to(device).eval(), head.to(device).eval()
    texts = TextCache(ckpt["text_cache"])
    train = FeatSplit(os.path.join(args.feat_dir, "train"))
    test = FeatSplit(os.path.join(args.feat_dir, args.split))
    keys, raws = collect_texts(train.meta["dataset"])
    assert keys == texts.keys, "text cache was built from a different text list"
    tok = T5Tokenizer.from_pretrained(texts.meta["model"], legacy=False)

    # vocabulary: content words of the training sentences, clip-level counts
    clip_count = Counter(w for t in train.texts for w in set(content_words(t)))
    vocab = sorted(w for w, c in clip_count.items() if c >= args.min_count)
    w_index = {w: i for i, w in enumerate(vocab)}
    train_tids = np.unique(texts.ids(train.texts))
    words, embs = word_instances(texts, raws, train_tids, tok, head, device)
    keep = [i for i, w in enumerate(words) if w in w_index]
    by_word = {}
    for i in keep:
        by_word.setdefault(w_index[words[i]], []).append(i)
    sel, sel_word = [], []
    for wi, lst in by_word.items():
        if len(lst) > args.max_inst:
            lst = list(rng.choice(lst, args.max_inst, replace=False))
        sel += lst
        sel_word += [wi] * len(lst)
    inst = embs[sel].half()
    inst_word = torch.as_tensor(sel_word, device=device)[None]
    proto = torch.zeros(len(vocab), embs.size(1), device=device)
    proto.index_add_(0, inst_word[0], embs[sel].float())
    proto = F.normalize(proto, dim=-1).half()
    print(f"vocabulary: {len(vocab)} content words (clip count >= {args.min_count}); "
          f"{len(sel)} instances (<= {args.max_inst} per word)")

    # per-word calibration on reference training clips
    ref = np.sort(rng.choice(len(train), min(args.n_ref, len(train)), replace=False))
    rm, rp = clip_scores(train, ref, phi, inst, inst_word, proto, len(vocab), device)
    stats = {k: (v.mean(0), v.std(0).clamp(min=1e-4)) for k, v in (("maxsim", rm), ("proto", rp))}

    ids = np.arange(len(test))
    sm, sp = clip_scores(test, ids, phi, inst, inst_word, proto, len(vocab), device)
    scores = {"maxsim": sm, "proto": sp,
              "maxsim_z": (sm - stats["maxsim"][0]) / stats["maxsim"][1],
              "proto_z": (sp - stats["proto"][0]) / stats["proto"][1]}
    K = max(KS)
    preds = {name: [[vocab[j] for j in row] for row in s.topk(K, dim=1).indices.tolist()]
             for name, s in scores.items()}

    golds = [set(content_words(t)) for t in test.texts]
    freq_rank = [w for w, _ in clip_count.most_common() if w in w_index][:K]
    preds["frequency_prior"] = [freq_rank] * len(ids)
    preds["random_words"] = [list(rng.choice(vocab, K, replace=False)) for _ in ids]
    if args.retrieval:
        ret = {r["query"]["name"]: r for r in json.load(open(args.retrieval))["results"]}
        sent_words = []
        for name in test.names:
            c = Counter()
            for rank, t in enumerate(ret[name]["topk"][:3]):
                for w in set(content_words(t["text"])):
                    c[w] += 10 - rank                 # more retrieved sentences first, then higher rank
            sent_words.append([w for w, _ in c.most_common(K)])
        preds["retrieved_sentences_top3"] = sent_words

    results = {name: evaluate(p, golds) for name, p in preds.items()}
    in_vocab = [len(g & set(vocab)) / len(g) for g in golds if g]
    results["_ceiling_recall_vocab"] = 100 * float(np.mean(in_vocab))
    os.makedirs(args.out_dir, exist_ok=True)
    write_json(os.path.join(args.out_dir, "metrics.json"),
               {"phi": os.path.abspath(args.phi), "split": args.split, "vocab_size": len(vocab),
                "instances": len(sel), "results": results}, indent=2)
    cols = [f"{m}@{k}" for k in KS for m in ("P", "R", "hit")]
    lines = ["\t".join(["method"] + cols)]
    for name, r in results.items():
        if not name.startswith("_"):
            lines.append("\t".join([name] + [f"{r[c]:.2f}" for c in cols]))
    with open(os.path.join(args.out_dir, "metrics.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    best = max((n for n in results if not n.startswith("_")), key=lambda n: results[n]["P@5"])
    with open(os.path.join(args.out_dir, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "gt", f"{best}_top10", "retrieved_sentences_top10"])
        for i, name in enumerate(test.names):
            w.writerow([name, test.texts[i], " ".join(preds[best][i]),
                        " ".join(preds.get("retrieved_sentences_top3", [[]] * len(ids))[i])])
    print("\n".join(lines))
    print(f"vocabulary recall ceiling: {results['_ceiling_recall_vocab']:.1f}%   best by P@5: {best}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--phi", required=True, help="fine-objective phi.pth from signrag_contrastive.py")
    p.add_argument("--retrieval", default="", help="retrieval_<split>_s2t.json for the sentence-retrieval baseline")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--max_inst", type=int, default=32)
    p.add_argument("--n_ref", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
