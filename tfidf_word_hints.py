"""Word-level hints from sentence retrieval results (e.g. the TF-IDF oracle).

For every query, the words of its top-K retrieved *training* clips are pooled
and ranked by

    weight(w) = sum_k score_k * idf(w) * [w in units(neighbour_k)]

and the top --top_n words form the hint. The units are:
    How2Sign : content words of the neighbour's translation (stopwords removed),
               hint "Words: w1 w2 ..."
    CSL-Daily: the neighbour's training gloss annotation (a training label of
               the retrieved clip, never the query's), hint "Glosses: g1 g2 ..."
IDF is computed on the training split. This is the word-level counterpart of
the sentence hints built from the same retrieval file (rag_mode real): same
neighbours, compressed to keywords. When the retrieval file is the TF-IDF
oracle (query = the clip's ground-truth translation), the hints are an
**oracle upper bound**.

    python tfidf_word_hints.py --dataset CSL_Daily --retrieval_dir RAGD --method tfidf_oracle \
        --splits train dev test --out_dir OUT
"""
import argparse
import json
import math
import os
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import utils
from signrag_common import label_path, norm_text, write_json

PREFIX = {"How2Sign": "Words: ", "CSL_Daily": "Glosses: "}


def content_words(text):
    return [w for w in norm_text(text).split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    labels = {s: {v["name"]: v for v in utils.load_dataset_file(label_path(args.dataset, s)).values()}
              for s in ["train"] + [s for s in args.splits if s != "train"]}
    if args.dataset == "CSL_Daily":
        units = lambda name, split: set(labels[split][name]["gloss"])
    else:
        units = lambda name, split: set(content_words(labels[split][name]["text"]))
    df = Counter(u for n in labels["train"] for u in units(n, "train"))
    n_docs = len(labels["train"])
    idf = {u: math.log(n_docs / c) + 1.0 for u, c in df.items()}

    metrics = {"dataset": args.dataset, "retrieval_dir": args.retrieval_dir, "method": args.method,
               "top_n": args.top_n, "k": args.k, "oracle": args.method == "tfidf_oracle"}
    for split in args.splits:
        path = os.path.join(args.retrieval_dir, f"retrieval_{split}_{args.method}.json")
        with open(path) as f:
            results = json.load(f)["results"]
        out, P, R, P_all, R_all, n_words = [], [], [], [], [], []
        for r in results:
            name = r["query"]["name"]
            w = Counter()
            pooled = set()
            for t in r["topk"][:args.k]:
                us = units(t["name"], "train")
                pooled |= us
                for u in us:
                    w[u] += max(t["score"], 0.0) * idf.get(u, 1.0)
            words = [u for u, _ in w.most_common(args.top_n)]
            out.append({"query": {"name": name}, "hint": PREFIX[args.dataset] + " ".join(words) if words else "",
                        "words": words})
            gold = units(name, split)            # evaluation only
            if gold:
                inter = len(set(words) & gold)
                P.append(inter / len(words) if words else 0.0)
                R.append(inter / len(gold))
                P_all.append(len(pooled & gold) / len(pooled) if pooled else 0.0)
                R_all.append(len(pooled & gold) / len(gold))
            n_words.append(len(words))
        write_json(os.path.join(args.out_dir, f"hints_{split}.json"),
                   {"split": split, "source": path, "results": out}, ensure_ascii=False)
        metrics[split] = {"n": len(results), "avg_words": float(np.mean(n_words)),
                          "precision": 100 * float(np.mean(P)), "recall": 100 * float(np.mean(R)),
                          "pooled_precision": 100 * float(np.mean(P_all)),
                          "pooled_recall": 100 * float(np.mean(R_all))}
        print(f"{split}: {metrics[split]}")
    ex = []
    split = args.splits[-1]
    hints = json.load(open(os.path.join(args.out_dir, f"hints_{split}.json")))["results"]
    for i in np.random.RandomState(1).choice(len(hints), 8, replace=False):
        name = hints[i]["query"]["name"]
        ex.append({"name": name, "text": labels[split][name]["text"], "hint": hints[i]["hint"]})
    metrics["examples_" + split] = ex
    write_json(os.path.join(args.out_dir, "metrics.json"), metrics, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["How2Sign", "CSL_Daily"])
    p.add_argument("--retrieval_dir", required=True)
    p.add_argument("--method", default="tfidf_oracle")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--top_n", type=int, default=8)
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
