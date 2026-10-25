"""Translator reranking of retrieval candidates (eval only, test split).

For every test clip, candidate training sentences come from
    (a) TF-IDF over the training sentences, queried with the baseline model's own
        translation of the clip (no ground truth), top --k_tfidf;
    (b) the S3dn sign->text retrieval top-K (--cand_json, pose only);
and are reranked by the SLT model's mean token log-likelihood
log p(candidate | pose). The top-1 is scored as a translation (retrieval BLEU,
signrag_retrieval.retrieval_metrics), next to the top-1 of (a) and (b) alone.

No ground truth is used to build or rank the candidates. With --oracle_json
two analysis-only numbers are added, both marked ORACLE: how often the TF-IDF
oracle's top-1 is in the pool, and the pool's best candidate by TF-IDF
similarity to the reference (the ceiling of any reranker over this pool).

    python rerank_retrieval.py --dataset CSL_Daily --ckpt CKPT --pred_file BASE/test_tmp_pres.txt \
        --cand_json CAND/retrieval_test_s2t.json --oracle_json RUN/retrieval_test_tfidf_oracle.json --out_dir OUT
"""
import argparse
import contextlib
import io
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from config import test_label_paths, train_label_paths
from datasets import S2T_Dataset
from signrag_common import norm_text, write_json
from signrag_retrieval import build_unisign, retrieval_metrics, tfidf_score_fn


def read_preds(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"sample: (.*?), prediction: ?(.*)$", line.rstrip("\n"))
        out[m.group(1)] = m.group(2)
    return out


def as_results(names, test, ranked):
    return [{"query": {"name": n, "text": test[n], "video_id": ""},
             "topk": [{"text": t, "video_id": "-"} for t in ranked[i][:3]]} for i, n in enumerate(names)]


def bleu_of(names, test, ranked):
    with contextlib.redirect_stdout(io.StringIO()):
        return retrieval_metrics(as_results(names, test, ranked))


@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    test = {v["name"]: v["text"] for v in utils.load_dataset_file(test_label_paths[args.dataset]).values()}
    train = sorted({v["text"] for v in utils.load_dataset_file(train_label_paths[args.dataset]).values()})
    preds = read_preds(args.pred_file)
    names = [n for n in test if n in preds]

    # (a) TF-IDF on the baseline translation
    q = [preds[n].replace(" ", "") if args.dataset == "CSL_Daily" else preds[n] for n in names]
    fn = tfidf_score_fn(train, q, "cpu")
    cand_a = []
    for s in range(0, len(q), 256):
        sc = fn(torch.arange(s, min(s + 256, len(q))))
        cand_a += [[train[j] for j in row] for row in sc.topk(args.k_tfidf, dim=1).indices.tolist()]
    # (b) S3dn
    with open(args.cand_json) as f:
        s3 = {r["query"]["name"]: [t["text"] for t in r["topk"]] for r in json.load(f)["results"]}
    cand_b = [s3[n] for n in names]
    pools = []
    for a, b in zip(cand_a, cand_b):
        seen, pool = set(), []
        for t in a + b:
            k = norm_text(t)
            if k and k not in seen:
                seen.add(k)
                pool.append(t)
        pools.append(pool)
    print(f"{args.dataset}: {len(names)} test clips, pool size mean {np.mean([len(p) for p in pools]):.1f}")

    model, uargs = build_unisign(args.dataset, args.ckpt)
    model = model.to(device).eval()
    tok = model.mt5_tokenizer
    ds = S2T_Dataset(path=test_label_paths[args.dataset], args=uargs, phase="test")
    index = {n: i for i, n in enumerate(names)}
    loader = DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn, num_workers=8)
    reranked, scores_out = [None] * len(names), [None] * len(names)
    prefix_text = f"Translate sign language video to {model.lang}: "
    done = 0
    for src, _ in loader:
        for k in ("body", "left", "right", "face_all"):
            src[k] = src[k].to(device).float()                # the dataset yields float64
        src["attention_mask"] = src["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode_pose(src)
            pre = tok([prefix_text] * z.size(0), return_tensors="pt", padding="longest").to(device)
            emb = torch.cat([model.mt5_model.encoder.embed_tokens(pre["input_ids"]), z], 1)
            att = torch.cat([pre["attention_mask"], src["attention_mask"]], 1)
            enc = model.mt5_model.encoder(inputs_embeds=emb, attention_mask=att).last_hidden_state
        for b, name in enumerate(src["name_batch"]):
            if name not in index:
                continue
            i = index[name]
            pool, ll = pools[i], []
            for c in range(0, len(pool), args.bs_cand):
                chunk = pool[c:c + args.bs_cand]
                lab = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=50)["input_ids"].to(device)
                mask = lab != tok.pad_token_id
                lab_in = lab.masked_fill(~mask, -100)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.mt5_model(encoder_outputs=(enc[b:b + 1].expand(len(chunk), -1, -1),),
                                          attention_mask=att[b:b + 1].expand(len(chunk), -1), labels=lab_in)
                lp = F.log_softmax(out.logits.float(), -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
                ll += ((lp * mask).sum(1) / mask.sum(1)).tolist()
            order = np.argsort(ll)[::-1]
            reranked[i] = [pool[j] for j in order]
            scores_out[i] = [round(float(ll[j]), 4) for j in order[:10]]
            done += 1
        if done % 200 < 8:
            print(f"  {done}/{len(names)}", flush=True)

    metrics = {"dataset": args.dataset, "ckpt": args.ckpt, "pred_file": args.pred_file,
               "cand_json": args.cand_json, "k_tfidf": args.k_tfidf,
               "pool_size_mean": float(np.mean([len(p) for p in pools])),
               "tfidf_on_translation_top1": bleu_of(names, test, cand_a),
               "s3dn_top1": bleu_of(names, test, cand_b),
               "translator_rerank": bleu_of(names, test, reranked)}
    if args.oracle_json:
        with open(args.oracle_json) as f:
            orc = {r["query"]["name"]: r["topk"][0]["text"] for r in json.load(f)["results"] if r["topk"]}
        metrics["ORACLE_pool_contains_tfidf_oracle_top1"] = 100 * float(np.mean(
            [norm_text(orc[n]) in {norm_text(t) for t in pools[i]} for i, n in enumerate(names) if n in orc]))
        fo = tfidf_score_fn(train, [test[n] for n in names], "cpu")
        tid = {t: j for j, t in enumerate(train)}
        best = []
        for i, n in enumerate(names):
            s = fo(torch.tensor([i]))[0]
            pidx = [tid.get(t, -1) for t in pools[i]]
            best.append([pools[i][int(np.argmax([s[j].item() if j >= 0 else -1 for j in pidx]))]])
        metrics["ORACLE_best_in_pool"] = bleu_of(names, test, best)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k:42s} BLEU-4 {v['bleu4']:6.2f}  ROUGE-L {v['rouge_l']:6.2f}  F1@1 {v['f1@1']:6.2f}  F1@3 {v['f1@3']:6.2f}")
        elif k.startswith("ORACLE") or k == "pool_size_mean":
            print(f"{k:42s} {v:.2f}")
    write_json(os.path.join(args.out_dir, "metrics.json"), metrics, indent=1)
    write_json(os.path.join(args.out_dir, "retrieval_test_rerank.json"),
               {"split": "test", "method": "translator_rerank",
                "results": [{"query": {"name": n, "text": test[n]}, "topk": [{"text": t} for t in reranked[i][:3]],
                             "scores": scores_out[i]} for i, n in enumerate(names)]}, ensure_ascii=False)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["CSL_Daily", "How2Sign"])
    p.add_argument("--ckpt", required=True, help="SLT checkpoint used as the reranker (never trained on test)")
    p.add_argument("--pred_file", required=True, help="test_tmp_pres.txt of the baseline translator")
    p.add_argument("--cand_json", required=True, help="S3dn retrieval_test_s2t.json with a large --k")
    p.add_argument("--oracle_json", default="", help="TF-IDF oracle retrieval (analysis only)")
    p.add_argument("--k_tfidf", type=int, default=50)
    p.add_argument("--bs_cand", type=int, default=16)
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
