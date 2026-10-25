"""Word-level retrieval for CSL-Daily: a CTC gloss spotter on cached frame features.

CSL-Daily has no isolated-sign lexicon or word-level timing, only sentence-level
gloss sequences. The lexicon is therefore the 2,000 glosses of the *train*
split, and a small CTC head over the frozen frame features of `extract`
(feats/E_csl, ST-GCN of the released CSL-Daily checkpoint) learns to emit them.
CTC localises every emitted gloss to the frames of its spike, so a clip's
local segments are mapped to words without timing labels.

At test time the spotter sees only the pose features: no translation or gloss
of the sample is used. Hints for the training clips are cross-fitted (5 folds
grouped by sentence ID, so a held-out sentence is never seen in any signer's
version), which keeps train-time hint quality close to test time.

Outputs (--out_dir):
    hints_{train,dev,test}.json   {"results": [{"query": {"name"}, "hint", "pred", "frames", "conf"}]}
                                  for fine_tuning_rag.py --rag_mode hint
    metrics.json                  WER and gloss-set P/R/F1 per split, frequency prior,
                                  cross-fitted train vs test (hint-quality mismatch)
    examples.json                 representative test clips
    model_full.pt, model_fold*.pt, log.txt

    python word_retrieval_csl.py --feat_dir $OUT/feats/E_csl --out_dir $OUT/word_retrieval/csl/ctc_E_csl
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils
from signrag_common import FeatSplit, label_path, write_json

GLOSS_PREFIX = "Glosses: "          # the same prefix as the oracle --rag_mode gloss


class CTCSpotter(nn.Module):
    def __init__(self, d_in, n_cls, d=512, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.rnn = nn.GRU(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(d, n_cls)

    def forward(self, x, lengths):
        h = self.inp(x)
        h = h + F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        h = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.out(h).log_softmax(-1)


def load_split(feat_dir, split):
    fs = FeatSplit(os.path.join(feat_dir, split))
    labels = {v["name"]: v for v in utils.load_dataset_file(label_path("CSL_Daily", split)).values()}
    X = np.asarray(fs.X)                                   # whole split in RAM (float16)
    clips = [X[fs.offsets[i]:fs.offsets[i + 1]] for i in range(len(fs))]
    gloss = [labels[n]["gloss"] for n in fs.names]
    return {"names": fs.names, "texts": fs.texts, "x": clips, "gloss": gloss}


def batches(data, ids, vocab, bs, shuffle, rng):
    ids = np.array(ids)
    if shuffle:
        rng.shuffle(ids)
    for s in range(0, len(ids), bs):
        sel = ids[s:s + bs]
        lens = torch.tensor([len(data["x"][i]) for i in sel])
        x = torch.zeros(len(sel), int(lens.max()), data["x"][sel[0]].shape[1])
        for j, i in enumerate(sel):
            x[j, :lens[j]] = torch.from_numpy(data["x"][i].astype(np.float32))
        tgt = [torch.tensor([vocab[g] for g in data["gloss"][i] if g in vocab]) for i in sel]
        yield sel, x, lens, tgt


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


@torch.no_grad()
def decode(model, data, ids, vocab, inv, device, bs=64):
    """Greedy CTC: per clip the emitted glosses, the first frame of each spike
    run and its mean posterior."""
    model.eval()
    out = {}
    for sel, x, lens, _ in batches(data, ids, vocab, bs, False, None):
        lp = model(x.to(device), lens.to(device)).float().cpu()
        prob, arg = lp.exp().max(-1)
        for j, i in enumerate(sel):
            a, p = arg[j, :lens[j]].tolist(), prob[j, :lens[j]].tolist()
            pred, frames, conf, t = [], [], [], 0
            while t < len(a):
                u = t
                while u < len(a) and a[u] == a[t]:
                    u += 1
                if a[t] != 0:
                    pred.append(inv[a[t]])
                    frames.append(t)
                    conf.append(round(float(np.mean(p[t:u])), 3))
                t = u
            out[int(i)] = (pred, frames, conf)
    return out


def scores(preds, golds):
    """Corpus WER and mean gloss-set precision / recall / F1."""
    errs = sum(edit_distance(p, g) for p, g in zip(preds, golds))
    P, R, F1 = [], [], []
    for p, g in zip(preds, golds):
        ps, gs = set(p), set(g)
        inter = len(ps & gs)
        pr = inter / len(ps) if ps else 0.0
        rc = inter / len(gs) if gs else 0.0
        P.append(pr)
        R.append(rc)
        F1.append(2 * pr * rc / (pr + rc) if inter else 0.0)
    return {"wer": 100 * errs / max(1, sum(len(g) for g in golds)),
            "set_precision": 100 * float(np.mean(P)), "set_recall": 100 * float(np.mean(R)),
            "set_f1": 100 * float(np.mean(F1)), "avg_pred_len": float(np.mean([len(p) for p in preds])),
            "avg_gold_len": float(np.mean([len(g) for g in golds])), "n": len(golds)}


def train_model(args, train, train_ids, dev, vocab, inv, device, tag, log):
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model = CTCSpotter(train["x"][0].shape[1], len(vocab) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train_ids) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    dev_ids = list(range(len(dev["names"])))
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _, x, lens, tgt in batches(train, train_ids, vocab, args.batch_size, True, rng):
            lp = model(x.to(device), lens.to(device))
            loss = F.ctc_loss(lp.transpose(0, 1), torch.cat(tgt).to(device), lens.to(device),
                              torch.tensor([len(t) for t in tgt]).to(device), blank=0, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        dec = decode(model, dev, dev_ids, vocab, inv, device)
        wer = scores([dec[i][0] for i in dev_ids], dev["gloss"])["wer"]
        log(f"[{tag}] epoch {ep}: loss {tot / n:.3f}  dev WER {wer:.2f}  ({time.time() - t0:.0f}s)")
        if wer < best:
            best, best_state = wer, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    log(f"[{tag}] best dev WER {best:.2f}")
    return model, best


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    logf = open(os.path.join(args.out_dir, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = {s: load_split(args.feat_dir, s) for s in ("train", "dev", "test")}
    if args.limit:
        for s in data:
            data[s] = {k: v[:args.limit] for k, v in data[s].items()}
    counts = Counter(g for gl in data["train"]["gloss"] for g in gl)
    inv = {i + 1: g for i, g in enumerate(sorted(counts))}
    vocab = {g: i for i, g in inv.items()}
    log(f"lexicon = train glosses: {len(vocab)}; clips train/dev/test "
        f"{len(data['train']['names'])}/{len(data['dev']['names'])}/{len(data['test']['names'])}")
    for s in ("dev", "test"):
        toks = [g for gl in data[s]["gloss"] for g in gl]
        log(f"{s}: {100 * sum(g in vocab for g in toks) / len(toks):.1f}% of gloss tokens in the lexicon")

    # full model first (dev / test hints), so its WER is known early
    n_tr = len(data["train"]["names"])
    path = os.path.join(args.out_dir, "model_full.pt")
    if os.path.isfile(path):
        model = CTCSpotter(data["train"]["x"][0].shape[1], len(vocab) + 1).to(device)
        model.load_state_dict(torch.load(path))
    else:
        model, _ = train_model(args, data["train"], list(range(n_tr)), data["dev"], vocab, inv, device, "full", log)
        torch.save(model.state_dict(), path)
    full_model = model
    for s in ("dev", "test"):
        ids = range(len(data[s]["names"]))
        dec = decode(full_model, data[s], list(ids), vocab, inv, device)
        log(f"[full] {s} WER {scores([dec[i][0] for i in ids], data[s]['gloss'])['wer']:.2f}")

    # cross-fitted train hints: folds grouped by sentence ID (S######)
    sents = sorted({n[:7] for n in data["train"]["names"]})
    fold_of = {s: k % args.folds for k, s in enumerate(np.random.RandomState(0).permutation(sents))}
    folds = np.array([fold_of[n[:7]] for n in data["train"]["names"]])
    train_dec = {}
    for k in range(args.folds):
        path = os.path.join(args.out_dir, f"model_fold{k}.pt")
        tr_ids, held = np.where(folds != k)[0].tolist(), np.where(folds == k)[0].tolist()
        if os.path.isfile(path):
            model = CTCSpotter(data["train"]["x"][0].shape[1], len(vocab) + 1).to(device)
            model.load_state_dict(torch.load(path))
            log(f"[fold{k}] loaded {path}")
        else:
            model, _ = train_model(args, data["train"], tr_ids, data["dev"], vocab, inv, device, f"fold{k}", log)
            torch.save(model.state_dict(), path)
        train_dec.update(decode(model, data["train"], held, vocab, inv, device))

    decs = {"train": train_dec}
    for s in ("dev", "test"):
        decs[s] = decode(full_model, data[s], list(range(len(data[s]["names"]))), vocab, inv, device)
    selffit = decode(full_model, data["train"], list(range(n_tr)), vocab, inv, device)

    metrics = {"lexicon": "CSL-Daily train glosses", "lexicon_size": len(vocab), "folds": args.folds,
               "feat_dir": args.feat_dir, "feature_encoder": FeatSplit(os.path.join(args.feat_dir, "train")).meta["finetune"]}
    for s in ("train", "dev", "test"):
        ids = range(len(data[s]["names"]))
        metrics[s if s != "train" else "train_crossfit"] = scores([decs[s][i][0] for i in ids], data[s]["gloss"])
    metrics["train_selffit"] = scores([selffit[i][0] for i in range(n_tr)], data["train"]["gloss"])
    # frequency prior: the N most frequent train glosses for every clip (N = mean predicted length)
    n_prior = max(1, round(metrics["test"]["avg_pred_len"]))
    prior = [g for g, _ in counts.most_common(n_prior)]
    metrics["test_frequency_prior"] = scores([prior] * len(data["test"]["names"]), data["test"]["gloss"])
    write_json(os.path.join(args.out_dir, "metrics.json"), metrics, indent=2)
    for k, v in metrics.items():
        if isinstance(v, dict):
            log(f"{k:22s} WER {v['wer']:6.2f}  set P/R/F1 {v['set_precision']:.1f}/{v['set_recall']:.1f}/"
                f"{v['set_f1']:.1f}  pred/gold len {v['avg_pred_len']:.1f}/{v['avg_gold_len']:.1f}")

    for s in ("train", "dev", "test"):
        res = []
        for i, name in enumerate(data[s]["names"]):
            pred, frames, conf = decs[s][i]
            res.append({"query": {"name": name}, "hint": GLOSS_PREFIX + " ".join(pred) if pred else "",
                        "pred": pred, "frames": frames, "conf": conf})
        write_json(os.path.join(args.out_dir, f"hints_{s}.json"),
                   {"split": s, "source": "crossfit" if s == "train" else "full", "results": res}, ensure_ascii=False)
    ex_ids = np.random.RandomState(1).choice(len(data["test"]["names"]), 10, replace=False)
    examples = [{"name": data["test"]["names"][i], "text": data["test"]["texts"][i],
                 "gold_gloss": data["test"]["gloss"][i], "pred": decs["test"][i][0],
                 "frames": decs["test"][i][1], "conf": decs["test"][i][2]} for i in ex_ids]
    write_json(os.path.join(args.out_dir, "examples.json"), examples, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N clips of every split (smoke tests)")
    main(p.parse_args())
