"""Step 2 of doc/signrag_style_retriever_plan.md: contrastive training of the
alignment head phi (and, with --train_encoder, the ST-GCN encoder) against a
frozen official mT5-base text tower plus a linear text head (init identity).

    --objective pooled  (ours, S2-a / S2-s)
        masked-mean phi(X) vs mean-pooled text; sign->text over the FULL
        deduplicated train-text gallery, text->sign in-batch, one tau
    --objective fine    (SignRAG Stage I, S2-d)
        frame->token soft alignment (Eq. 21-23, learnable tau) inside a
        bidirectional in-batch InfoNCE with tau'; optional --slt_weight adds
        L_slt through the checkpoint's frozen translation mT5

    --positives exact     identical normalised texts are positives
    --positives semantic  also every text whose paraphrase-model cosine is
                          >= --pos_threshold (sentences that repeat the same
                          meaning are not used as negatives)

    --train_encoder off (default): ST-GCN frozen, trains on cached X (fast)
    --train_encoder on           : ST-GCN trained too (--lr_encoder), poses
                                   are loaded per batch; also writes
                                   encoder.pth, which Step 3 must re-extract
                                   features with

Memory does not grow with the number of negatives: the pooled sign->text
loss over the whole gallery is one [B, N] matrix (< 2 GiB at B=256 on
OpenASL). What grows with B is the fine objective's [B, B, M, L] tensor
(--sim_chunk: checkpointed chunks) and the ST-GCN activations with
--train_encoder (--encoder_chunk: GradCache, two passes). Both give the
same loss and gradients as the unchunked computation.

phi is initialised from the checkpoint's pose_proj (and the encoder from the
same checkpoint), so step 0 reproduces Step 1; phi_init.pth is saved for that
check. --audit_positives prints how many semantic positives each threshold
creates, with example pairs, and exits (needs only the label files).

    python signrag_contrastive.py --feat_dir FEATS --run_dir RUN [--positives semantic] [--train_encoder]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import utils
from config import mt5_path
from signrag_common import (DEFAULT_OUT, FeatSplit, Phi, bidirectional_recall, csr_rows_mask, gather_padded,
                            get_mt5_cache, get_sem_cache, identity_linear, label_path, masked_mean,
                            multi_positive_nce, pose_proj_phi, semantic_neighbours, soft_match_pair_scores,
                            write_json)


def split_train_dev(feat_dir, dev_frac, seed):
    """(train feats, train ids, dev feats, dev ids). The real dev split is
    used when extracted (OpenASL); otherwise a video-disjoint pseudo-dev of
    `dev_frac` of the train videos (How2Sign has no dev split)."""
    train = FeatSplit(os.path.join(feat_dir, "train"))
    dev_dir = os.path.join(feat_dir, "dev")
    if os.path.isfile(os.path.join(dev_dir, "meta.json")):
        dev = FeatSplit(dev_dir)
        return train, np.arange(len(train)), dev, np.arange(len(dev))
    videos = sorted(set(train.videos))
    held = set(np.random.RandomState(seed).choice(videos, max(1, round(dev_frac * len(videos))),
                                                  replace=False))
    is_dev = np.array([v in held for v in train.videos])
    return train, np.where(~is_dev)[0], train, np.where(is_dev)[0]


def to_cuda(src):
    for k, v in src.items():
        if torch.is_tensor(v):
            src[k] = v.cuda(non_blocking=True)
            if v.is_floating_point():
                src[k] = src[k].float()
    return src


def split_src(src, size):
    """A collated pose batch -> list of batch slices of `size` clips."""
    n = src["attention_mask"].size(0)
    return [{k: v[s:s + size] if torch.is_tensor(v) or isinstance(v, list) else v
             for k, v in src.items()} for s in range(0, n, size)]


def pose_loader(ds_cls, uargs, dataset, split, names, batch_size, workers, train):
    """DataLoader over the clips `names` of a label split (encoder mode)."""
    ds = ds_cls(path=label_path(dataset, split), args=uargs, phase=split)
    keep = set(names)
    ds.list = [k for k in ds.list if ds.raw_data[k]["name"] in keep]
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=workers, collate_fn=ds.collate_fn, pin_memory=True)


class SltLoss:
    """L_slt: the checkpoint's translation mT5 (frozen) reading phi(X) in
    place of pose_proj(X), exactly the baseline input [prompt][pose]."""

    def __init__(self, model):
        self.mt5, self.tok, self.lang = model.mt5_model.eval(), model.mt5_tokenizer, model.lang
        for p in self.mt5.parameters():
            p.requires_grad_(False)

    def __call__(self, z, mask, texts):
        dev = z.device
        prefix = self.tok([f"Translate sign language video to {self.lang}: "] * len(texts),
                          padding="longest", return_tensors="pt").to(dev)
        emb = self.mt5.encoder.embed_tokens(prefix["input_ids"])
        labels = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=50)["input_ids"].to(dev)
        labels[labels == self.tok.pad_token_id] = -100
        out = self.mt5(inputs_embeds=torch.cat([emb, z.to(emb.dtype)], 1),
                       attention_mask=torch.cat([prefix["attention_mask"], mask.long()], 1),
                       labels=labels)
        logits = out.logits.float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               ignore_index=-100, label_smoothing=0.2)


@torch.no_grad()
def audit_positives(args, sem, g_tids, clip_cols, test_tids):
    """How many extra positives each threshold creates, on train texts and
    clips, how often test texts have a semantic near-duplicate in train (the
    input for train-time retrieval filters), and example pairs per band."""
    device = torch.device("cuda")
    E = torch.as_tensor(sem.pooled[g_tids], device=device).float()
    report, rng = {"thresholds": {}}, np.random.RandomState(args.seed)
    test_E = torch.as_tensor(sem.pooled[test_tids], device=device).float()
    g_t = torch.as_tensor(g_tids, device=device)
    test_max = []
    for s in range(0, len(test_E), 1024):
        sim = test_E[s:s + 1024] @ E.t()
        same = torch.as_tensor(test_tids[s:s + 1024], device=device)[:, None] == g_t[None]
        test_max.append(sim.masked_fill(same, -1).max(-1).values)
    test_max = torch.cat(test_max).cpu().numpy()
    keys = sem.keys
    for th in args.audit_thresholds:
        indptr, indices = semantic_neighbours(E, th, args.pos_max, device)
        extra = (indptr[1:] - indptr[:-1] - 1).cpu().numpy()
        clip_extra = extra[clip_cols]
        row = {"texts_with_extra_pos": float((extra > 0).mean()),
               "clips_with_extra_pos": float((clip_extra > 0).mean()),
               "mean_extra_pos_per_clip": float(clip_extra.mean()),
               "test_with_neighbour_ge_th": float((test_max >= th).mean())}
        cand = np.where(extra > 0)[0]
        examples = []
        for i in (rng.choice(cand, min(6, len(cand)), replace=False) if len(cand) else []):
            nb = indices[indptr[i] + 1:indptr[i + 1]].cpu().numpy()
            j = nb[-1]                                   # least similar positive: the borderline case
            examples.append([keys[g_tids[i]], keys[g_tids[j]], round(float(E[i] @ E[j]), 3)])
        row["examples_least_similar_positive"] = examples
        report["thresholds"][str(th)] = row
        print(f"\nthreshold {th}: texts with extra positives {row['texts_with_extra_pos']:.1%}, clips "
              f"{row['clips_with_extra_pos']:.1%}, mean extra/clip {row['mean_extra_pos_per_clip']:.2f}, "
              f"test texts with a train neighbour >= th {row['test_with_neighbour_ge_th']:.1%}")
        for a, b, c in examples:
            print(f"   {c:.3f}  {a!r}  ~  {b!r}")
    os.makedirs(args.run_dir, exist_ok=True)
    write_json(os.path.join(args.run_dir, "positive_audit.json"), report, indent=2)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.run_dir, exist_ok=True)

    if args.audit_positives:
        # texts only: runs on the label files, before any feature extraction
        sem = get_sem_cache(args.cache_root, args.dataset, args.sem_model)
        label_tids = lambda split: sem.ids([it["text"] for it in
                                            utils.load_dataset_file(label_path(args.dataset, split)).values()])
        tids = label_tids("train")
        g_tids = np.unique(tids)
        audit_positives(args, sem, g_tids, np.searchsorted(g_tids, tids), np.unique(label_tids("test")))
        return

    train, train_ids, dev, dev_ids = split_train_dev(args.feat_dir, args.dev_frac, args.seed)
    dataset = train.meta["dataset"]
    finetune = args.finetune or train.meta["finetune"]
    text_ckpt = None if args.text_encoder == "official" else (args.text_ckpt or finetune)
    texts = get_mt5_cache(args.cache_root, dataset, args.mt5_path, args.text_layer, encoder_ckpt=text_ckpt)
    sem = get_sem_cache(args.cache_root, dataset, args.sem_model)
    assert sem.keys == texts.keys, "text caches disagree on the text list"
    print(f"text tower: {'official mT5-base' if text_ckpt is None else 'mT5 encoder of ' + text_ckpt} (frozen)")

    # training gallery = deduplicated texts of the training clips
    tids = texts.ids(train.texts)
    g_tids = np.unique(tids[train_ids])
    clip_cols = np.searchsorted(g_tids, tids[train_ids])            # column of each training clip
    n_g = len(g_tids)
    print(f"{dataset}: {len(train_ids)} training clips, {n_g} unique texts; dev {len(dev_ids)} clips; "
          f"encoder {'TRAINED' if args.train_encoder else 'frozen (cached X)'}")

    if args.positives == "semantic":
        indptr, indices = semantic_neighbours(sem.pooled[g_tids], args.pos_threshold, args.pos_max, device)
        extra = (indptr[1:] - indptr[:-1] - 1).float()
        print(f"semantic positives (cos >= {args.pos_threshold}): {(extra > 0).float().mean():.1%} of texts "
              f"gain extra positives, mean {extra.mean():.2f}")
    else:
        indptr = torch.arange(n_g + 1, device=device)
        indices = torch.arange(n_g, device=device)

    cols = torch.as_tensor(clip_cols, device=device)
    t_pool = torch.as_tensor(texts.pooled[g_tids], device=device).float()
    if args.objective == "fine":
        t_tok = torch.from_numpy(np.array(texts.tokens)).to(device)
        t_off = torch.as_tensor(texts.tok_offsets, device=device)
        g_tids_t = torch.as_tensor(g_tids, device=device)

    # ---- model pieces ------------------------------------------------------
    model = None
    if args.train_encoder or args.slt_weight > 0:
        from signrag_retrieval import build_unisign
        model, uargs = build_unisign(dataset, finetune)
        model.cuda()
    if args.train_encoder:
        assert model.pose_proj.in_features == train.dim
        pose_proj = model.pose_proj
        model.pose_proj = nn.Identity()                 # encode_pose -> X
        enc_params = [p for n, p in model.named_parameters() if not n.startswith("mt5_model")]
        model.train()
        model.mt5_model.eval()                          # frozen translator: no dropout in L_slt
    else:
        # the whole split on the GPU when it fits (H2S ~10 GB); otherwise it stays a
        # memmap-backed CPU tensor (OpenASL ~37 GB, shared page cache) gathered per batch
        x_dev = args.x_device if args.x_device != "auto" else \
            ("cuda" if train.X.nbytes < 0.4 * torch.cuda.mem_get_info()[0] else "cpu")
        if x_dev == "cuda":
            X = torch.from_numpy(np.array(train.X)).to(device)
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")              # read-only memmap: never written
                X = torch.from_numpy(np.asarray(train.X))
        off = torch.as_tensor(train.offsets, device=X.device)
        tr_ids = torch.as_tensor(train_ids, device=X.device)
        print(f"cached X ({train.X.nbytes / 2**30:.1f} GiB) kept on {x_dev}")
    slt = SltLoss(model) if args.slt_weight > 0 else None

    # text->sign over the whole training sign gallery: with a linear phi on frozen X,
    # masked_mean(phi(X)) == phi(masked_mean(X)), so it costs one [N, 1024] matmul per step
    linear_frozen = args.objective == "pooled" and not args.train_encoder and not args.phi_mlp_hidden
    t2s_gallery = args.t2s_negatives == "gallery" or (args.t2s_negatives == "auto" and linear_frozen)
    if t2s_gallery:
        if not linear_frozen:
            raise ValueError("--t2s_negatives gallery needs --objective pooled, a linear phi "
                             "(--phi_mlp_hidden 0) and a frozen encoder")
        with torch.no_grad():
            xbar = torch.cat([masked_mean(x.to(device).float(), m.to(device)) for x, m in
                              (gather_padded(X, off, tr_ids[s:s + 512]) for s in range(0, len(tr_ids), 512))])
    print(f"text->sign negatives: {'all ' + str(len(train_ids)) + ' training clips' if t2s_gallery else 'in-batch'}")

    phi = pose_proj_phi(args.feat_dir, args.phi_mlp_hidden) if args.init == "pose_proj" \
        else Phi(train.dim, t_pool.size(1), args.phi_mlp_hidden)
    phi = phi.to(device)
    head = identity_linear(t_pool.size(1)).to(device)
    log_tau = nn.Parameter(torch.tensor(math.log(args.tau_init), device=device))
    log_tau_attn = nn.Parameter(torch.tensor(math.log(args.tau_attn_init), device=device))
    temps = [log_tau] + ([log_tau_attn] if args.objective == "fine" else [])
    groups = [
        {"params": phi.parameters(), "lr": args.lr_phi, "weight_decay": args.weight_decay},
        {"params": head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": temps, "lr": args.lr_head, "weight_decay": 0.0},
    ]
    if args.train_encoder:
        groups.append({"params": enc_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay})
    opt = torch.optim.AdamW(groups)

    # ---- data --------------------------------------------------------------
    if args.train_encoder:
        from datasets import S2T_Dataset
        from signrag_retrieval import S2T_Dataset_Uniform
        pos_of = {train.names[i]: p for p, i in enumerate(train_ids)}
        # random frame subsampling of long clips (the baseline's behaviour) for training,
        # uniform for dev so the selection metric is repeatable
        train_loader = pose_loader(S2T_Dataset, uargs, dataset, "train", list(pos_of), args.batch_size,
                                   args.num_workers, train=True)
        dev_split = dev.meta["split"]
        dev_loader = pose_loader(S2T_Dataset_Uniform, uargs, dataset, dev_split,
                                 [dev.names[i] for i in dev_ids], 64, args.num_workers, train=False)
        steps_per_epoch = len(train_loader)
    else:
        steps_per_epoch = len(train_ids) // args.batch_size

    def train_batches(gen):
        """Yields (X [B, L, 1024] -- or the pose batch with --train_encoder --,
        mask, positions into train_ids)."""
        if args.train_encoder:
            for src, _ in train_loader:
                src = to_cuda(src)
                b = torch.as_tensor([pos_of[n] for n in src["name_batch"]], device=device)
                yield src, src["attention_mask"].bool(), b
        else:
            perm = torch.randperm(len(train_ids), generator=gen, device=device)
            for s in range(0, steps_per_epoch * args.batch_size, args.batch_size):
                b = perm[s:s + args.batch_size]
                x, m = gather_padded(X, off, tr_ids[b.to(X.device)])
                yield x.to(device, non_blocking=True).float(), m.to(device), b

    def dev_batches():
        """Yields (X, mask, texts) for every dev clip, with the current encoder."""
        if args.train_encoder:
            for src, tgt in dev_loader:
                src = to_cuda(src)
                yield model.encode_pose(src), src["attention_mask"].bool(), tgt["gt_sentence"]
        else:
            for s in range(0, len(dev_ids), 64):
                x, m = dev.padded(dev_ids[s:s + 64])
                yield x.to(device).float(), m.to(device), [dev.texts[i] for i in dev_ids[s:s + 64]]

    total = args.max_steps or args.epochs * steps_per_epoch
    warmup = min(100, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    tau = lambda: log_tau.exp().clamp(min=0.01)
    tau_attn = lambda: log_tau_attn.exp().clamp(min=0.01)

    @torch.no_grad()
    def dev_eval():
        """Within-dev retrieval in both directions (sign->text r@K, text->sign
        t2s_r@K, and their paraphrase-tolerant sem_* variants)."""
        phi.eval(), head.eval()
        if model is not None and args.train_encoder:
            model.eval()
        out = bidirectional_recall(phi, head, texts, sem, dev_batches(), args.objective, tau_attn(),
                                   args.pos_threshold, device=device)
        phi.train(), head.train()
        if model is not None and args.train_encoder:
            model.train()
            model.mt5_model.eval()
        return out

    def save(tag, epoch, dev_metrics):
        ckpt = {"phi": phi.state_dict(), "phi_cfg": phi.cfg, "text_head": head.state_dict(),
                "text_cache": texts.dir, "objective": args.objective, "tau": float(tau()),
                "tau_attn": float(tau_attn()), "positives": args.positives,
                "pos_threshold": args.pos_threshold, "epoch": epoch, "dev": dev_metrics,
                "args": vars(args), "encoder": None}
        if args.train_encoder:
            # a full Uni-Sign checkpoint with the trained ST-GCN; Step 3 re-extracts X with it
            enc_path = os.path.abspath(os.path.join(args.run_dir, f"encoder{tag}.pth"))
            model.pose_proj = pose_proj
            torch.save({"model": model.state_dict()}, enc_path)
            model.pose_proj = nn.Identity()
            ckpt["encoder"] = enc_path
        torch.save(ckpt, os.path.join(args.run_dir, f"phi{tag}.pth"))

    write_json(os.path.join(args.run_dir, "args.json"), vars(args), indent=2)
    log = open(os.path.join(args.run_dir, "log.txt"), "a")
    d0 = dev_eval()
    save("_init", 0, d0)
    print(f"epoch 0 (init = {args.init}): dev {d0}")
    log.write(json.dumps({"epoch": 0, **{f"dev_{k}": v for k, v in d0.items()}}) + "\n")
    best, bad, step = -1.0, 0, 0
    gen = torch.Generator(device=device).manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        for x, m, b in train_batches(gen):
            if args.train_encoder:
                # GradCache pass 1: X of the whole batch without a graph, so the
                # contrastive batch is not limited by ST-GCN activation memory
                chunks = split_src(x, args.encoder_chunk)
                with torch.no_grad():
                    x = torch.cat([model.encode_pose(c) for c in chunks])
                x.requires_grad_(True)
            y = cols[b]
            pos_full = csr_rows_mask(indptr, indices, y, n_g)        # [B, n_g]
            pos_batch = pos_full[:, y]                               # [B, B]
            f = phi(x)
            if args.objective == "pooled":
                zs = F.normalize(masked_mean(f, m), dim=-1)
                zt = F.normalize(head(t_pool), dim=-1)
                if t2s_gallery:                               # every training clip is a candidate
                    l_t2s = zt[y] @ F.normalize(phi(xbar), dim=-1).t() / tau()
                    pos_t2s = pos_full[:, cols]               # clip j positive if its text is in P(y_i)
                else:
                    l_t2s, pos_t2s = zt[y] @ zs.t() / tau(), pos_batch
                loss = 0.5 * (multi_positive_nce(zs @ zt.t() / tau(), pos_full, args.pos_loss)
                              + multi_positive_nce(l_t2s, pos_t2s, args.pos_loss))
            else:
                t, tm = gather_padded(t_tok, t_off, g_tids_t[y])
                fn = F.normalize(f, dim=-1)
                tn = F.normalize(head(t.float()), dim=-1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s2t, t2s = soft_match_pair_scores(fn, m, tn, tm, tau_attn(), args.sim_chunk)
                loss = 0.5 * (multi_positive_nce(s2t / tau(), pos_batch, args.pos_loss)
                              + multi_positive_nce(t2s / tau(), pos_batch, args.pos_loss))
            if slt is not None:
                k = min(args.slt_batch, len(b))
                ids_k = train_ids[b[:k].cpu().numpy()]
                loss = loss + args.slt_weight * slt(f[:k], m[:k], [train.texts[i] for i in ids_k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.train_encoder:
                # GradCache pass 2: re-encode chunk by chunk and push dL/dX into the ST-GCN
                for c, g in zip(chunks, x.grad.split(args.encoder_chunk)):
                    model.encode_pose(c).backward(g)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        d = dev_eval()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "tau": float(tau()),
               "tau_attn": float(tau_attn()) if args.objective == "fine" else None,
               **{f"dev_{k}": v for k, v in d.items()}, "time_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch}: loss {rec['train_loss']:.4f} tau {rec['tau']:.4f} dev {d} ({rec['time_s']}s)",
              flush=True)
        if d[args.select] > best:
            best, bad = d[args.select], 0
            save("", epoch, d)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no {args.select} gain for {args.patience} epochs")
                break
        if args.max_steps and step >= args.max_steps:
            break
    # done.json marks a finished run (phi.pth already exists after the first epoch)
    write_json(os.path.join(args.run_dir, "done.json"), {"best_dev": {args.select: best}, "epochs_run": epoch})
    print(f"best dev {args.select} = {best:.2f} -> {os.path.join(args.run_dir, 'phi.pth')}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feat_dir", default="", help="output of `signrag_retrieval.py extract`")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", default="How2Sign", choices=["How2Sign", "OpenASL", "CSL_Daily"],
                   help="only for --audit_positives (training reads it from the features)")
    p.add_argument("--cache_root", default=os.path.join(DEFAULT_OUT, "text_cache"))
    p.add_argument("--mt5_path", default=mt5_path, help="official mT5-base (text tower)")
    p.add_argument("--text_layer", type=int, default=-1, help="mT5 encoder layer; -1 = final")
    p.add_argument("--text_encoder", default="official", choices=["official", "checkpoint"],
                   help="text tower weights: official mT5-base, or the translation-finetuned mT5 "
                        "encoder of a Uni-Sign checkpoint (--text_ckpt, default: the extract checkpoint)")
    p.add_argument("--text_ckpt", default="", help="Uni-Sign checkpoint for --text_encoder checkpoint")
    p.add_argument("--objective", default="pooled", choices=["pooled", "fine"])
    p.add_argument("--positives", default="exact", choices=["exact", "semantic"])
    p.add_argument("--sem_model", default="sentence-transformers/paraphrase-MiniLM-L6-v2")
    p.add_argument("--pos_threshold", type=float, default=0.85,
                   help="paraphrase cosine for semantic positives (audit: doc §3.4)")
    p.add_argument("--pos_max", type=int, default=64)
    p.add_argument("--pos_loss", default="any", choices=["any", "all"])
    p.add_argument("--init", default="pose_proj", choices=["pose_proj", "random"])
    p.add_argument("--phi_mlp_hidden", type=int, default=0, help=">0: residual MLP (S2-c)")
    p.add_argument("--train_encoder", action="store_true", help="also train the ST-GCN encoder (S2-f)")
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--x_device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="where the cached training features live (auto: GPU if < 40%% of free memory)")
    p.add_argument("--encoder_chunk", type=int, default=16,
                   help="--train_encoder: clips per ST-GCN pass (GradCache); memory ~8 GiB per 16 clips")
    p.add_argument("--sim_chunk", type=int, default=32,
                   help="fine objective: texts per checkpointed chunk of the [B, B, M, L] tensor; 0 = off")
    p.add_argument("--num_workers", type=int, default=16, help="pose loading (--train_encoder)")
    p.add_argument("--slt_weight", type=float, default=0.0, help=">0: add L_slt (S2-d)")
    p.add_argument("--slt_batch", type=int, default=32)
    p.add_argument("--finetune", default="", help="Uni-Sign checkpoint; default: the extract checkpoint")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=0, help="stop after N steps (smoke tests)")
    p.add_argument("--lr_phi", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--tau_init", type=float, default=0.07)
    p.add_argument("--tau_attn_init", type=float, default=0.07)
    p.add_argument("--dev_frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--t2s_negatives", default="auto", choices=["auto", "gallery", "batch"],
                   help="text->sign negatives: all training clips (pooled + linear phi + frozen "
                        "encoder; auto picks it when possible) or in-batch")
    p.add_argument("--select", default="r5_mean",
                   choices=["r5_mean", "sem_r5_mean", "r1", "r5", "r10", "t2s_r1", "t2s_r5", "t2s_r10",
                            "sem_r5", "t2s_sem_r5"],
                   help="dev metric for model selection; r5_mean = mean of sign->text and text->sign R@5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--audit_positives", action="store_true")
    p.add_argument("--audit_thresholds", type=float, nargs="+", default=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7])
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
