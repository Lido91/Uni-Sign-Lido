"""Contrastive probe: is the SignSMD description of a clip semantically
aligned with that clip's GT translation?

Text-only diagnostic for the SignSMD path (no pose, no DeepSpeed, single
GPU).  Both texts of a pair are encoded by ONE shared (siamese) text
encoder:

    GT translation ── encoder ── pool ── head ── z_A ─┐
                         (shared weights)             ├─ cosine similarity
    SignSMD text   ── encoder ── pool ── head ── z_B ─┘

and trained with a symmetric InfoNCE loss over in-batch negatives, so
matched (GT, SMD) pairs move together and mismatched pairs apart.

Readout ("is the SMD correct?"):

- corpus level: retrieval R@1/R@5/R@10 over the FULL evaluation split
  (chance = 1/N).  If even a trained probe cannot match SMD to its own
  translation above chance, the SMD texts carry no clip-specific semantic
  signal; the higher the retrieval, the more translation-relevant content
  the compiler preserves.
- clip level: ``per_sample_scores_<split>.csv`` ranks every clip by matched
  cosine similarity and by margin over its hardest in-split negative —
  low-margin clips are the ones whose SMD is likely wrong/generic.
- zero-shot floor: ``--eval --head none`` without a checkpoint scores the
  raw pretrained encoder (surface forms differ a lot, so expect ~chance;
  the trained probe is the meaningful number).

Encoder default is the repo's local mT5-base (config.mt5_path — offline,
same tokenizer as the fusion models); ``--encoder hf --hf_model
bert-base-uncased`` switches to any HuggingFace encoder if downloads are
available.

Train / evaluate (see script/contrastive_gt_smd.sh):

    python contrastive_gt_smd.py \
        --dataset How2Sign \
        --signsmd_text_root data/How2Sign/signsmd2d_stats_v0_1/texts \
        --output_dir out/contrastive_gt_smd_h2s

    python contrastive_gt_smd.py ... --eval \
        --resume out/contrastive_gt_smd_h2s/best_checkpoint.pth
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import get_scheduler

import utils
from config import (
    dev_label_paths,
    mt5_path,
    test_label_paths,
    train_label_paths,
)

MISSING_POLICIES = ("error", "skip", "empty")
KNOWN_DATASETS = ("How2Sign", "OpenASL")


def parse_datasets(spec):
    datasets = [d.strip() for d in spec.replace("+", ",").split(",") if d.strip()]
    for d in datasets:
        if d not in KNOWN_DATASETS:
            raise ValueError(f"unknown dataset {d!r}; "
                             f"choose from {KNOWN_DATASETS}")
    return datasets


def resolve_text_root(args, dataset):
    root = args.signsmd_text_root
    if "{}" in root:
        root = root.format(dataset)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"SignSMD text root for {dataset} does not exist: {root}"
        )
    return root


def get_args_parser():
    parser = argparse.ArgumentParser("GT-translation vs SignSMD contrastive probe")
    # data
    parser.add_argument("--dataset", default="How2Sign",
                        help="Comma-separated subset of {How2Sign, OpenASL}; "
                             "multiple datasets pool their train pairs into "
                             "one probe (eval stays per-dataset).")
    parser.add_argument("--signsmd_text_root", required=True,
                        help="Directory of compiled SignSMD descriptions, one "
                             "<clip-name>.txt per sample (clip-name = "
                             "video_path minus .mp4). With multiple "
                             "datasets, use a {} template resolved per "
                             "dataset, e.g. data/{}/signsmd2d_stats_v0_1/texts")
    parser.add_argument("--signsmd_on_missing", default="error",
                        choices=list(MISSING_POLICIES),
                        help="error: fail fast on a missing description; "
                             "skip: drop the sample (reported); "
                             "empty: substitute \"\" (smoke tests only).")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Held-out slice of the train split used for "
                             "checkpoint selection when the dataset has no "
                             "dev split (How2Sign); datasets with a dev "
                             "split (OpenASL) validate on it instead.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: keep only the first N pairs per split.")
    # model
    parser.add_argument("--encoder", default="mt5", choices=["mt5", "hf"],
                        help="mt5: local config.mt5_path encoder (offline); "
                             "hf: any HuggingFace AutoModel via --hf_model.")
    parser.add_argument("--hf_model", default="bert-base-uncased")
    parser.add_argument("--head", default="mlp", choices=["mlp", "linear", "none"],
                        help="Projection on top of the mean-pooled encoder "
                             "output; none = raw pooled embedding (use for "
                             "the zero-shot floor).")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Train only the projection head (linear-probe "
                             "regime); default fine-tunes the encoder too.")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Init of the learnable CLIP-style temperature.")
    parser.add_argument("--text_max_tokens", type=int, default=64)
    parser.add_argument("--smd_max_tokens", type=int, default=512,
                        help="Tail-truncation cap for the SMD segment "
                             "(same semantics as --signsmd_max_tokens).")
    # optimization
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Also the number of in-batch negatives + 1.")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LR of the head + temperature.")
    parser.add_argument("--lr_encoder", type=float, default=1e-5,
                        help="LR of the encoder when not frozen.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true",
                        help="bfloat16 autocast for the encoder forward.")
    parser.add_argument("--grad_checkpointing", action="store_true")
    # run control
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate on the test split only (no training).")
    parser.add_argument("--resume", default="",
                        help="Checkpoint to load (strict=False).")
    parser.add_argument("--output_dir", default="out/contrastive_gt_smd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ---------------------------------------------------------------- data


def load_pairs(label_path, text_root, args, dataset, phase):
    """(name, GT translation, SMD text) triples; no pose is touched."""
    raw = utils.load_dataset_file(label_path)
    pairs, skipped = [], 0
    for key in raw:
        sample = raw[key]
        text_path = os.path.join(
            text_root,
            sample["video_path"].replace(".mp4", ".txt"),
        )
        if os.path.isfile(text_path):
            with open(text_path, encoding="utf-8") as f:
                smd = f.read().strip()
        elif args.signsmd_on_missing == "skip":
            skipped += 1
            continue
        elif args.signsmd_on_missing == "empty":
            smd = ""
        else:
            raise FileNotFoundError(
                f"SignSMD text missing for sample "
                f"{sample['video_path']!r}: {text_path} (generate it first, "
                f"or pass --signsmd_on_missing skip)"
            )
        pairs.append(
            {"name": sample["name"], "gt": sample["text"], "smd": smd,
             "dataset": dataset}
        )
    if skipped:
        print(f"[{dataset}/{phase}] skipped {skipped} samples without "
              f"an SMD text ({len(pairs)} kept)")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[{dataset}/{phase}] {len(pairs)} (GT, SMD) pairs")
    return pairs


def split_train_val(pairs, val_fraction, seed):
    """Deterministic held-out slice for checkpoint selection."""
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(pairs) * val_fraction)))
    val_idx = set(order[:n_val])
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [pairs[i] for i in sorted(val_idx)]
    return train, val


# ---------------------------------------------------------------- model


class TextContrastiveProbe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.encoder == "mt5":
            from transformers import MT5EncoderModel, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(mt5_path)
            self.encoder = MT5EncoderModel.from_pretrained(mt5_path)
        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
            self.encoder = AutoModel.from_pretrained(args.hf_model)

        if args.grad_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        if args.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        hidden = getattr(self.encoder.config, "hidden_size", None) \
            or self.encoder.config.d_model
        if args.head == "linear":
            self.head = nn.Linear(hidden, args.proj_dim)
        elif args.head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, args.proj_dim),
            )
        else:
            self.head = None

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / args.temperature))
        )

    def encode(self, texts, max_tokens):
        device = next(self.encoder.parameters()).device
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        if self.head is not None:
            pooled = self.head(pooled.float())
        return F.normalize(pooled.float(), dim=-1)

    def forward(self, gt_texts, smd_texts):
        """Symmetric InfoNCE over in-batch negatives."""
        z_gt = self.encode(gt_texts, self.args.text_max_tokens)
        z_smd = self.encode(smd_texts, self.args.smd_max_tokens)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * z_gt @ z_smd.t()
        labels = torch.arange(len(gt_texts), device=logits.device)
        loss = (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)
        ) / 2
        return loss


# ---------------------------------------------------------------- eval


@torch.no_grad()
def encode_split(model, pairs, args):
    if len(pairs) < 2:
        raise ValueError(
            f"retrieval needs at least 2 pairs, got {len(pairs)} — "
            f"SMD texts missing for this split?"
        )
    model.eval()
    z_gt, z_smd = [], []
    for i in range(0, len(pairs), args.eval_batch_size):
        chunk = pairs[i : i + args.eval_batch_size]
        with torch.autocast(
            device_type=args.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=args.bf16,
        ):
            z_gt.append(
                model.encode([p["gt"] for p in chunk],
                             args.text_max_tokens).cpu()
            )
            z_smd.append(
                model.encode([p["smd"] for p in chunk],
                             args.smd_max_tokens).cpu()
            )
    return torch.cat(z_gt), torch.cat(z_smd)


def retrieval_metrics(z_gt, z_smd):
    sim = z_gt @ z_smd.t()  # [N, N]; row = GT, column = SMD
    n = sim.size(0)
    diag = sim.diag()
    # rank of the matched item among all candidates (1 = best)
    rank_g2s = (sim > diag.unsqueeze(1)).sum(1) + 1
    rank_s2g = (sim > diag.unsqueeze(0)).sum(0) + 1
    off = sim.clone()
    off.fill_diagonal_(float("-inf"))
    hardest_neg = off.max(dim=1).values
    off_mean = sim.sum() - diag.sum()
    off_mean = (off_mean / (n * n - n)).item() if n > 1 else float("nan")

    def recalls(rank):
        return {
            "r1": (rank <= 1).float().mean().item(),
            "r5": (rank <= 5).float().mean().item(),
            "r10": (rank <= 10).float().mean().item(),
            "median_rank": rank.float().median().item(),
        }

    metrics = {
        "n": n,
        "chance_r1": 1.0 / n,
        "gt2smd": recalls(rank_g2s),
        "smd2gt": recalls(rank_s2g),
        "matched_cos_mean": diag.mean().item(),
        "mismatched_cos_mean": off_mean,
        "cos_gap": diag.mean().item() - off_mean,
    }
    per_sample = {
        "matched_cos": diag,
        "rank_gt2smd": rank_g2s,
        "rank_smd2gt": rank_s2g,
        "margin": diag - hardest_neg,
    }
    return metrics, per_sample


def evaluate_split(model, pairs, args, split_name, write_csv=False):
    z_gt, z_smd = encode_split(model, pairs, args)
    metrics, per_sample = retrieval_metrics(z_gt, z_smd)
    print(f"[{split_name}] N={metrics['n']} (chance R@1="
          f"{metrics['chance_r1']:.4f})")
    for direction in ("gt2smd", "smd2gt"):
        m = metrics[direction]
        print(f"[{split_name}] {direction}: R@1={m['r1']:.4f} "
              f"R@5={m['r5']:.4f} R@10={m['r10']:.4f} "
              f"medR={m['median_rank']:.0f}")
    print(f"[{split_name}] matched cos={metrics['matched_cos_mean']:.4f} "
          f"mismatched cos={metrics['mismatched_cos_mean']:.4f} "
          f"gap={metrics['cos_gap']:.4f}")

    if write_csv and args.output_dir:
        out = Path(args.output_dir)
        with (out / f"results_{split_name}.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        rows = sorted(
            range(len(pairs)), key=lambda i: per_sample["margin"][i].item()
        )
        csv_path = out / f"per_sample_scores_{split_name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "matched_cos", "margin",
                             "rank_gt2smd", "rank_smd2gt", "gt_snippet"])
            for i in rows:
                writer.writerow([
                    pairs[i]["name"],
                    f"{per_sample['matched_cos'][i].item():.4f}",
                    f"{per_sample['margin'][i].item():.4f}",
                    int(per_sample["rank_gt2smd"][i].item()),
                    int(per_sample["rank_smd2gt"][i].item()),
                    pairs[i]["gt"][:80],
                ])
        print(f"[{split_name}] per-clip scores (worst margin first) -> "
              f"{csv_path}")
        worst = [pairs[i]["name"] for i in rows[:5]]
        best = [pairs[i]["name"] for i in rows[-5:]]
        print(f"[{split_name}] lowest-margin clips (SMD likely "
              f"wrong/generic): {worst}")
        print(f"[{split_name}] highest-margin clips: {best}")
    return metrics


# ---------------------------------------------------------------- train


def train(args, model, train_pairs, val_sets):
    head_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and not n_.startswith("encoder.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if not args.freeze_encoder:
        groups.append({
            "params": [p for p in model.encoder.parameters()
                       if p.requires_grad],
            "lr": args.lr_encoder,
        })
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(
        len(train_pairs) / args.batch_size / args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    best_r1 = -1.0
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        for step, start in enumerate(range(0, len(order), args.batch_size)):
            batch = [train_pairs[i] for i in order[start:start + args.batch_size]]
            if len(batch) < 2:
                continue  # InfoNCE needs at least one negative
            with torch.autocast(
                device_type=args.device.split(":")[0],
                dtype=torch.bfloat16,
                enabled=args.bf16,
            ):
                loss = model([p["gt"] for p in batch],
                             [p["smd"] for p in batch])
            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f} "
                      f"scale={model.logit_scale.exp().item():.1f}")

        # retrieval is scored inside each dataset's own val pool (a mixed
        # pool would reward corpus-style cues, not GT<->SMD alignment)
        val_metrics = {}
        val_r1s = []
        for tag, val_pairs in val_sets:
            m = evaluate_split(model, val_pairs, args, f"val_{tag}")
            val_metrics[tag] = m
            val_r1s.append((m["gt2smd"]["r1"] + m["smd2gt"]["r1"]) / 2)
        val_r1 = sum(val_r1s) / len(val_r1s)
        trainable = {k: v.detach().cpu()
                     for k, v in model.named_parameters()
                     if v.requires_grad}
        torch.save({"model": trainable, "epoch": epoch, "val_r1": val_r1},
                   output_dir / "last_checkpoint.pth")
        if val_r1 > best_r1:
            best_r1 = val_r1
            torch.save({"model": trainable, "epoch": epoch,
                        "val_r1": val_r1},
                       output_dir / "best_checkpoint.pth")
        print(f"epoch {epoch}: train loss={epoch_loss / max(1, n_batches):.4f} "
              f"val avg R@1={val_r1:.4f} (best {best_r1:.4f})")
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, n_batches),
                "val": val_metrics,
            }) + "\n")


# ---------------------------------------------------------------- main


def main(args):
    utils.set_seed(args.seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = TextContrastiveProbe(args).to(args.device)
    n_train_params = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
    print(f"trainable parameters: {n_train_params / 1e6:.1f}M "
          f"(encoder {'frozen' if args.freeze_encoder else 'fine-tuned'})")

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")["model"]
        ret = model.load_state_dict(state, strict=False)
        print(f"loaded {args.resume}: {len(state)} tensors, "
              f"{len(ret.missing_keys)} missing (frozen/untrained), "
              f"{len(ret.unexpected_keys)} unexpected")
    elif args.eval and args.head != "none":
        print("WARNING: --eval without --resume scores a RANDOMLY "
              "initialized head; use --head none for the zero-shot floor.")

    datasets = parse_datasets(args.dataset)
    if len(datasets) > 1 and "{}" not in args.signsmd_text_root:
        raise ValueError(
            "multiple datasets need a per-dataset --signsmd_text_root "
            "template containing {}, e.g. data/{}/signsmd2d_stats_v0_1/texts"
        )

    if args.eval:
        for ds in datasets:
            root = resolve_text_root(args, ds)
            test_pairs = load_pairs(
                test_label_paths[ds], root, args, ds, "test",
            )
            if not test_pairs:
                raise ValueError(
                    f"0 (GT, SMD) pairs for {ds}/test — are the SMD texts "
                    f"generated under {root}?"
                )
            evaluate_split(model, test_pairs, args, f"test_{ds}",
                           write_csv=True)
        return

    train_pairs, val_sets = [], []
    for ds in datasets:
        root = resolve_text_root(args, ds)
        ds_train = load_pairs(
            train_label_paths[ds], root, args, ds, "train",
        )
        if not ds_train:
            raise ValueError(
                f"0 (GT, SMD) pairs for {ds}/train — are the SMD texts "
                f"generated under {root}?"
            )
        dev_path = dev_label_paths.get(ds, "")
        ds_val = load_pairs(dev_path, root, args, ds, "dev") \
            if dev_path else []
        if ds_val:
            print(f"[{ds}] selection on the dev split "
                  f"({len(ds_val)} pairs)")
        else:
            if dev_path:
                print(f"WARNING: [{ds}] dev split has 0 pairs (SMD texts "
                      f"not generated for dev under {root}?) — falling "
                      f"back to a held-out train slice")
            ds_train, ds_val = split_train_val(
                ds_train, args.val_fraction, args.seed
            )
            print(f"[{ds}] held-out train slice for selection, "
                  f"train/val = {len(ds_train)}/{len(ds_val)}")
        train_pairs += ds_train
        val_sets.append((ds, ds_val))
    print(f"pooled training pairs: {len(train_pairs)} "
          f"from {'+'.join(datasets)}")
    train(args, model, train_pairs, val_sets)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(get_args_parser().parse_args())
