"""Back-translation + MPJPE2D evaluation for the unified T2P path.

A frozen, trained ST-GCN Uni-Sign SLT checkpoint acts as the judge: pose
sequences are fed to it and its BLEU against the conditioning text measures
semantic quality. Four modes build the evaluation ladder:

  gt        GT poses as-is                      -> judge BLEU ceiling
  gt_conf1  GT coords, conf channel forced to 1 -> cost of the conf=1
            mismatch (generated poses carry no confidence)
  vae_recon frozen KL-VAE encode->decode of GT  -> generation quality
            ceiling (plus MPJPE2D vs GT)
  gen       text -> Uni_Sign_Unified.generate_pose -> back-translation BLEU
            (plus MPJPE2D vs GT, GT-length oracle)

gt/gt_conf1/vae_recon need NO trained generation model — run them first to
validate the harness and the decoder port.

    python back_translation_eval.py --dataset How2Sign --split test \
        --mode vae_recon --slt_checkpoint <slt best_checkpoint.pth> \
        --vae_checkpoint <net_best_hand.pth> --output_dir out/backtrans_h2s

Single GPU, no DeepSpeed.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import utils
from config import dev_label_paths, test_label_paths
from datasets_pose2d_klvae import (
    FRAME_SAMPLING_MODES,
    S2T_Dataset_Pose2D_KLVAE,
)
from models import Uni_Sign
from models_unified import (
    POSE2D_FEATURE_DIM,
    POSE2D_PART_JOINTS,
    POSE2D_PART_ORDER,
    Uni_Sign_Unified,
)
from SLRT_metrics import translation_performance
from vae_decoder import load_frozen_klvae_decoder
from vae_encoder import DOWNSAMPLE_FACTOR, load_trainable_klvae_encoder

DEFAULT_VAE_META = (
    "/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/"
    "kl_vae_h2s_openasl_pose2d/meta"
)

MODES = ("gt", "gt_conf1", "vae_recon", "gen")


def get_backtrans_args_parser():
    parser = argparse.ArgumentParser(
        "Uni-Sign back-translation evaluation",
        parents=[utils.get_args_parser()],
    )
    parser.add_argument("--mode", required=True, choices=list(MODES))
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--slt_checkpoint", required=True,
                        help="Trained ST-GCN Uni_Sign checkpoint (judge).")
    parser.add_argument("--unified_checkpoint", default="",
                        help="Uni_Sign_Unified checkpoint (gen mode).")
    parser.add_argument("--vae_checkpoint", default="",
                        help="Required for vae_recon and gen modes.")
    parser.add_argument("--vae_mean_path", default=f"{DEFAULT_VAE_META}/mean.npy")
    parser.add_argument("--vae_std_path", default=f"{DEFAULT_VAE_META}/std.npy")
    parser.add_argument("--frame_sampling", default="vae_linear",
                        choices=list(FRAME_SAMPLING_MODES))
    parser.add_argument("--latent_mean_path", default="",
                        help="Must match the trained unified model "
                             "(gen mode).")
    parser.add_argument("--latent_std_path", default="")
    parser.add_argument("--backtrans_max_frames", type=int, default=256,
                        help="The judge never saw clips longer than this; "
                             "longer sequences are linearly resampled.")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    # gen-mode diffusion geometry (must match the trained unified model)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--diff_layers", type=int, default=4)
    parser.add_argument("--diff_ff_size", type=int, default=1024)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--trans_infer_timesteps", type=int, default=9)
    parser.add_argument("--guidance_param", type=float, default=3.0)
    return parser


def parts_to_motion(parts, mean, std):
    chunks = []
    for part in POSE2D_PART_ORDER:
        xy = parts[part][..., :2]
        chunks.append(xy.reshape(xy.shape[0], xy.shape[1], -1))
    motion = torch.cat(chunks, dim=-1)
    return (motion - mean.to(motion.dtype)) / std.to(motion.dtype)


def motion_to_parts(motion, conf_value=1.0):
    joints = motion.reshape(*motion.shape[:-1], POSE2D_FEATURE_DIM // 2, 2)
    joints = joints.clamp(-1.0, 1.0)
    conf = torch.full_like(joints[..., :1], conf_value)
    packed = torch.cat([joints, conf], dim=-1)
    parts, start = {}, 0
    for part in POSE2D_PART_ORDER:
        count = POSE2D_PART_JOINTS[part]
        parts[part] = packed[..., start:start + count, :]
        start += count
    return parts


def resample_batch_parts(parts, lengths, max_frames):
    """Per-sample deterministic linear resampling to <= max_frames frames."""
    lengths = [int(x) for x in lengths]
    new_lengths = [min(length, max_frames) for length in lengths]
    max_new = max(new_lengths)
    out = {}
    for part in POSE2D_PART_ORDER:
        source = parts[part]
        batch, _, joints, channels = source.shape
        resampled = source.new_zeros(batch, max_new, joints, channels)
        for i, (length, new_length) in enumerate(zip(lengths, new_lengths)):
            if length <= new_length:
                resampled[i, :length] = source[i, :length]
                if length < max_new:  # repeat-last padding, collate-style
                    resampled[i, length:] = source[i, length - 1]
            else:
                idx = torch.round(
                    torch.linspace(0, length - 1, new_length)
                ).long()
                resampled[i, :new_length] = source[i, idx]
                if new_length < max_new:
                    resampled[i, new_length:] = resampled[i, new_length - 1]
        out[part] = resampled
    return out, new_lengths


def build_judge_input(parts, lengths, names, device):
    lengths_tensor = torch.LongTensor(lengths)
    max_len = max(lengths)
    positions = torch.arange(max_len).unsqueeze(0)
    attention_mask = (positions < lengths_tensor.unsqueeze(1)).long()
    src_input = {
        part: parts[part][:, :max_len].to(device=device, dtype=torch.float32)
        for part in POSE2D_PART_ORDER
    }
    src_input["attention_mask"] = attention_mask.to(device)
    src_input["src_length_batch"] = lengths_tensor.to(device)
    src_input["name_batch"] = names
    return src_input


class MPJPEAccumulator:
    """Conf-masked mean L2 joint error in the normalized 2D space."""

    def __init__(self):
        self.sums = {part: 0.0 for part in POSE2D_PART_ORDER}
        self.counts = {part: 0 for part in POSE2D_PART_ORDER}

    def update(self, pred_parts, gt_parts, lengths):
        for part in POSE2D_PART_ORDER:
            pred = pred_parts[part][..., :2].float()
            gt = gt_parts[part][..., :2].float()
            conf = gt_parts[part][..., 2].float()
            frames = min(pred.shape[1], gt.shape[1])
            for i, length in enumerate(lengths):
                valid_t = min(int(length), frames)
                error = torch.linalg.norm(
                    pred[i, :valid_t] - gt[i, :valid_t], dim=-1
                )
                mask = conf[i, :valid_t] > 0
                self.sums[part] += float(error[mask].sum())
                self.counts[part] += int(mask.sum())

    def results(self):
        out = {}
        for part in POSE2D_PART_ORDER:
            out[f"mpjpe2d_{part}"] = (
                self.sums[part] / self.counts[part]
                if self.counts[part] else float("nan")
            )
        hand_sum = self.sums["left"] + self.sums["right"]
        hand_count = self.counts["left"] + self.counts["right"]
        out["mpjpe2d_hand"] = (
            hand_sum / hand_count if hand_count else float("nan")
        )
        total_sum = sum(self.sums.values())
        total_count = sum(self.counts.values())
        out["mpjpe2d_all"] = (
            total_sum / total_count if total_count else float("nan")
        )
        return out


@torch.no_grad()
def vae_reconstruct(parts, lengths, encoder, decoder, mean, std, device):
    """Frozen encode->decode of GT parts (posterior mean), grouped by the
    per-sample truncated length so padding never leaks through the convs."""
    motion = parts_to_motion(
        {p: parts[p].to(device) for p in POSE2D_PART_ORDER}, mean, std
    )
    unit = DOWNSAMPLE_FACTOR
    trunc = [max((int(x) // unit) * unit, unit) for x in lengths]
    groups = {}
    for index, length in enumerate(trunc):
        groups.setdefault(length, []).append(index)
    recon = motion.new_zeros(motion.shape[0], max(trunc), POSE2D_FEATURE_DIM)
    for length, indices in groups.items():
        chunk = motion[indices, :length]
        if chunk.shape[1] < length:
            pad = chunk[:, -1:].expand(-1, length - chunk.shape[1], -1)
            chunk = torch.cat([chunk, pad], dim=1)
        moments = encoder(chunk)
        z, _ = torch.chunk(moments, 2, dim=-1)  # posterior mean
        decoded = decoder(z)
        recon[indices, : decoded.shape[1]] = decoded.to(recon.dtype)
    recon = recon * std.to(recon.dtype) + mean.to(recon.dtype)
    return motion_to_parts(recon), trunc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    args.distributed = False

    if args.mode in ("vae_recon", "gen") and not args.vae_checkpoint:
        raise ValueError(f"--vae_checkpoint is required for mode {args.mode}")
    if args.mode == "gen" and not args.unified_checkpoint:
        raise ValueError("--unified_checkpoint is required for mode gen")
    if args.dataset == "How2Sign" and args.split == "dev":
        raise ValueError("How2Sign has no dev split")

    label_path = (dev_label_paths if args.split == "dev"
                  else test_label_paths)[args.dataset]
    dataset = S2T_Dataset_Pose2D_KLVAE(
        path=label_path, args=args, phase=args.split
    )
    print(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        sampler=torch.utils.data.SequentialSampler(dataset),
        pin_memory=args.pin_mem,
    )

    # ---- frozen SLT judge --------------------------------------------------
    print("Loading SLT judge:")
    judge = Uni_Sign(args=args)
    state = torch.load(args.slt_checkpoint, map_location="cpu")
    state = state.get("model", state)
    ret = judge.load_state_dict(state, strict=False)
    print(f"  judge missing={len(ret.missing_keys)} "
          f"unexpected={len(ret.unexpected_keys)}")
    if ret.missing_keys:
        raise ValueError(
            "--slt_checkpoint must be a complete trained ST-GCN Uni_Sign "
            f"checkpoint; missing e.g. {ret.missing_keys[:5]}"
        )
    judge.to(device).eval()
    for parameter in judge.parameters():
        parameter.requires_grad = False

    # ---- mode-specific machinery ------------------------------------------
    mean = std = vae_encoder = vae_decoder = unified = None
    if args.mode in ("vae_recon",):
        mean = torch.from_numpy(
            np.load(args.vae_mean_path).astype(np.float32)
        ).to(device)
        std = torch.from_numpy(
            np.load(args.vae_std_path).astype(np.float32)
        ).clamp(min=1e-10).to(device)
        vae_encoder = load_trainable_klvae_encoder(
            args.vae_checkpoint, input_emb_width=POSE2D_FEATURE_DIM
        ).to(device).eval()
        for parameter in vae_encoder.parameters():
            parameter.requires_grad = False
        vae_decoder = load_frozen_klvae_decoder(
            args.vae_checkpoint, input_emb_width=POSE2D_FEATURE_DIM
        ).to(device)
    elif args.mode == "gen":
        from diffusion_gen.factory import (
            create_gaussian_diffusion,
            create_gaussian_diffusion_ddim,
        )
        from diffusion_gen.resample import create_named_schedule_sampler

        args.default_task = "t2p"
        unified = Uni_Sign_Unified(args=args)
        unified.attach_diffusion(
            create_gaussian_diffusion(args.diffusion_steps, args.noise_schedule),
            create_gaussian_diffusion_ddim(
                args.diffusion_steps, args.ddim_steps, args.noise_schedule
            ),
            None,
        )
        from diffusion_gen.resample import UniformSampler
        unified.schedule_sampler = UniformSampler(unified.train_diffusion)
        state = torch.load(args.unified_checkpoint, map_location="cpu")
        state = state.get("model", state)
        ret = unified.load_state_dict(state, strict=False)
        print(f"  unified missing={len(ret.missing_keys)} "
              f"unexpected={len(ret.unexpected_keys)}")
        # only VAE-owned tensors may be absent — anything else means a wrong
        # checkpoint kind (e.g. plain SLT) and the generation head would stay
        # randomly initialized
        allowed = ("vae_encoder.", "vae_decoder.", "motion_mean",
                   "motion_std", "latent_mean", "latent_std")
        violations = [
            key for key in ret.missing_keys if not key.startswith(allowed)
        ]
        if violations:
            raise ValueError(
                "--unified_checkpoint is not a unified (SLT+T2P) checkpoint; "
                f"missing e.g. {violations[:5]}"
            )
        unified.to(device).eval()

    # ---- evaluation loop ---------------------------------------------------
    metric_logger = utils.MetricLogger(delimiter="  ")
    mpjpe = MPJPEAccumulator()
    predictions, references, names = [], [], []

    with torch.no_grad():
        for src_input, tgt_input in metric_logger.log_every(
            loader, 10, f"{args.mode}:"
        ):
            gt_parts = {p: src_input[p].float() for p in POSE2D_PART_ORDER}
            lengths = [int(x) for x in src_input["src_length_batch"]]
            batch_names = src_input["name_batch"]
            sentences = tgt_input["gt_sentence"]

            if args.mode == "gt":
                parts, part_lengths = gt_parts, lengths
            elif args.mode == "gt_conf1":
                parts = {}
                for part in POSE2D_PART_ORDER:
                    tensor = gt_parts[part].clone()
                    tensor[..., 2] = 1.0
                    parts[part] = tensor
                part_lengths = lengths
            elif args.mode == "vae_recon":
                parts, part_lengths = vae_reconstruct(
                    gt_parts, lengths, vae_encoder, vae_decoder,
                    mean, std, device,
                )
                mpjpe.update(
                    {p: parts[p].cpu() for p in POSE2D_PART_ORDER},
                    gt_parts, part_lengths,
                )
            else:  # gen
                frame_lens = torch.tensor(lengths)
                gen_parts, gen_lens = unified.generate_pose(
                    sentences,
                    frame_lens,
                    timesteps=args.trans_infer_timesteps,
                    cond_scale=args.guidance_param,
                )
                parts = gen_parts
                part_lengths = [int(x) for x in gen_lens]
                mpjpe.update(
                    {p: parts[p].cpu() for p in POSE2D_PART_ORDER},
                    gt_parts, part_lengths,
                )

            parts, part_lengths = resample_batch_parts(
                parts, part_lengths, args.backtrans_max_frames
            )
            judge_input = build_judge_input(
                parts, part_lengths, batch_names, device
            )
            stack_out = judge(judge_input, tgt_input)
            metric_logger.update(loss=stack_out["loss"].item())
            output = judge.generate(
                stack_out,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions.extend(output)
            references.extend(sentences)
            names.extend(batch_names)

    tokenizer = judge.text_tokenizer
    prediction_ids = pad_sequence(
        predictions, batch_first=True, padding_value=tokenizer.eos_token_id
    )
    decoded = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
    bleu, rouge = translation_performance(references, decoded)

    results = {
        "mode": args.mode,
        "dataset": args.dataset,
        "split": args.split,
        **bleu,
        "rouge": rouge,
        "judge_loss": metric_logger.meters["loss"].global_avg,
    }
    if args.mode in ("vae_recon", "gen"):
        results.update(mpjpe.results())
    print(json.dumps(results, indent=2))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{args.mode}_{args.dataset}_{args.split}"
        with (output_dir / f"{tag}_results.json").open("w") as f:
            json.dump(results, f, indent=2)
        with (output_dir / f"{tag}_pres.txt").open("w") as f:
            for name, prediction in zip(names, decoded):
                f.write(f"sample: {name}, prediction: {prediction}\n")
        with (output_dir / f"{tag}_refs.txt").open("w") as f:
            for name, reference in zip(names, references):
                f.write(f"sample: {name}, ground-truth: {reference}\n")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parsed_args = get_backtrans_args_parser().parse_args()
    if parsed_args.rgb_support:
        raise ValueError("--rgb_support is not supported here")
    main(parsed_args)
