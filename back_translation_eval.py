"""Back-translation + MPJPE2D + SOKE/MoMADiff DTW-MPJPE2D evaluation.

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
  pred_dir  externally generated poses (MoMADiff eval_t2m.py --save_predictions
            dump: one {SENTENCE_NAME}.pkl per sample with feats_rst/feats_ref/
            conf_ref/text in the VAE-normalized 138D space) -> back-translation
            BLEU (plus MPJPE2D vs the dumped GT)

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
import pickle
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
from pose2d_metrics import DTWMPJPE2DMetrics
from pose2d_predictions import save_pose_prediction_batch
from vae_decoder import load_frozen_klvae_decoder
from vae_encoder import DOWNSAMPLE_FACTOR, load_trainable_klvae_encoder

DEFAULT_VAE_META = (
    "/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/"
    "kl_vae_h2s_openasl_pose2d/meta"
)

MODES = ("gt", "gt_conf1", "vae_recon", "gen", "pred_dir")


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
    parser.add_argument("--pred_dir", default="",
                        help="Directory of external test predictions (mode "
                             "pred_dir): one {SENTENCE_NAME}.pkl per sample "
                             "as dumped by MoMADiff eval_t2m.py with "
                             "save_predictions. Comma-separate several dirs "
                             "to merge per-rank DDP dumps. The feats are "
                             "denormalized with --vae_mean_path/--vae_std_path"
                             ", which must match the meta stats the dump was "
                             "produced with.")
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
    parser.add_argument("--save_predictions", "--save-predictions",
                        action="store_true",
                        help="Save unclipped generated/reconstructed pose, GT, "
                             "confidence, caption and back-translation as "
                             "pred_dir-compatible per-sample .pkl files.")
    return parser


def parts_to_motion(parts, mean, std):
    chunks = []
    for part in POSE2D_PART_ORDER:
        xy = parts[part][..., :2]
        chunks.append(xy.reshape(xy.shape[0], xy.shape[1], -1))
    motion = torch.cat(chunks, dim=-1)
    return (motion - mean.to(motion.dtype)) / std.to(motion.dtype)


def motion_to_parts(motion, conf_value=1.0, clip_coordinates=True):
    joints = motion.reshape(*motion.shape[:-1], POSE2D_FEATURE_DIM // 2, 2)
    if clip_coordinates:
        joints = joints.clamp(-1.0, 1.0)
    conf = torch.full_like(joints[..., :1], conf_value)
    packed = torch.cat([joints, conf], dim=-1)
    parts, start = {}, 0
    for part in POSE2D_PART_ORDER:
        count = POSE2D_PART_JOINTS[part]
        parts[part] = packed[..., start:start + count, :]
        start += count
    return parts


def clamp_judge_parts(parts):
    """Clip only the SLT input; motion metrics use native decoded coordinates."""
    result = {}
    for part, value in parts.items():
        packed = value.clone()
        packed[..., :2] = packed[..., :2].clamp(-1.0, 1.0)
        result[part] = packed
    return result


def conf_to_parts(conf):
    """Split a [B, T, 69] per-joint confidence tensor by part."""
    parts, start = {}, 0
    for part in POSE2D_PART_ORDER:
        count = POSE2D_PART_JOINTS[part]
        parts[part] = conf[..., start:start + count]
        start += count
    return parts


def pad_stack(seqs, zero_pad=False):
    """Stack variable-length [T, ...] tensors, repeat-last (or zero) padded."""
    max_len = max(seq.shape[0] for seq in seqs)
    padded = []
    for seq in seqs:
        if seq.shape[0] < max_len:
            fill = torch.zeros_like(seq[-1:]) if zero_pad else seq[-1:]
            fill = fill.expand(max_len - seq.shape[0], *seq.shape[1:])
            seq = torch.cat([seq, fill], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def list_prediction_files(pred_dir):
    files = []
    for directory in pred_dir.split(","):
        directory = directory.strip()
        if not directory:
            continue
        entries = sorted(os.listdir(directory))
        files.extend(
            os.path.join(directory, entry)
            for entry in entries if entry.endswith(".pkl")
        )
    if not files:
        raise ValueError(f"no .pkl predictions found under {pred_dir}")
    return files


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
    return motion_to_parts(recon, clip_coordinates=False), trunc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    args.distributed = False

    if args.mode in ("vae_recon", "gen") and not args.vae_checkpoint:
        raise ValueError(f"--vae_checkpoint is required for mode {args.mode}")
    if args.mode == "gen" and not args.unified_checkpoint:
        raise ValueError("--unified_checkpoint is required for mode gen")
    if args.mode == "pred_dir" and not args.pred_dir:
        raise ValueError("--pred_dir is required for mode pred_dir")
    if args.dataset == "How2Sign" and args.split == "dev":
        raise ValueError("How2Sign has no dev split")
    save_predictions = getattr(args, "save_predictions", False)
    if save_predictions and (not args.output_dir
                             or args.mode not in ("gen", "vae_recon")):
        raise ValueError("--save_predictions needs --output_dir and mode gen/vae_recon")

    if args.mode == "pred_dir":
        # The dump is self-contained (pred + GT + conf + caption); no label
        # file or dataset loader is needed.
        prediction_files = list_prediction_files(args.pred_dir)
        print(f"pred_dir: {len(prediction_files)} prediction files")
        loader = None
    else:
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
    if args.mode in ("vae_recon", "pred_dir"):
        mean = torch.from_numpy(
            np.load(args.vae_mean_path).astype(np.float32)
        ).to(device)
        std = torch.from_numpy(
            np.load(args.vae_std_path).astype(np.float32)
        ).clamp(min=1e-10).to(device)
    if args.mode == "vae_recon":
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
    dtw = DTWMPJPE2DMetrics()
    predictions, references, names = [], [], []
    prediction_dir = (Path(args.output_dir)
                      / f"{args.mode}_{args.dataset}_{args.split}_predictions")
    saved_prediction_count = 0
    saved_names = set()

    if args.mode == "pred_dir":
        batches = [
            prediction_files[i:i + args.batch_size]
            for i in range(0, len(prediction_files), args.batch_size)
        ]
        with torch.no_grad():
            for paths in metric_logger.log_every(batches, 10, "pred_dir:"):
                payloads = []
                for path in paths:
                    with open(path, "rb") as handle:
                        payloads.append(pickle.load(handle))
                batch_names = [str(payload.get("name", Path(path).stem))
                               for path, payload in zip(paths, payloads)]
                sentences = [str(payload["text"]) for payload in payloads]
                rst = [
                    torch.from_numpy(
                        np.asarray(payload["feats_rst"], dtype=np.float32)
                    )
                    for payload in payloads
                ]
                ref = [
                    torch.from_numpy(
                        np.asarray(payload["feats_ref"], dtype=np.float32)
                    )
                    for payload in payloads
                ]
                conf = [
                    torch.from_numpy(
                        np.asarray(payload["conf_ref"], dtype=np.float32)
                    )
                    for payload in payloads
                ]
                lengths_rst = [int(seq.shape[0]) for seq in rst]
                lengths_ref = [int(seq.shape[0]) for seq in ref]

                motion_rst = pad_stack(rst).to(device) * std + mean
                motion_ref = pad_stack(ref).to(device) * std + mean
                # Generated poses carry no confidence -> conf=1 (the gt_conf1
                # ceiling measures the cost of exactly this mismatch).
                parts = motion_to_parts(motion_rst, clip_coordinates=False)
                gt_parts = motion_to_parts(motion_ref, clip_coordinates=False)
                conf_parts = conf_to_parts(
                    pad_stack(conf, zero_pad=True).to(device)
                )
                for part in POSE2D_PART_ORDER:
                    gt_parts[part][..., 2] = conf_parts[part]

                mpjpe.update(
                    {p: parts[p].cpu() for p in POSE2D_PART_ORDER},
                    {p: gt_parts[p].cpu() for p in POSE2D_PART_ORDER},
                    [min(a, b) for a, b in zip(lengths_rst, lengths_ref)],
                )
                dtw.update_parts(parts, gt_parts, lengths_ref, lengths_rst,
                                 args.dataset, batch_names)

                parts, part_lengths = resample_batch_parts(
                    clamp_judge_parts(parts), lengths_rst, args.backtrans_max_frames
                )
                judge_input = build_judge_input(
                    parts, part_lengths, batch_names, device
                )
                tgt_input = {
                    "gt_sentence": sentences,
                    "gt_gloss": [""] * len(sentences),
                }
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
    else:
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
                    dtw.update_parts(parts, gt_parts, lengths, part_lengths,
                                     args.dataset, batch_names)
                else:  # gen
                    frame_lens = torch.tensor(lengths)
                    gen_parts, gen_lens = unified.generate_pose(
                        sentences,
                        frame_lens,
                        timesteps=args.trans_infer_timesteps,
                        cond_scale=args.guidance_param,
                        clip_coordinates=False,
                    )
                    parts = gen_parts
                    part_lengths = [int(x) for x in gen_lens]
                    mpjpe.update(
                        {p: parts[p].cpu() for p in POSE2D_PART_ORDER},
                        gt_parts, part_lengths,
                    )
                    dtw.update_parts(parts, gt_parts, lengths, part_lengths,
                                     args.dataset, batch_names)

                native_parts, native_lengths = parts, list(part_lengths)
                parts, part_lengths = resample_batch_parts(
                    clamp_judge_parts(parts), part_lengths, args.backtrans_max_frames
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
                if save_predictions:
                    if any(str(name) in saved_names for name in batch_names):
                        raise ValueError("duplicate sample ID while saving predictions")
                    motion_mean = unified.motion_mean if unified is not None else mean
                    motion_std = unified.motion_std if unified is not None else std
                    saved_paths = save_pose_prediction_batch(
                        prediction_dir, native_parts, gt_parts,
                        native_lengths, lengths, batch_names, sentences,
                        motion_mean, motion_std, POSE2D_PART_ORDER,
                        POSE2D_PART_JOINTS,
                        back_translations=judge.text_tokenizer.batch_decode(
                            output, skip_special_tokens=True
                        ),
                        metadata={
                            "mode": args.mode, "dataset": args.dataset,
                            "split": args.split, "seed": args.seed,
                            "frame_sampling": args.frame_sampling,
                            "max_length": args.max_length,
                            "slt_checkpoint": args.slt_checkpoint,
                            "unified_checkpoint": args.unified_checkpoint,
                            "vae_checkpoint": args.vae_checkpoint,
                            "latent_mean_path": args.latent_mean_path,
                            "latent_std_path": args.latent_std_path,
                            "ddim_steps": args.ddim_steps,
                            "mar_states": args.trans_infer_timesteps,
                            "guidance_param": args.guidance_param,
                        },
                    )
                    saved_prediction_count += len(saved_paths)
                    saved_names.update(str(name) for name in batch_names)
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
        "num_samples": len(names),
        "seed": args.seed,
        "slt_checkpoint": args.slt_checkpoint,
        **bleu,
        "rouge": rouge,
        "judge_loss": metric_logger.meters["loss"].global_avg,
    }
    if args.mode == "pred_dir":
        results["pred_dir"] = args.pred_dir
    if args.mode == "gen":
        results["unified_checkpoint"] = args.unified_checkpoint
        results["gt_length_conditioned"] = True
        results["generation_settings"] = {
            "ddim_steps": args.ddim_steps,
            "mar_states": args.trans_infer_timesteps,
            "guidance_param": args.guidance_param,
            "frame_sampling": args.frame_sampling,
            "max_length": args.max_length,
            "backtrans_max_frames": args.backtrans_max_frames,
        }
    if save_predictions:
        if saved_prediction_count != len(names):
            raise RuntimeError("saved pose count does not match evaluated sample count")
        results["prediction_dir"] = str(prediction_dir)
        results["saved_prediction_count"] = saved_prediction_count
    if args.mode in ("vae_recon", "gen", "pred_dir"):
        results.update(mpjpe.results())
        results.update(dtw.results())
        results['dtw_protocol'] = 'MoMADiff/SOKE: accumulated sequence cost, independent part alignment, sequence mean, hand=(lhand+rhand)/2'
        results['dtw_valid_sequence_counts'] = dict(dtw.counts)
    print(json.dumps(results, indent=2))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{args.mode}_{args.dataset}_{args.split}"
        with (output_dir / f"{tag}_results.json").open("w") as f:
            json.dump(results, f, indent=2)
        if dtw.name2scores:
            with (output_dir / f"{tag}_dtw_per_sample.json").open("w") as f:
                json.dump(dtw.name2scores, f, indent=2)
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
