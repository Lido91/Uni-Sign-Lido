"""Real-checkpoint CPU generation/DTW smoke and independent upstream parity.

One OpenASL test sample, 40 frames, MAR-2/DDIM-4. Not a full test score.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets_pose2d_klvae import S2T_Dataset_Pose2D_KLVAE
from diffusion_gen.factory import create_gaussian_diffusion, create_gaussian_diffusion_ddim
from diffusion_gen.resample import UniformSampler
from models_unified import Uni_Sign_Unified
from pose2d_metrics import PART_SLICES, pack_pose_parts
from train_unified import (VAE_OWNED_PREFIXES, evaluate_slg_generation,
                           get_unified_args_parser, load_partial, resolve_validation_split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', default='/tmp/unisign-dtw-smoke.json')
    opts = parser.parse_args()
    torch.set_num_threads(4)
    root = '/data/hwu/momadiff_repo/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d'
    stats = '/data/hwu/Uni_sign_repo/out/latent_stats'
    args = get_unified_args_parser().parse_args([
        '--dataset', 'OpenASL', '--task', 'SLT', '--dtype', 'fp32',
        '--max_length', '40', '--t2p_max_length', '40', '--num_workers', '0',
        '--ddim_steps', '4', '--trans_infer_timesteps', '2',
        '--vae_checkpoint', root + '/net_best_hand.pth',
        '--vae_mean_path', root + '/meta/mean.npy', '--vae_std_path', root + '/meta/std.npy',
        '--latent_mean_path', stats + '/latent_mean.npy', '--latent_std_path', stats + '/latent_std.npy',
    ])
    model = Uni_Sign_Unified(args)
    load_partial(model, opts.checkpoint, 'DTW smoke', VAE_OWNED_PREFIXES)
    model.attach_diffusion(create_gaussian_diffusion(), create_gaussian_diffusion_ddim(ddim_steps=4), None)
    model.schedule_sampler = UniformSampler(model.train_diffusion)
    phase, path = resolve_validation_split(args)
    data = S2T_Dataset_Pose2D_KLVAE(path, args, phase)
    loader = DataLoader(Subset(data, [0]), batch_size=1, collate_fn=data.collate_fn)

    state = torch.random.get_rng_state().clone()
    result = evaluate_slg_generation(args, loader, model)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.training  # evaluate_slg_generation restores the prior mode
    assert result['slg_num_samples'] == 1

    # Independent MoMADiff double-loop implementation, read without importing
    # its training dependencies or modifying the external repository.
    upstream_path = Path('/home/student/hwu/Workplace/MoMADiff/utils/pose2d_metrics.py')
    tree = ast.parse(upstream_path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_masked_dtw_cost')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(upstream_path), 'exec'), namespace)
    oracle = namespace['_masked_dtw_cost']

    model.eval()
    torch.manual_seed(args.seed)
    src, tgt = next(iter(loader))
    parts, lengths = model.generate_pose(tgt['gt_sentence'], src['src_length_batch'],
                                        timesteps=2, cond_scale=3, clip_coordinates=False)
    pred, _ = pack_pose_parts(parts)
    ref, conf = pack_pose_parts(src)
    pred = pred.detach().float().reshape(1, -1, 69, 2).numpy()[0, :int(lengths[0])]
    ref = ref.float().reshape(1, -1, 69, 2).numpy()[0, :int(src['src_length_batch'][0])]
    conf = conf.numpy()[0, :len(ref)] > 0
    expected = {}
    for part, sl in PART_SLICES.items():
        value = oracle(pred[:, sl], ref[:, sl], conf[:, sl])
        if value is not None:
            expected[f'openasl_DTW_MPJPE2D_{part}'] = value
    expected['openasl_DTW_MPJPE2D_hand'] = (expected['openasl_DTW_MPJPE2D_lhand'] + expected['openasl_DTW_MPJPE2D_rhand']) / 2
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, rtol=1e-6, atol=1e-6)
    report = {'checkpoint': opts.checkpoint, 'sample': src['name_batch'], 'device': 'cpu',
              'scope': 'one real OpenASL test clip, 40 frames, MAR-2/DDIM-4',
              'status': 'passed', 'rng_restored': True, 'upstream_parity': True, 'metrics': result}
    Path(opts.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
