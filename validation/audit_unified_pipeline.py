"""Short, real-checkpoint CPU audit of the Janus-inspired sign adaptation.

This checks execution/gradient contracts, not generation quality or a full
distributed training run. It never writes a model checkpoint or starts training.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import dev_label_paths, test_label_paths
from datasets_pose2d_klvae import S2T_Dataset_Pose2D_KLVAE
from diffusion_gen.factory import create_gaussian_diffusion, create_gaussian_diffusion_ddim
from diffusion_gen.resample import UniformSampler
from mask_generation import MaskGeneration, cosine_schedule, lengths_to_mask
from models_unified import POSE2D_PART_ORDER, Uni_Sign_Unified
from train_unified import (
    VAE_OWNED_PREFIXES, apply_stage_freezing, get_unified_args_parser, load_partial,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['OpenASL', 'How2Sign'], default='OpenASL')
    parser.add_argument('--report', default='/tmp/unisign-janus-contract-audit.json')
    parser.add_argument('--threads', type=int, default=4)
    opts = parser.parse_args()
    torch.set_num_threads(opts.threads)
    torch.manual_seed(42)
    root = '/data/hwu/momadiff_repo/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d'
    stats = '/data/hwu/Uni_sign_repo/out/latent_stats'
    args = get_unified_args_parser().parse_args([
        '--dataset', opts.dataset, '--task', 'SLT', '--dtype', 'fp32',
        '--max_length', '40', '--num_workers', '0',
        '--vae_checkpoint', root + '/net_best_hand.pth',
        '--vae_mean_path', root + '/meta/mean.npy',
        '--vae_std_path', root + '/meta/std.npy',
        '--latent_mean_path', stats + '/latent_mean.npy',
        '--latent_std_path', stats + '/latent_std.npy',
    ])
    model = Uni_Sign_Unified(args)
    load_partial(model, opts.checkpoint, 'contract audit', VAE_OWNED_PREFIXES)
    model.attach_diffusion(create_gaussian_diffusion(),
                           create_gaussian_diffusion_ddim(ddim_steps=4), None)
    model.schedule_sampler = UniformSampler(model.train_diffusion)
    model.eval()
    phase = 'dev' if dev_label_paths[opts.dataset] else 'test'
    data = S2T_Dataset_Pose2D_KLVAE(
        (dev_label_paths if phase == 'dev' else test_label_paths)[opts.dataset], args, phase,
    )
    sample = data[0]
    src, tgt = data.collate_fn([sample])
    for part in POSE2D_PART_ORDER:
        src[part] = src[part].float()
    report = {'checkpoint': opts.checkpoint, 'dataset': opts.dataset,
              'device': 'cpu', 'sample': src['name_batch'], 'checks': {}}
    checks = report['checks']

    # Check the real coordinate round-trip without clipping valid data.
    with torch.no_grad():
        motion = model.parts_to_motion(src)
        raw = motion * model.motion_std + model.motion_mean
        original = torch.cat([src[p][..., :2].flatten(2) for p in POSE2D_PART_ORDER], -1)
        torch.testing.assert_close(raw, original, atol=1e-6, rtol=1e-5)
        checks['representation_round_trip_max_error'] = (raw - original).abs().max().item()

        # True-length VAE encoding must not depend on a longer padded neighbor.
        short = (sample[0], {p: sample[1][p][:28] for p in sample[1]}, *sample[2:])
        short_src, _ = data.collate_fn([short])
        mixed_src, _ = data.collate_fn([short, sample])
        for input_ in (short_src, mixed_src):
            for part in POSE2D_PART_ORDER:
                input_[part] = input_[part].float()
        alone, _ = model.encode_gt_latents(short_src, sample=False)
        together, lens = model.encode_gt_latents(mixed_src, sample=False)
        torch.testing.assert_close(alone[0], together[0, :7], atol=1e-5, rtol=1e-4)
        checks['vae_true_length_padding_invariance'] = True

        # Shrinking masks cover each valid slot once; padding is never sampled.
        valid = lengths_to_mask(lens, 10)
        masks = list(MaskGeneration(lens, 10, 9, cosine_schedule,
                                    device='cpu', drop_last=False))
        assert torch.equal(masks[0], valid) and not masks[-1].any()
        reveals = torch.zeros_like(valid, dtype=torch.long)
        for previous, current in zip(masks[:-1], masks[1:]):
            assert not (current & ~previous).any()
            reveals += (previous & ~current).long()
        assert torch.equal(reveals, valid.long())
        checks['mar_masks_cover_valid_positions_exactly_once'] = True
        checks['mar_states_and_transitions'] = [len(masks), len(masks) - 1]

        # DDIM must feed original training timesteps to the denoiser.
        observed = []
        def spy(x, t, **kwargs):
            observed.append(t.tolist())
            return torch.zeros_like(x)
        model.eval_diffusion.ddim_sample_loop(
            spy, (1, 512), device='cpu', clip_denoised=False,
            model_kwargs={'y': {}}, eta=0.0,
        )
        assert [row[0] for row in observed] == list(reversed(model.eval_diffusion.timestep_map))
        checks['ddim_original_timestep_mapping'] = [row[0] for row in observed]
        output, gen_lens = model.generate_pose(tgt['gt_sentence'], src['src_length_batch'], timesteps=2)
        assert gen_lens.tolist() == [40]
        assert all(torch.isfinite(output[p]).all() for p in POSE2D_PART_ORDER)
        checks['generation_smoke'] = {'mar_states': 2, 'ddim_steps': 4,
            'lengths': gen_lens.tolist(), 'shapes': {p: list(output[p].shape) for p in POSE2D_PART_ORDER}}

    # Stage 1 must propagate through the frozen encoder into new adapters.
    apply_stage_freezing(model, 1)
    model.train()
    frozen_probe = model.mt5_model.shared.weight[:1].detach().clone()
    head_probe = next(model.diff_head.parameters()).detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    out = model(src, tgt, task='t2p')
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    gradient_norms = {}
    for name in ('latent_in_proj', 'cond_out_head', 'diff_head'):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        gradient_norms[name] = sum(g.float().norm().item() for g in grads)
    assert model.mask_emb.grad is not None and torch.isfinite(model.mask_emb.grad).all()
    assert model.mask_emb.grad.norm().item() > 0
    assert all(p.grad is None for p in model.mt5_model.parameters())
    assert all(p.grad is None for p in model.vae_encoder.parameters())
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    assert not torch.equal(head_probe, next(model.diff_head.parameters()).detach())
    torch.testing.assert_close(frozen_probe, model.mt5_model.shared.weight[:1], rtol=0, atol=0)
    checks['stage1_real_optimizer_step'] = {'loss': out['loss'].item(),
        'gradient_norms': gradient_norms, 'frozen_mt5_unchanged': True}
    model.zero_grad(set_to_none=True)
    del optimizer, out
    print('Stage 1 real backward/update passed', flush=True)

    # Shared encoder receives gradients from the joint task in Stage 2.
    apply_stage_freezing(model, 2)
    model.eval()  # no dropout; this remains a valid gradient check
    torch.manual_seed(99)
    joint = model(src, tgt, task='joint', t2p_src_input=src, t2p_tgt_input=tgt)
    assert torch.isfinite(joint['loss'])
    torch.testing.assert_close(joint['loss'].detach(), joint['slt_loss'] + joint['t2p_loss'])
    joint['loss'].backward()
    shared_grad = model.mt5_model.shared.weight.grad
    assert shared_grad is not None and torch.isfinite(shared_grad).all() and shared_grad.norm() > 0
    assert all(p.grad is None for p in model.gcn_modules.parameters())
    checks['stage2_real_joint_backward'] = {'slt_loss': joint['slt_loss'].item(),
        't2p_loss': joint['t2p_loss'].item(), 'shared_embedding_gradient_norm': shared_grad.norm().item()}
    model.zero_grad(set_to_none=True)
    del joint, shared_grad

    for stage in (1, 2, 3):
        apply_stage_freezing(model, stage)
        model.train()
        assert not model.vae_encoder.training and not model.vae_decoder.training
        assert not any(p.requires_grad for p in model.vae_encoder.parameters())
        assert not any(p.requires_grad for p in model.vae_decoder.parameters())
        assert any(p.requires_grad for p in model.mt5_model.parameters()) == (stage >= 2)
        assert any(p.requires_grad for p in model.gcn_modules.parameters()) == (stage == 3)
        for name in model.frozen_eval_modules:
            assert not getattr(model, name).training
        checks[f'stage{stage}_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['status'] = 'passed'
    report['limitations'] = ['one real clip; CPU only', 'short sampling schedule',
        'no full distributed run', 'no claim of generation quality or equivalence to JanusFlow']
    path = Path(opts.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
