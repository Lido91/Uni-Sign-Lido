"""Small real CUDA/DDP/DeepSpeed test; does not load production checkpoints.

Uses a tiny mT5 and linear pose fixture to exercise the exact Geo-Sign forward,
the actual unified joint training loop, both optimizers and collective resume.
The production VAE/diffusion and full-data metrics are covered separately.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
import utils
from models_unified import Uni_Sign_Unified
from train_unified import (get_unified_args_parser, apply_stage_freezing,
                           train_joint_epoch)
from geosign_training import (create_geosign_optimizers, patch_geosign_engine,
                              restore_geosign_optimizer)
from test_geosign_unified import make_fixture, fixture_batch, tiny_t2p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    root = Path(options.output)
    root.mkdir(parents=True, exist_ok=True)
    args = get_unified_args_parser().parse_args([
        '--vae_checkpoint', 'fixture-unused', '--dataset', 'OpenASL',
        '--dtype', 'fp32', '--zero_stage', '0', '--train_stage', '2',
        '--use_hyperbolic', '--batch-size', '2',
        '--gradient-accumulation-steps', '2', '--epochs', '1',
        '--output_dir', str(root), '--skip_nonfinite_loss',
    ])
    utils.init_distributed_mode_ds(args)
    torch.set_num_threads(2)
    summaries = []
    with patch.object(Uni_Sign_Unified, 'forward_t2p', tiny_t2p):
        for stage in (2, 3):
            torch.manual_seed(31)
            module = make_fixture().cuda()
            apply_stage_freezing(module, stage)
            wrapped = torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[args.gpu], find_unused_parameters=True,
            )
            optimizer, hyp = create_geosign_optimizers(module.args, module)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            engine, optimizer, scheduler = utils.init_deepspeed(args, wrapped, optimizer, scheduler)
            patch_geosign_engine(engine)
            src, tgt, gen = fixture_batch(f'cuda:{args.gpu}')
            # Different rank inputs verify external gradient averaging.
            src['body'] = src['body'] + args.rank * 0.01
            before_hyp = module.hyp_attn_b.detach().clone()
            before_slt = module.pose_proj.weight.detach().clone()
            before_gen = module.diff_head.weight.detach().clone()
            stats = train_joint_epoch(
                args, engine, [(src, tgt)] * 4, [(gen, tgt)] * 4,
                optimizer, 0, hyp_optimizer=hyp,
            )
            assert module.global_step.item() == 2
            assert torch.isfinite(torch.tensor(list(stats.values()))).all()
            assert not torch.equal(before_hyp, module.hyp_attn_b)
            assert not torch.equal(before_gen, module.diff_head.weight)
            assert torch.equal(before_slt, module.pose_proj.weight) == (stage == 2)
            if args.world_size > 1:
                for parameter in (module.hyp_attn_b, module.manifold.isp_c,
                                  module.mt5_model.shared.weight):
                    copies = [torch.empty_like(parameter) for _ in range(args.world_size)]
                    torch.distributed.all_gather(copies, parameter.detach())
                    for other in copies[1:]:
                        torch.testing.assert_close(copies[0], other, rtol=0, atol=1e-6)
            expected_hyp = copy.deepcopy(hyp.state_dict())
            expected_point = module.hyp_attn_b.detach().clone()
            path = root / f'stage{stage}_resume'
            engine.save_checkpoint(str(path), tag='smoke',
                                   client_state={'hyp_optimizer': expected_hyp, 'epoch': 1})
            with torch.no_grad():
                module.hyp_attn_b.add_(0.01)
            loaded_path, state = engine.load_checkpoint(str(path), tag='smoke',
                load_optimizer_states=True, load_lr_scheduler_states=True)
            assert loaded_path is not None
            restore_geosign_optimizer(state, hyp)
            torch.testing.assert_close(module.hyp_attn_b, expected_point)
            for key, value in expected_hyp['state'].items():
                for name, tensor in value.items():
                    if torch.is_tensor(tensor):
                        torch.testing.assert_close(hyp.state_dict()['state'][key][name], tensor)
            # Resume is followed by a real update, not just successful deserialization.
            train_joint_epoch(args, engine, [(src, tgt)] * 2, [(gen, tgt)] * 2,
                              optimizer, 0, hyp_optimizer=hyp)
            assert module.global_step.item() == 3
            summaries.append({'stage': stage, 'world_size': args.world_size,
                              'updates_before_resume': 2, 'updates_after_resume': 1,
                              'metrics': stats})
            del engine, wrapped, optimizer, hyp, scheduler, module
            torch.cuda.empty_cache()
        if args.rank == 0:
            (root / 'summary.json').write_text(json.dumps(summaries, indent=2))
            print('Geo-Sign CUDA/DDP/DeepSpeed stages 2/3 and resume: PASS')
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
