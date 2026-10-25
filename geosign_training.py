"""Optimizer plumbing for the unchanged vendored Geo-Sign objective."""
from types import MethodType

import torch
import utils
from vendor.geosign.training_helpers import (
    _clip_fp32_gradients_with_scalar_support,
    _discard_accumulated_step,
    _grads_are_finite,
    synchronize_optimizer_gradients,
)


def add_geosign_args(parser):
    parser.add_argument('--use_hyperbolic', action='store_true')
    parser.add_argument('--hyp_dim', type=int, default=256)
    parser.add_argument('--init_c', type=float, default=0.5)
    parser.add_argument('--hyp_lr', type=float, default=0.001)
    parser.add_argument('--hyp_stabilize', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--label_smoothing_hyp', type=float, default=0.2)
    parser.add_argument('--hyp_text_emb_src', choices=['token', 'decoder'], default='decoder')
    parser.add_argument('--hyp_text_cmp', choices=['pooled', 'attn', 'token'], default='token')
    parser.add_argument('--max_nonfinite_skips', type=int, default=20)
    parser.add_argument('--allow_plain_geosign_init', action='store_true',
                        help='Allow fresh geometric parameters when loading a plain unified checkpoint.')


def create_geosign_optimizers(args, module):
    from geoopt import ManifoldParameter
    from geoopt.optim import RiemannianAdam
    euclid_params, hyp_params = [], []
    # Preserve Geo-Sign's grouping exactly, including geoopt's isp_c
    # curvature parameter staying in the Euclidean optimizer.
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if isinstance(parameter, ManifoldParameter) or name.endswith('manifold.c'):
            hyp_params.append(parameter)
        else:
            euclid_params.append(parameter)
    if not hyp_params:
        raise ValueError('Geo-Sign enabled but no manifold parameters were found')
    optimizer = torch.optim.AdamW(
        euclid_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=args.opt_betas if args.opt_betas is not None else (0.9, 0.98),
    )
    hyp_optimizer = RiemannianAdam(
        hyp_params, lr=args.hyp_lr, stabilize=args.hyp_stabilize, weight_decay=0.0,
    )
    print(f'Geo-Sign: {len(hyp_params)} manifold parameters; '
          f'{len(euclid_params)} Euclidean parameters; separate RiemannianAdam')
    return optimizer, hyp_optimizer


def patch_geosign_engine(engine):
    engine.clip_fp32_gradients = MethodType(
        _clip_fp32_gradients_with_scalar_support, engine)


def step_geosign(engine, module, hyp_optimizer):
    """Geo-Sign's accumulation-boundary order: sync, check, hyp step, DS step."""
    boundary = engine.is_gradient_accumulation_boundary()
    if boundary:
        synchronize_optimizer_gradients(hyp_optimizer)
        bad = torch.tensor(
            int(not _grads_are_finite(engine, hyp_optimizer)),
            device=next(module.parameters()).device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            _discard_accumulated_step(engine, hyp_optimizer)
            return False
        hyp_optimizer.step()
        hyp_optimizer.zero_grad(set_to_none=True)
    engine.step()
    if boundary:
        module.global_step += 1
    return True


def restore_geosign_optimizer(client_state, hyp_optimizer):
    state = client_state.get('hyp_optimizer')
    if state is None:
        raise ValueError('Enhanced resume state lacks the Riemannian optimizer; '
                         'use a fresh output directory for enhanced training')
    hyp_optimizer.load_state_dict(state)
