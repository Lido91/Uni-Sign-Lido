"""Functions copied verbatim from the local Geo-Sign training implementation."""
import torch
import utils
from utils import get_world_size
import deepspeed.comm as dist

def _clip_fp32_gradients_with_scalar_support(engine):
    """Keep DeepSpeed's distributed norm calculation compatible with scalar grads."""
    from deepspeed.runtime.utils import clip_grad_norm_

    parameters = []
    for parameter in engine.module.parameters():
        if parameter.grad is None:
            continue
        if parameter.ndim == 0:
            # Older DeepSpeed uses a [1] clipping coefficient in an in-place
            # multiply. A detached view shares gradient storage without changing
            # the actual parameter/gradient shape or checkpoint representation.
            view = parameter.detach().view(1)
            view.grad = parameter.grad.view(1)
            for attr in ('model_parallel', 'tensor_model_parallel'):
                if hasattr(parameter, attr):
                    setattr(view, attr, getattr(parameter, attr))
            parameter = view
        parameters.append(parameter)
    if parameters:
        return clip_grad_norm_(parameters, engine.gradient_clipping(), mpu=engine.mpu)

def synchronize_optimizer_gradients(optimizer):
    """Average gradients owned by an optimizer outside the DeepSpeed optimizer.

    Call at the accumulation boundary, before clipping or stepping. Re-averaging
    already reduced gradients (ZeRO-0 FP32) leaves their values unchanged.
    """
    if get_world_size() == 1:
        return
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(get_world_size())

def _grads_are_finite(model, hyp_optimizer) -> bool:
    """True when every gradient about to be applied is finite.

    DeepSpeed's bf16 optimizer accumulates gradients in fp32 flat buffers; in fp32 mode the
    gradients live on the module parameters. RiemannianAdam parameters are checked too.
    """
    grads = []
    ds_optimizer = getattr(model, 'optimizer', None)
    hp_grads = getattr(ds_optimizer, 'fp32_groups_gradient_flat_partition', None)
    if hp_grads:
        grads.extend(g for g in hp_grads if g is not None)
    else:
        module = getattr(model, 'module', model)
        grads.extend(p.grad for p in module.parameters() if p.grad is not None)
    if hyp_optimizer is not None:
        grads.extend(p.grad for group in hyp_optimizer.param_groups
                     for p in group['params'] if p.grad is not None)
    if not grads:
        return True
    return bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item())

def _discard_accumulated_step(model, hyp_optimizer) -> None:
    """Drop the accumulated gradients and book the step as skipped without calling model.step().

    model.step() cannot be used here: DeepSpeed's fp32 path would apply the non-finite update
    and its bf16 optimizer asserts on a non-finite (or zero) global norm.
    """
    ds_optimizer = getattr(model, 'optimizer', None)
    for method in ('clear_hp_grads', 'clear_lp_grads'):
        fn = getattr(ds_optimizer, method, None)
        if callable(fn):
            fn()
    model.zero_grad()
    if hyp_optimizer is not None:
        hyp_optimizer.zero_grad(set_to_none=True)
    if hasattr(model, 'micro_steps'):
        model.micro_steps += 1
    if hasattr(model, 'skipped_steps'):
        model.skipped_steps += 1
