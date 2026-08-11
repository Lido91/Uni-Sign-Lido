"""DeepSpeed-free trainer for the Qwen (LoRA) variant of Uni-Sign.

Plain PyTorch training loop (AdamW over trainable params, bf16 autocast,
cosine schedule, optional torchrun DDP) so it runs in the motion-smd `smd`
conda env without installing deepspeed/timm/decord/sacrebleu. Translation
metrics use pycocoevalcap (the same package as the motion-smd How2Sign
evaluation) instead of the vendored SLRT sacrebleu.

Single GPU (smoke test):
    python train_qwen.py --batch-size 2 --epochs 1 --dataset OpenASL ...
Multi GPU:
    CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port 29512 \
        train_qwen.py --batch-size 8 --gradient-accumulation-steps 16 ...

Checkpoint format is identical to fine_tuning.py: {'model': trainable_state_dict}
(only LoRA + pose encoder tensors), interchangeable between the two entries.
"""
import contextlib
import copy
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import argparse

import utils
from config import train_label_paths, dev_label_paths, test_label_paths
from datasets import S2T_Dataset
from models import get_requires_grad_dict
from models_qwen import Uni_Sign_Qwen


def translation_metrics(refs, preds):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    gts = {i: [r.strip().lower() or "<empty>"] for i, r in enumerate(refs)}
    res = {i: [p.strip().lower() or "<empty>"] for i, p in enumerate(preds)}
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    out = {f'bleu{i + 1}': bleu[i] * 100 for i in range(4)}
    out['rouge'] = rouge * 100
    return out


def autocast_ctx(args):
    if args.dtype == 'bf16':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    if args.dtype == 'fp16':
        return torch.autocast('cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def move_to_cuda(src_input):
    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            # Pose arrays loaded from NumPy may be float64.  The trainable pose
            # encoder is kept in float32, so normalize floating inputs before
            # its Linear layers while preserving integer masks/indices.
            if value.is_floating_point():
                value = value.float()
            src_input[key] = value.cuda(non_blocking=True)


def main(args):
    assert args.task == 'SLT', 'train_qwen.py only supports SLT'
    args.llm_type = 'qwen'

    utils.init_distributed_mode(args)
    print(args)
    utils.set_seed(args.seed)

    print('Creating dataset:')
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], args=args, phase='train')
    print(train_data)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  sampler=train_sampler,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)

    eval_args = copy.copy(args)
    if args.eval_on_how2sign_test:
        eval_args.dataset = 'How2Sign'
        print('WARNING: evaluating directly on How2Sign test; '
              'best_checkpoint.pth will be selected using How2Sign test metrics.')

    test_data = S2T_Dataset(path=test_label_paths[eval_args.dataset], args=eval_args, phase='test')
    print(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 sampler=torch.utils.data.SequentialSampler(test_data),
                                 pin_memory=args.pin_mem)

    has_distinct_dev = 'How2Sign' not in eval_args.dataset
    if has_distinct_dev:
        dev_data = S2T_Dataset(path=dev_label_paths[eval_args.dataset], args=eval_args, phase='dev')
        print(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=torch.utils.data.SequentialSampler(dev_data),
                                    pin_memory=args.pin_mem)
    else:
        dev_dataloader = test_dataloader
    selection_phase = 'dev' if has_distinct_dev else 'test'

    print('Creating model:')
    model = Uni_Sign_Qwen(args=args)
    model.cuda()
    model.train()

    # trainable params (LoRA + pose encoder) in fp32; frozen base stays in args.dtype
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        ret = model.load_state_dict(state_dict, strict=False)
        # missing keys are the frozen Qwen base weights (come from the HF checkpoint)
        print(f'Missing keys ({len(ret.missing_keys)}): ', ret.missing_keys[:5], '...')
        print(f'Unexpected keys ({len(ret.unexpected_keys)}): ', ret.unexpected_keys[:5])
        assert len(ret.unexpected_keys) == 0, 'checkpoint does not match the qwen model'

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params) / 1e6
    n_total = sum(p.numel() for p in model_without_ddp.parameters()) / 1e6
    print(f'trainable params: {n_trainable:.1f}M / total {n_total:.1f}M')

    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  eps=args.opt_eps)
    updates_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(name='cosine',
                                 optimizer=optimizer,
                                 num_warmup_steps=int(args.warmup_epochs * updates_per_epoch),
                                 num_training_steps=int(args.epochs * updates_per_epoch))

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.eval:
        if utils.is_main_process():
            if has_distinct_dev:
                print('dev result')
                evaluate(eval_args, dev_dataloader, model_without_ddp, phase='dev')
            print('test result')
            evaluate(eval_args, test_dataloader, model_without_ddp, phase='test')
        return

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    max_accuracy = 0

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, lr_scheduler, epoch)

        if output_dir and utils.is_main_process():
            checkpoint_name = 'last_checkpoint.pth' if args.save_last_only else f'checkpoint_{epoch}.pth'
            torch.save({'model': get_requires_grad_dict(model_without_ddp)},
                       output_dir / checkpoint_name)

        if utils.is_main_process():
            selection_stats = evaluate(eval_args, dev_dataloader, model_without_ddp,
                                       phase=selection_phase)
            if has_distinct_dev:
                evaluate(eval_args, test_dataloader, model_without_ddp, phase='test')

            if max_accuracy < selection_stats['bleu4']:
                max_accuracy = selection_stats['bleu4']
                if output_dir:
                    torch.save({'model': get_requires_grad_dict(model_without_ddp)},
                               output_dir / 'best_checkpoint.pth')

            print(f"BLEU-4 on {eval_args.dataset} {selection_phase}: {selection_stats['bleu4']:.2f}")
            print(f'Max BLEU-4: {max_accuracy:.2f}')

            if output_dir:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in selection_stats.items()},
                             'eval_dataset': eval_args.dataset,
                             'eval_split': selection_phase,
                             'epoch': epoch,
                             'n_trainable_M': n_trainable}
                with (output_dir / 'log.txt').open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')

        if args.distributed:
            torch.distributed.barrier()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time}')


def train_one_epoch(args, model, data_loader, optimizer, lr_scheduler, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'
    optimizer.zero_grad()

    accum = args.gradient_accumulation_steps
    num_batches = len(data_loader)

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        move_to_cuda(src_input)

        with autocast_ctx(args):
            stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping. step={step}, '
                  f"samples={src_input.get('name_batch', [])}")
            sys.exit(1)

        (total_loss / accum).backward()

        if (step + 1) % accum == 0 or (step + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.gradient_clipping)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model_without_ddp, phase):
    # runs on the unwrapped model on rank 0 only (calling the DDP wrapper on a
    # single rank would hang on buffer broadcast)
    model_without_ddp.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    tgt_pres, tgt_refs, tgt_name = [], [], []

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, 'Test:')):
        move_to_cuda(src_input)

        with autocast_ctx(args):
            stack_out = model_without_ddp(src_input, tgt_input)
        metric_logger.update(loss=stack_out['loss'].item())

        output = model_without_ddp.generate(stack_out, max_new_tokens=100, num_beams=4)

        tokenizer = model_without_ddp.text_tokenizer
        for i in range(len(output)):
            tgt_pres.append(tokenizer.decode(output[i], skip_special_tokens=True))
            tgt_refs.append(tgt_input['gt_sentence'][i])
            tgt_name.append(src_input['name_batch'][i])

    scores = translation_metrics(tgt_refs, tgt_pres)
    for k, v in scores.items():
        metric_logger.meters[k].update(v)

    if args.output_dir:
        with open(os.path.join(args.output_dir, f'{phase}_tmp_pres.txt'), 'w') as f:
            for name, pres in zip(tgt_name, tgt_pres):
                f.write(f'sample: {name}, prediction: {pres}\n')
        with open(os.path.join(args.output_dir, f'{phase}_tmp_refs.txt'), 'w') as f:
            for name, ref in zip(tgt_name, tgt_refs):
                f.write(f'sample: {name}, ground-truth: {ref}\n')

    model_without_ddp.train()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser('Uni-Sign Qwen scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
