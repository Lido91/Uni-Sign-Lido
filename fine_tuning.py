import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
import copy
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *

def move_src_input_to_device(src_input, model):
    target_dtype = torch.float32
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model.fp16_enabled():
        target_dtype = torch.float16

    for key, value in src_input.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            value = value.to(dtype=target_dtype)
        src_input[key] = value.cuda(non_blocking=True)


def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")
        
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], 
                             args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)

    eval_args = copy.copy(args)
    if args.eval_on_how2sign_test:
        eval_args.dataset = "How2Sign"
        print(
            "WARNING: evaluating directly on How2Sign test; "
            "best_checkpoint.pth will be selected using How2Sign test metrics."
        )

    test_data = S2T_Dataset(path=test_label_paths[eval_args.dataset],
                            args=eval_args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    has_distinct_dev = "How2Sign" not in eval_args.dataset
    if has_distinct_dev:
        dev_data = S2T_Dataset(path=dev_label_paths[eval_args.dataset],
                               args=eval_args, phase='dev')
        print(dev_data)
        # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=dev_sampler,
                                    pin_memory=args.pin_mem)
    else:
        dev_dataloader = test_dataloader
    selection_phase = 'dev' if has_distinct_dev else 'test'

    print(f"Creating model:")
    if args.llm_type == "qwen":
        from models_qwen import Uni_Sign_Qwen
        model = Uni_Sign_Qwen(args=args)
    else:
        model = Uni_Sign(args=args)

    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        # qwen checkpoints only contain trainable tensors (LoRA + pose encoder);
        # the frozen base weights come from the HF checkpoint, so strict=False.
        strict = args.llm_type == "mt5"
        ret = model.load_state_dict(state_dict, strict=strict)
        print(f'Missing keys ({len(ret.missing_keys)}): \n', '\n'.join(ret.missing_keys[:10]))
        print(f'Unexpected keys ({len(ret.unexpected_keys)}): \n', '\n'.join(ret.unexpected_keys[:10]))
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    
    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR" and has_distinct_dev:
                print("📄 dev result")
                evaluate(eval_args, dev_dataloader, model, model_without_ddp, phase='dev')
            print("📄 test result")
            evaluate(eval_args, test_dataloader, model, model_without_ddp, phase='test')

        return 
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            checkpoint_name = 'last_checkpoint.pth' if args.save_last_only else f'checkpoint_{epoch}.pth'
            checkpoint_paths = [output_dir / checkpoint_name]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)

        # single gpu inference
        if utils.is_main_process():
            selection_stats = evaluate(
                eval_args,
                dev_dataloader,
                model,
                model_without_ddp,
                phase=selection_phase,
            )
            if has_distinct_dev:
                evaluate(eval_args, test_dataloader, model, model_without_ddp, phase='test')

            if args.task == "SLT":
                if max_accuracy < selection_stats["bleu4"]:
                    max_accuracy = selection_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(
                    f"BLEU-4 on {eval_args.dataset} {selection_phase} "
                    f"({len(dev_dataloader)} batches): {selection_stats['bleu4']:.2f}"
                )
                print(f'Max BLEU-4: {max_accuracy:.2f}%')
            
            elif args.task == "ISLR":
                if max_accuracy < selection_stats["top1_acc_pi"]:
                    max_accuracy = selection_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(
                    f"PI accuracy on {eval_args.dataset} {selection_phase} "
                    f"({len(dev_dataloader)} batches): {selection_stats['top1_acc_pi']:.2f}"
                )
                print(f'Max PI accuracy: {max_accuracy:.2f}%')
            
            elif args.task == "CSLR":
                if max_accuracy > selection_stats["wer"]:
                    max_accuracy = selection_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)
                            
                print(
                    f"WER on {eval_args.dataset} {selection_phase} "
                    f"({len(dev_dataloader)} batches): {selection_stats['wer']:.2f}"
                )
                print(f'Min WER: {max_accuracy:.2f}%')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in selection_stats.items()},
                        'eval_dataset': eval_args.dataset,
                        'eval_split': selection_phase,
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def _finite_report(d, prefix=""):
    report = {}
    for k, v in d.items():
        if isinstance(v, dict):
            report.update(_finite_report(v, prefix=f"{prefix}{k}."))
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            report[prefix + k] = bool(torch.isfinite(v).all())
    return report


def handle_nonfinite_loss(args, loss_value, step, epoch, streak,
                          src_input, tgt_input, extra=None):
    """Cross-rank handling of a non-finite training loss.

    EVERY rank must call this EVERY step (it issues a collective): a lone
    rank skipping backward() would deadlock the others in the gradient
    all-reduce. Returns 0 when all ranks are finite; otherwise the
    incremented consecutive-skip count, and the caller must skip
    backward/step for this batch. Aborts all ranks together when
    --skip_nonfinite_loss is off, or after --nonfinite_streak_limit
    consecutive skips (persistent non-finite loss = diverged weights)."""
    local_bad = 0.0 if math.isfinite(loss_value) else 1.0
    if torch.distributed.is_initialized():
        bad_flag = torch.tensor(local_bad, device='cuda')
        torch.distributed.all_reduce(bad_flag)
        any_bad = bad_flag.item() > 0
    else:
        any_bad = local_bad > 0
    if not any_bad:
        return 0

    streak += 1
    if local_bad > 0:
        finite_report = _finite_report({**src_input, **(extra or {})})
        msg = (
            "Loss is {} at rank={}, epoch={}, step={}, input_finite={}, "
            "samples={}".format(
                loss_value, utils.get_rank(), epoch, step, finite_report,
                src_input.get('name_batch', [])
            )
        )
        # setup_for_distributed suppresses print on non-master ranks,
        # so also write straight to stderr to survive on any rank
        print(msg)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
        dump_path = os.path.join(
            args.output_dir or ".",
            f"nonfinite_batch_rank{utils.get_rank()}.pt",
        )
        torch.save(
            {
                "src_input": _to_cpu(src_input),
                "tgt_input": tgt_input,
                "extra": _to_cpu(extra) if extra else None,
                "loss": loss_value,
                "step": step,
                "epoch": epoch,
            },
            dump_path,
        )
        sys.stderr.write(f"offending batch saved to {dump_path}\n")

    if not getattr(args, 'skip_nonfinite_loss', False):
        sys.stderr.write(
            f"rank={utils.get_rank()}: aborting on non-finite loss "
            "(pass --skip_nonfinite_loss to skip such steps instead)\n"
        )
        sys.exit(1)
    if streak >= args.nonfinite_streak_limit:
        sys.stderr.write(
            f"rank={utils.get_rank()}: {streak} consecutive non-finite steps "
            "— weights have likely diverged, aborting\n"
        )
        sys.exit(1)
    sys.stderr.write(
        f"rank={utils.get_rank()}: skipping step {step} (consecutive "
        f"non-finite: {streak}/{args.nonfinite_streak_limit})\n"
    )
    return streak


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    nonfinite_streak = 0
    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        move_src_input_to_device(src_input, model)

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)

        total_loss = stack_out['loss']
        loss_value = total_loss.item()

        nonfinite_streak = handle_nonfinite_loss(
            args, loss_value, step, epoch, nonfinite_streak,
            src_input, tgt_input,
        )
        if nonfinite_streak:
            continue

        model.backward(total_loss)
        model.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        tgt_name = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            move_src_input_to_device(src_input, model)
            
            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(stack_out, 
                                                max_new_tokens=100, 
                                                num_beams = 4,
                        )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])
                tgt_name.append(src_input['name_batch'][i])

    tokenizer = model_without_ddp.text_tokenizer
    padding_value = tokenizer.eos_token_id
    
    pad_tensor = torch.ones(150-len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
        if args.eval and (args.dataset == 'How2Sign' or args.dataset == 'OpenASL'):
            if args.skip_bleurt:
                print('BLEURT: skipped (--skip_bleurt)')
            else:
                # BLEURT # follow GloFE
                # Due to the long processing time, only --eval will be executed.
                from bleurt import score
                checkpoint = "./BLEURT-20"
                scorer = score.BleurtScorer(checkpoint)
                scores_bleurt = scorer.score(references=tgt_refs[:], candidates=tgt_pres[:])
                # assert isinstance(scores, list) and len(scores) == 1
                print('BLEURT:', sum(scores_bleurt)/len(scores_bleurt))

    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+f'/{phase}_tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(f"sample: {tgt_name[i]}, prediction: " + tgt_pres[i]+'\n')
        with open(args.output_dir+f'/{phase}_tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(f"sample: {tgt_name[i]}, ground-truth: " + tgt_refs[i]+'\n')
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
