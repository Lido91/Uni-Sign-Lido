"""Retrieval-augmented SFT (RA-SFT) for Uni-Sign, doc/signrag_style_retriever_plan.md §11.

Same training loop as fine_tuning_signsmd.py (deepspeed, multi-GPU), with
1. dataset ``datasets_rag.S2T_Dataset_RAG``: retrieved training sentences per
   clip from signrag_retrieval.py JSON files, retrieval dropout in training;
2. model ``models_signsmd.Uni_Sign_SignSMD`` unchanged: mT5 input
   [retrieved sentences][translate prompt][pose], zero new parameters, so
   any baseline Uni-Sign checkpoint warm-starts with strict=True.

    deepspeed --include localhost:3 --master_port 29531 fine_tuning_rag.py \\
        --finetune CKPT --dataset How2Sign --task SLT --dtype fp32 \\
        --rag_train_json RAG/retrieval_train_s2t.json --rag_test_json RAG/retrieval_test_s2t.json \\
        --rag_mode real --output_dir OUT

--rag_mode random / empty are the controls (random sentences / no sentences).
"""
import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path

import torch
from timm.optim import create_optimizer
from torch.utils.data import DataLoader
from transformers import get_scheduler

import utils
from config import dev_label_paths, test_label_paths, train_label_paths
from datasets_rag import S2T_Dataset_RAG
from fine_tuning import evaluate, train_one_epoch
from models import get_requires_grad_dict
from models_signsmd import Uni_Sign_SignSMD


def get_rag_args_parser():
    parser = argparse.ArgumentParser("Uni-Sign RA-SFT", parents=[utils.get_args_parser()])
    parser.add_argument("--rag_train_json", default="", help="retrieval_train_s2t.json (filtered retrieval)")
    parser.add_argument("--rag_test_json", default="", help="retrieval_test_s2t.json")
    parser.add_argument("--rag_dev_json", default="", help="retrieval_dev_s2t.json (datasets with a dev split)")
    parser.add_argument("--rag_mode", default="real", choices=["real", "random", "empty", "gloss", "hint"])
    parser.add_argument("--rag_k", type=int, default=3)
    parser.add_argument("--rag_dropout", type=float, default=0.5, help="training-time retrieval dropout")
    parser.add_argument("--signsmd_max_tokens", type=int, default=256,
                        help="max tokens of the retrieved-sentence segment (tail-truncated)")
    parser.add_argument("--limit", type=int, default=0, help="first N clips of every split (smoke tests)")
    parser.add_argument("--train_label", default="", help="label file replacing the train split (e.g. a fold)")
    parser.add_argument("--dev_label", default="", help="label file replacing the dev split")
    parser.add_argument("--test_label", default="", help="label file replacing the test split (e.g. train "
                        "clips to translate with --eval; CSL-Daily only: its pose dir is split-independent)")
    return parser


def loader(data, args, train):
    if args.limit:
        data.list = data.list[:args.limit]
    sampler =torch.utils.data.distributed.DistributedSampler(data, shuffle=True) if train \
        else torch.utils.data.SequentialSampler(data)
    return DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers,
                      collate_fn=data.collate_fn, sampler=sampler, pin_memory=args.pin_mem,
                      drop_last=train), sampler


def main(args):
    utils.init_distributed_mode_ds(args)
    print(args)
    utils.set_seed(args.seed)

    print("Creating dataset:")
    train_data = S2T_Dataset_RAG(args.train_label or train_label_paths[args.dataset], args, "train",
                                 args.rag_train_json)
    print(train_data)
    train_dataloader, train_sampler = loader(train_data, args, train=True)
    test_data = S2T_Dataset_RAG(args.test_label or test_label_paths[args.dataset], args, "test", args.rag_test_json)
    test_dataloader, _ = loader(test_data, args, train=False)
    has_distinct_dev = "How2Sign" not in args.dataset
    if has_distinct_dev:
        dev_data = S2T_Dataset_RAG(args.dev_label or dev_label_paths[args.dataset], args, "dev", args.rag_dev_json)
        dev_dataloader, _ = loader(dev_data, args, train=False)
    else:
        dev_dataloader = test_dataloader
    selection_phase = "dev" if has_distinct_dev else "test"
    eval_args = copy.copy(args)

    print(f"Creating RA-SFT model (rag_mode={args.rag_mode}, k={args.rag_k}, dropout={args.rag_dropout}):")
    model = Uni_Sign_SignSMD(args=args)
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    if args.finetune != "":
        state_dict = torch.load(args.finetune, map_location="cpu")["model"]
        ret = model.load_state_dict(state_dict, strict=True)       # zero new parameters
        print(f"Loaded {args.finetune}: missing {len(ret.missing_keys)}, unexpected {len(ret.unexpected_keys)}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_dataloader) / args.gradient_accumulation_steps),
        num_training_steps=int(args.epochs * len(train_dataloader) / args.gradient_accumulation_steps),
    )
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    output_dir = Path(args.output_dir)

    if args.eval:
        if utils.is_main_process():
            if has_distinct_dev:
                print("dev result")
                evaluate(eval_args, dev_dataloader, model, model_without_ddp, phase="dev")
            print("test result")
            evaluate(eval_args, test_dataloader, model, model_without_ddp, phase="test")
        return

    start_time, max_accuracy = time.time(), 0
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)
        if args.output_dir:
            name = "last_checkpoint.pth" if args.save_last_only else f"checkpoint_{epoch}.pth"
            utils.save_on_master({"model": get_requires_grad_dict(model_without_ddp)}, output_dir / name)
        if utils.is_main_process():                                  # single-GPU inference
            stats = evaluate(eval_args, dev_dataloader, model, model_without_ddp, phase=selection_phase)
            test_stats = stats
            if has_distinct_dev:
                test_stats = evaluate(eval_args, test_dataloader, model, model_without_ddp, phase="test")
            if max_accuracy < stats["bleu4"]:
                max_accuracy = stats["bleu4"]
                if args.output_dir:
                    utils.save_on_master({"model": get_requires_grad_dict(model_without_ddp)},
                                         output_dir / "best_checkpoint.pth")
            print(f"BLEU-4 on {args.dataset} {selection_phase}: {stats['bleu4']:.2f}  (max {max_accuracy:.2f})")
            if args.output_dir:
                row = {**{f"train_{k}": v for k, v in train_stats.items()},
                       **{f"test_{k}": v for k, v in test_stats.items()}}
                if has_distinct_dev:
                    row.update({f"dev_{k}": v for k, v in stats.items()})
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps({**row, "eval_split": selection_phase, "rag_mode": args.rag_mode,
                                        "epoch": epoch, "n_parameters": n_parameters}) + "\n")
    print("Training time {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parsed = get_rag_args_parser().parse_args()
    if parsed.rag_mode in ("real", "hint") and not (parsed.rag_train_json and parsed.rag_test_json):
        raise SystemExit(f"--rag_mode {parsed.rag_mode} needs --rag_train_json and --rag_test_json")
    if parsed.rag_mode in ("real", "hint") and "How2Sign" not in parsed.dataset and not parsed.rag_dev_json:
        raise SystemExit(f"--rag_mode {parsed.rag_mode} on {parsed.dataset} also needs --rag_dev_json")
    if (parsed.train_label or parsed.dev_label or parsed.test_label) and parsed.dataset != "CSL_Daily":
        raise SystemExit("--train_label / --dev_label / --test_label are only supported for CSL_Daily")
    if parsed.output_dir:
        Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed)
