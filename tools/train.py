#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
import torch.utils.tensorboard as tensorboard
from contextlib import suppress
from modeling import models
from data import datasets, transforms, dataloaders
from engine.trainer import Trainer
from engine.tester import Tester
from engine.evaluator import Evaluator
from utils.logger import Logger
from utils.serialization import load_checkpoint, save_checkpoint

# PyTorch 메모리 단편화 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"


def argument_parser():
    parser = argparse.ArgumentParser(description='HDMapNet Safe Training')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--arch', type=str, default='CondLSTR2DRes10', choices=models.names())
    parser.add_argument('--dataset', type=str, default='tusimple', help='dataset name')
    parser.add_argument('--data-dir', type=str, default='../temp/data')
    parser.add_argument('--version', type=str, default='v1.0')
    parser.add_argument('--task', type=str, default='lane_det_2d')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-epochs', '-e', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--precision', type=str, default='amp', choices=['fp32','amp','amp_bf16'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logs-dir', type=str, default='../temp/logs')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    
    # 이미지 관련 인자 추가
    parser.add_argument('--img-height', type=int, default=360, help='Input image height')
    parser.add_argument('--img-width', type=int, default=640, help='Input image width')
    parser.add_argument('--cut-height', type=int, default=0, help='Height to cut from top of image')
    
    return parser


# BatchNorm -> GroupNorm 변환
def convert_bn_to_gn(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=min(num_groups, module.num_features), num_channels=module.num_features)
            setattr(model, name, gn)
        else:
            convert_bn_to_gn(module, num_groups)
    return model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True
    cudnn.deterministic = False
    cuda.matmul.allow_tf32 = True

    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
        dist.barrier()

    # Logger & Tensorboard
    if not args.distributed or args.local_rank == 0:
        os.makedirs(args.logs_dir, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
        tb_writer = tensorboard.SummaryWriter(os.path.join(args.logs_dir, 'tensorboard'))
    else:
        tb_writer = None

    # Datasets & Dataloaders (이미지 크기 축소 포함)
    data_root = os.path.join(args.data_dir, args.version)
    train_transforms, train_collate_fn = transforms.create(
        args.dataset, train=True, root=data_root, version=args.version,
        resize=(args.img_height, args.img_width), cut_height=args.cut_height
    )
    test_transforms, test_collate_fn = transforms.create(
        args.dataset, train=False, root=data_root, version=args.version,
        resize=(args.img_height, args.img_width), cut_height=args.cut_height
    )

    train_dataset = datasets.create(args.dataset, data_root, split='train', transform=train_transforms)
    test_dataset = datasets.create(args.dataset, data_root, split='val', transform=test_transforms)

    train_loader = dataloaders.create(args.dataset, train_dataset, args.batch_size, train=True,
                                      distributed=args.distributed, num_workers=args.num_workers,
                                      collate_fn=train_collate_fn)
    test_loader = dataloaders.create(args.dataset, test_dataset, args.batch_size, train=False,
                                     distributed=args.distributed, num_workers=args.num_workers,
                                     collate_fn=test_collate_fn)

    # Model
    model = models.create(args.arch, norm_layer=nn.BatchNorm2d, num_classes=1)
    model = convert_bn_to_gn(model)  # BatchNorm -> GroupNorm
    model = model.cuda()

    # Gradient Checkpointing 활성화
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Mixed precision
    grad_scaler = amp.GradScaler() if args.precision in ['amp','amp_bf16'] else None
    autocast = amp.autocast if args.precision in ['amp','amp_bf16'] else suppress

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # DDP
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)

    # Trainer/Evaluator
    trainer = Trainer(model, optimizer, scheduler, grad_scaler, autocast=autocast, tb_writer=tb_writer)
    evaluator = Evaluator(args.task, model, autocast=autocast, tb_writer=tb_writer, num_classes=1)

    if args.eval:
        evaluator(test_loader)
        return

    # Training loop
    for epoch in range(args.num_epochs):
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)

        # CUDA 메모리 정리
        torch.cuda.empty_cache()

        trainer(train_loader, epoch, best_prec1=0)
        scheduler.step()

    # Final evaluation
    evaluator(test_loader)


if __name__ == '__main__':
    parser = argument_parser()
    main(parser.parse_args())
