from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import numpy as np
import sys
import math

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from dg import datasets
from dg import models_gem_waveblock_balance_cos
from dg.trainers_cos_ema_pattern_condition_ddp import Trainer
#from dg.evaluators import Evaluator
from dg.utils.data import IterLoader
from dg.utils.data import transforms as T
from dg.utils.sampler_ddp import build_train_sampler
#from dg.utils.data.sampler import RandomMultipleGallerySampler
from dg.utils.data.preprocessor import Preprocessor
from dg.utils.logging import Logger
from dg.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dg.utils.lr_scheduler import WarmupMultiStepLR

start_epoch = best_mAP = 0

torch.set_float32_matmul_precision('high')

import torch._dynamo
torch._dynamo.config.verbose=True
torch._dynamo.config.suppress_errors = True

import pickle

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, epochs, rank, iters=2000):
    root = osp.join(data_dir, name)
    
    #print("ckpt_1")
    with open('%s_reverse.pkl'%name, 'rb') as f:
        reverse_labels = pickle.load(f)
    #print("ckpt_2")

    dataset = datasets.create(name, root, rank)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
         T.Resize((height, width)),
         #T.RandomGrayscale(p=1),
         #T.RandomHorizontalFlip(),
         #T.Pad(10),
         #T.RandomCrop((height, width)),
         T.ToTensor(),
         normalizer
     ])

    test_transformer = T.Compose([
             T.Resize((height, width)),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = build_train_sampler(num_instances, train_set, epoch=epochs)
        #sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    
    
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, reverse_labels, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),reverse_labels,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader, sampler

def main():
    args = parser.parse_args()
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        #mp.spawn(main_worker)
        #mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,args))
        main_worker(args.world_size, args)
    else:
        # Simply call main_worker function
        main_worker(args)
    
    
    
    #main_worker(args)


def main_worker(world_size, args):
    global start_epoch, best_mAP
    
    if args.distributed:
        if args.dist_url == "env://":
            args.rank = int(os.environ["RANK"])
            
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / world_size)
        args.workers = int((args.workers + world_size - 1) / world_size)
    
    if args.rank is not None:
        print("Use GPU: {} for training".format(args.rank))

    cudnn.benchmark = True

    if args.rank == 0:
        if not args.evaluate:
            sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        else:
            log_dir = osp.dirname(args.resume)
            sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
        print("==========\nArgs:{}\n==========".format(args))
    
    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    
    dataset_source, num_classes, train_loader, _, train_sampler = \
        get_data(args.dataset_source, args.data_dir, args.height, args.width, \
             args.batch_size, args.workers, args.num_instances, args.epochs, args.rank, iters)
    
    # Create model
    if args.distributed:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        model = models_gem_waveblock_balance_cos.create(
            args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes
        )
        #if '50' not in args.arch:
        #    model = torch.compile(model)
        
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        
        model.cuda(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=local_rank)
    
    
    if args.rank == 0:
        print("Warning! No norm!")
        print(model)
        
        
    # Evaluator
    lr_rate = args.lr
    '''
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params":[value]}]
    '''
    def f(epoch):
        if epoch<5:
            return (0.99*epoch / 5 + 0.01)
        elif (epoch >= 5 and epoch<10):
            return 1
        else:
            return 0.5 * (math.cos((epoch - 10)/(25 - 10) * math.pi) + 1)

        
    #params_to_optimize = [p for name, p in model.named_parameters() if 'vit.norm.weight_spe' not in name]
    
    params_to_optimize = [] 
    for name, p in model.named_parameters():
        if ('vit.norm.weight_spe' not in name) and ('vit.norm.bias_spe' not in name):
            params_to_optimize.append(p)
        else:
            if args.rank == 0:
                print("Not opt: ", name)
    
    
    
    lambda1 = lambda epoch: f(epoch)
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr_rate)
    
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        #start_epoch = checkpoint['epoch']
        #print("=> Start epoch {}".format(start_epoch))
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    if args.auto_resume:
        ckpts = os.listdir(args.logs_dir)
        num = 0
        for c in range(len(ckpts)):
            if ("checkpoint_" in ckpts[c]):
                name = int(ckpts[c].split('_')[1].split('.')[0])
                if(name > num):
                    num = name
        checkpoint = load_checkpoint(args.logs_dir + '/' + 'checkpoint_' + str(num) +'.pth.tar')
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}".format(start_epoch))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    # Trainer
    trainer = Trainer(model, num_classes, margin=args.margin, gpu=local_rank)
    
    # Start training
    for epoch in range(start_epoch, args.epochs):
        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, ema, train_iters=len(train_loader), print_freq=args.print_freq, rank = args.rank) 
        lr_scheduler.step()
        is_best = False
        if args.rank == 0:
            save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_'+str(epoch)+'.pth.tar'))
        
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        
        if args.rank == 0:
            save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_'+str(epoch)+'_ema.pth.tar'))
        
        ema.restore(model.parameters())
        
        if args.rank == 0:
            old_ckpt = osp.join(args.logs_dir, 'checkpoint_'+str(epoch-1)+'.pth.tar')
            old_ckpt_ema = osp.join(args.logs_dir, 'checkpoint_'+str(epoch-1)+'_ema.pth.tar')
            if osp.exists(old_ckpt):
                os.remove(old_ckpt)
            if osp.exists(old_ckpt_ema):
                os.remove(old_ckpt_ema)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on the source domain")
    #distribute learning
    parser.add_argument('--world-size', default=1, type=int,help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='randperson',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=512, help="input height")
    parser.add_argument('--width', type=int, default=512, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models_gem_waveblock_balance_cos.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--auto_resume', action='store_true',help="auto resume")
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()