from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import pickle

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth#, CosfacePairwiseLoss
from .utils.meters import AverageMeter
from .layer import MarginCosineProduct


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import Pool

class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0, gpu=0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        self.criterion_ce_1 = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        self.gpu = gpu
        


    def train(self, epoch, data_loader, optimizer, ema, train_iters=200, print_freq=1, rank=0):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_ce_1 = AverageMeter()
        losses_bce = AverageMeter()
        #losses_cos_pair = AverageMeter()
        #losses_tr = AverageMeter()
        precisions = AverageMeter()
        precisions_1 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader.next()
            data_time.update(time.time() - end)

            s_inputs, s_inputs_support, s_inputs_support_o, targets_list = self._parse_data(source_inputs, self.gpu)
            
            
            s_features, s_cls_out, s_cls_out_1, logits = self.model(s_inputs, s_inputs_support, \
                                                                   s_inputs_support_o, targets_list[0])
            

            loss_ce, loss_ce_1, loss_bce, prec, prec_1 = self._forward(logits, s_cls_out, s_cls_out_1, targets_list)
            loss = loss_ce + loss_ce_1 + 0.5 * loss_bce

            losses_ce.update(loss_ce.item())
            losses_ce_1.update(loss_ce_1.item())
            losses_bce.update(loss_bce.item())
            precisions.update(prec)
            precisions_1.update(prec_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0) and rank==0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR:{:.8f}\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_ce_1 {:.3f} ({:.3f})\t'
                      'Loss_bce {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%}) \t'
                      'Prec_1 {:.2%} ({:.2%}) \t'
                      .format(epoch, i + 1, train_iters,optimizer.param_groups[0]["lr"],
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_ce_1.val, losses_ce_1.avg,
                              losses_bce.val, losses_bce.avg,
                              precisions.val, precisions.avg,
                              precisions_1.val, precisions_1.avg))

    def _parse_data(self, inputs, gpu):
        imgs, imgs_support, imgs_support_o, p1, p2, p3, pids, _, targets_pattern = inputs
        inputs = imgs.cuda(gpu, non_blocking=True)
        inputs_support = imgs_support.cuda(gpu, non_blocking=True)
        inputs_support_o = imgs_support_o.cuda(gpu, non_blocking=True)
        targets = pids.cuda(gpu, non_blocking=True)
        targets_pattern = targets_pattern.cuda(gpu, non_blocking=True)
        targets_list = [targets, targets_pattern]
        return inputs, inputs_support, inputs_support_o, targets_list

    def _forward(self, logits, s_outputs, s_outputs_1, targets_list):
        logits = logits.cuda()
        s_outputs = s_outputs.cuda()
        s_outputs_1 = s_outputs_1.cuda()
        targets_list[0] = targets_list[0].cuda()
        targets_list[1] = targets_list[1].cuda()
        
        loss_ce = self.criterion_ce(s_outputs, targets_list[0])
        loss_ce_1 = self.criterion_ce(s_outputs_1, targets_list[0])
        
        prec, = accuracy(s_outputs.data, targets_list[0].data)
        prec = prec[0]
        
        prec_1, = accuracy(s_outputs_1.data, targets_list[0].data)
        prec_1 = prec_1[0]

        
        pattern_target = targets_list[1]
        pos_weight = torch.ones([pattern_target.shape[1]])
        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight).cuda()
        loss_bce = bce_criterion(logits, pattern_target)

        return loss_ce, loss_ce_1, loss_bce, prec, prec_1


