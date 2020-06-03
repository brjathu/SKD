from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100, CIFAR100_toy
from dataset.transform_cfg import transforms_options, transforms_test_options, transforms_list

from dataset.dataset_selfsupervision import SSDatasetWrapper

import numpy as np

    
def get_dataloaders(opt):
    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    
    if opt.dataset == 'toy':

        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100_toy(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100_toy(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

#         train_trans, test_trans = transforms_test_options[opt.transform]
        
#         meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans),
#                                      batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                      num_workers=opt.num_workers)
#         meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val', 
#                                                  train_transform=train_trans,
#                                                  test_transform=test_trans),
#                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                     num_workers=opt.num_workers)
        n_cls = 5
        
        return train_loader, val_loader, 5, 5, n_cls    
    
    
    if opt.dataset == 'miniImageNet':

        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        
        
#         ns = [opt.n_shots].copy()
#         opt.n_ways = 32
#         opt.n_shots = 5
#         opt.n_aug_support_samples = 2
        meta_trainloader = DataLoader(MetaCIFAR100(args=opt, partition='train',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=1, shuffle=True, drop_last=False,
                                     num_workers=opt.num_workers)
        
#         opt.n_ways = 5
#         opt.n_shots = ns[0]
#         print(opt.n_shots)
#         opt.n_aug_support_samples = 5
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val', 
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
#         return train_loader, val_loader, meta_trainloader, meta_testloader, meta_valloader, n_cls        
    else:
        raise NotImplementedError(opt.dataset)
        
    return train_loader, val_loader, meta_testloader, meta_valloader, n_cls