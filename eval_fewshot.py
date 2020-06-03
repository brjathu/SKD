from __future__ import print_function

import argparse
import socket
import time
import os
import mkl


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list

from eval.meta_eval import meta_test, meta_test_tune
from eval.cls_eval import validate, embedding
from dataloader import get_dataloaders

mkl.set_num_threads(2)



def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default="", help='absolute path to .pth model')
#     parser.add_argument('--model_path', type=str, default="/raid/data/IncrementLearn/imagenet/neurips20/model/maml_miniimagenet_test_5shot_step_5_5ways_5shots/pretrain_maml_miniimagenet_test_5shot_step_5_5ways_5shots.pt", help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', "toy"])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # specify data_root
    parser.add_argument('--data_root', type=str, default='/raid/data/IncrementLearn/imagenet/Datasets/MiniImagenet/', help='path to data root')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    
    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    
    opt = parser.parse_args()
    
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
        
    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        if(opt.dataset=="toy"):
            opt.data_root = '{}/{}'.format(opt.data_root, "CIFAR-FS")
        else:   
            opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    return opt


def main():

    opt = parse_option()

    opt.n_test_runs = 600
    train_loader, val_loader, meta_testloader, meta_valloader, n_cls = get_dataloaders(opt)

    # load model
    model = create_model(opt.model, n_cls, opt.dataset)
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt["model"])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    
#     start = time.time()
#     test_acc, test_std = meta_test(model, meta_testloader)
#     test_time = time.time() - start
#     print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))


    for i in range(1):
        val_acc, val_acc_top5, val_loss = embedding(val_loader, model, opt)
        
        
#         start = time.time()
#         test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False)
#         test_time = time.time() - start
#         print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))
    
    
#     start = time.time()
#     test_acc, test_std = meta_test_tune(model, meta_testloader)
#     test_time = time.time() - start
#     print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

#     start = time.time()
#     test_acc_feat, test_std_feat = meta_test_tune(model, meta_testloader, use_logit=False)
#     test_time = time.time() - start
#     print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))



if __name__ == '__main__':
    main()
