from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from tqdm import tqdm
import mkl

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_test_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test, meta_test_tune
from eval.cls_eval import validate

from models.resnet import resnet12
import numpy as np
from util import Logger
import wandb
from dataloader import get_dataloaders

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

os.environ["CUDA_VISIBLE_DEVICES"]=str(get_freer_gpu())
mkl.set_num_threads(2)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    parser.add_argument('--ssl', type=bool, default=True, help='use self supervised learning')
    parser.add_argument('--tags', type=str, default="gen0, ssl", help='add tags for the experiment')


    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='tb/', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='/raid/data/IncrementLearn/imagenet/Datasets/MiniImagenet/', help='path to data root')

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
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    
    
    
    #hyper parameters
    parser.add_argument('--gamma', type=float, default=2, help='loss cofficient for ssl loss')
    
    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
        
    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.transform)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()
    
    
    #extras
    opt.fresh_start = True
    
    
    return opt


def main():

    opt = parse_option()
    wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    wandb.config.update(opt)
    wandb.save('*.py')
    wandb.run.save()
    
        
    train_loader, val_loader, meta_testloader, meta_valloader, n_cls = get_dataloaders(opt)

    # model
    model = create_model(opt.model, n_cls, opt.dataset)
    wandb.watch(model)
    
    # optimizer
    if opt.adam:
        print("Adam")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        print("SGD")
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        
        

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        
        
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

       
        val_acc, val_acc_top5, val_loss = 0,0,0 #validate(val_loader, model, criterion, opt)
        
        
        #validate
        start = time.time()
        meta_val_acc, meta_val_std = 0,0#meta_test(model, meta_valloader)
        test_time = time.time() - start
        print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}, Time: {:.1f}'.format(meta_val_acc, meta_val_std, test_time))

        #evaluate
        start = time.time()
        meta_test_acc, meta_test_std = 0,0#meta_test(model, meta_testloader)
        test_time = time.time() - start
        print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}, Time: {:.1f}'.format(meta_test_acc, meta_test_std, test_time))
        

        # regular saving
        if epoch % opt.save_freq == 0 or epoch==opt.epochs:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }            
            save_file = os.path.join(opt.save_folder, 'model_'+str(wandb.run.name)+'.pth')
            torch.save(state, save_file)
            
            #wandb saving
            torch.save(state, os.path.join(wandb.run.dir, "model.pth"))
            
            ## onnx saving
            #dummy_input = torch.autograd.Variable(torch.randn(1, 3, 32, 32)).cuda()
            #torch.onnx.export(model, dummy_input, os.path.join(wandb.run.dir, "model.onnx"))

        wandb.log({'epoch': epoch, 
                   'Train Acc': train_acc,
                   'Train Loss':train_loss,
                   'Val Acc': val_acc,
                   'Val Loss':val_loss,
                   'Meta Test Acc': meta_test_acc,
                   'Meta Test std': meta_test_std,
                   'Meta Val Acc': meta_val_acc,
                   'Meta Val std': meta_val_std
                  })

    #final report 
    generate_final_report(model, opt, wandb)
    
    #remove output.txt log file 
    output_log_file = os.path.join(wandb.run.dir, "output.log")
    if os.path.isfile(output_log_file):
        os.remove(output_log_file)
    else:    ## Show an error ##
        print("Error: %s file not found" % output_log_file)
        
        
        
def train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for idx, (input, target, _) in enumerate(pbar):
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
            
            batch_size = input.size()[0]
            x = input
            x_90 = x.transpose(2,3).flip(2)
            x_180 = x.flip(2).flip(3)
            x_270 = x.flip(2).transpose(2,3)
            generated_data = torch.cat((x, x_90, x_180, x_270),0)
            train_targets = target.repeat(4)
            
            rot_labels = torch.zeros(4*batch_size).cuda().long()
            for i in range(4*batch_size):
                if i < batch_size:
                    rot_labels[i] = 0
                elif i < 2*batch_size:
                    rot_labels[i] = 1
                elif i < 3*batch_size:
                    rot_labels[i] = 2
                else:
                    rot_labels[i] = 3

            # ===================forward=====================
            
            (_,_,_,_, feat), (train_logit, rot_logits) = model(generated_data, rot=True)
            
            rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
            loss_ss = torch.sum(F.binary_cross_entropy_with_logits(input = rot_logits, target = rot_labels))
            loss_ce = criterion(train_logit, train_targets)
            
            loss = opt.gamma * loss_ss + loss_ce
            
            acc1, acc5 = accuracy(train_logit, train_targets, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
            
            
            pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                              "Acc@5":'{0:.2f}'.format(top5.avg.cpu().numpy(),2), 
                              "Loss" :'{0:.2f}'.format(losses.avg,2), 
                             })

    print('Train_Acc@1 {top1.avg:.3f} Train_Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg



def generate_final_report(model, opt, wandb):
    
    
    opt.n_shots = 1
    train_loader, val_loader, meta_testloader, meta_valloader, _ = get_dataloaders(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader, use_logit=True)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader, use_logit=True)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))
    
    
    wandb.log({'Final Meta Test Acc @1': meta_test_acc,
               'Final Meta Test std @1': meta_test_std,
               'Final Meta Test Acc  (feat) @1': meta_test_acc_feat,
               'Final Meta Test std  (feat) @1': meta_test_std_feat,
               'Final Meta Val Acc @1': meta_val_acc,
               'Final Meta Val std @1': meta_val_std,
               'Final Meta Val Acc   (feat) @1': meta_val_acc_feat,
               'Final Meta Val std   (feat) @1': meta_val_std_feat
              })

    
    opt.n_shots = 5
    train_loader, val_loader, meta_testloader, meta_valloader, _ = get_dataloaders(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader, use_logit=True)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader, use_logit=True)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))

    wandb.log({'Final Meta Test Acc @5': meta_test_acc,
               'Final Meta Test std @5': meta_test_std,
               'Final Meta Test Acc  (feat) @5': meta_test_acc_feat,
               'Final Meta Test std  (feat) @5': meta_test_std_feat,
               'Final Meta Val Acc @5': meta_val_acc,
               'Final Meta Val std @5': meta_val_std,
               'Final Meta Val Acc   (feat) @5': meta_val_acc_feat,
               'Final Meta Val std   (feat) @5': meta_val_std_feat
              })
    
    
if __name__ == '__main__':
    main()
