from __future__ import print_function

import torch
import time
from tqdm import tqdm
from .util import AverageMeter, accuracy
import numpy as np

def validate(val_loader, model, criterion, opt):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader)) as pbar:
            end = time.time()
            for idx, (input, target, _) in enumerate(pbar):

                if(opt.simclr):
                    input = input[0].float()
                else:
                    input = input.float()
                    
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                                  "Acc@5":'{0:.2f}'.format(top1.avg.cpu().numpy(),2), 
                                  "Loss" :'{0:.2f}'.format(losses.avg,2), 
                                 })
#                 if idx % opt.print_freq == 0:
#                     print('Test: [{0}/{1}]\t'
#                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                            idx, len(val_loader), batch_time=batch_time, loss=losses,
#                            top1=top1, top5=top5))

            print('Val_Acc@1 {top1.avg:.3f} Val_Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg




def embedding(val_loader, model, opt):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    
    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader)) as pbar:
            end = time.time()
            for idx, (input, target, _) in enumerate(pbar):

                if(opt.simclr):
                    input = input[0].float()
                else:
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
                
                # compute output
#                 output = model(input)
                (_,_,_,_, feat), (output, rot_logits) = model(generated_data, rot=True)
#                 loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output[:batch_size], target, topk=(1, 5))
#                 losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))
                
                if(idx==0):
                    embeddings = output
                    classes    = train_targets
                else:
                    embeddings = torch.cat((embeddings, output),0)
                    classes    = torch.cat((classes, train_targets),0)
                    
                    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                                  "Acc@5":'{0:.2f}'.format(top1.avg.cpu().numpy(),2)
                                 })
#                 if idx % opt.print_freq == 0:
#                     print('Test: [{0}/{1}]\t'
#                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                            idx, len(val_loader), batch_time=batch_time, loss=losses,
#                            top1=top1, top5=top5))

            print('Val_Acc@1 {top1.avg:.3f} Val_Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            print(embeddings.size())
            print(classes.size())
            
            np.save("embeddings.npy", embeddings.detach().cpu().numpy())
            np.save("classes.npy", classes.detach().cpu().numpy())
            
            
            
#         with tqdm(val_loader, total=len(val_loader)) as pbar:
#             end = time.time()
#             for idx, (input, target, _) in enumerate(pbar):

#                 if(opt.simclr):
#                     input = input[0].float()
#                 else:
#                     input = input.float()
                    
#                 if torch.cuda.is_available():
#                     input = input.cuda()
#                     target = target.cuda()
                
#                 generated_data = torch.cat((x, x_180),0)
#                 # compute output
# #                 output = model(input)
#                 (_,_,_,_, feat), (output, rot_logits) = model(input, rot=True)
# #                 loss = criterion(output, target)

#                 # measure accuracy and record loss
#                 acc1, acc5 = accuracy(output, target, topk=(1, 5))
# #                 losses.update(loss.item(), input.size(0))
#                 top1.update(acc1[0], input.size(0))
#                 top5.update(acc5[0], input.size(0))
                
#                 if(idx==0):
#                     embeddings = output
#                     classes    = target
#                 else:
#                     embeddings = torch.cat((embeddings, output),0)
#                     classes    = torch.cat((classes, target),0)
                    
                    
#                 # measure elapsed time
#                 batch_time.update(time.time() - end)
#                 end = time.time()
                
#                 pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
#                                   "Acc@5":'{0:.2f}'.format(top1.avg.cpu().numpy(),2)
#                                  })
# #                 if idx % opt.print_freq == 0:
# #                     print('Test: [{0}/{1}]\t'
# #                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
# #                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
# #                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
# #                            idx, len(val_loader), batch_time=batch_time, loss=losses,
# #                            top1=top1, top5=top5))

#             print('Val_Acc@1 {top1.avg:.3f} Val_Acc@5 {top5.avg:.3f}'
#                   .format(top1=top1, top5=top5))
#             print(embeddings.size())
#             print(classes.size())
            
#             np.save("embeddings.npy", embeddings.detach().cpu().numpy())
#             np.save("classes.npy", classes.detach().cpu().numpy())
    return top1.avg, top5.avg, losses.avg
