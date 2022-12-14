import argparse
import os
import random
import shutil
import time
import warnings
import utils
import sys

import wandb
import yaml

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
#import torch.distributed as dist
import torch.optim
#import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
#import torchvision.models as torchvision_models
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
#from apex import parallel
from apex.parallel.LARC import LARC

from model.model import Model
from model.loss import Loss
from utils import utils 


parser = argparse.ArgumentParser(description='PyTorch ImageNet Self-Supervised Training')
parser.add_argument('--config', type=str,
                    help='path to config file', default="./configs/train100ep.yaml")
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset', default="./cifar10")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='linear warmup epochs (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=4.8, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--start-warmup', default=0.3, type=float,
                    help='initial warmup learning rate')
parser.add_argument('--final-lr', default=0.0048, type=float,
                    help='final learning rate (None for constant learning rate)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--sgd', action='store_true',
                    help='use SGD optimizer')
parser.add_argument('--lars', action='store_true',
                    help='use LARS optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=16, type=int,
                    metavar='N', help='print frequency (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--cls-size', type=int, default=10,
                    help='number of classes')
parser.add_argument('--save-path', default='./saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--queue-len', default=100, type=int,
                    help='length of nearest neighbor queue')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=2, type=int,
                    help='number of MLP hidden layers')
parser.add_argument('--row-tau', default=0.1, type=float,
                    help='row softmax temperature (default: 0.1)')
parser.add_argument('--col-tau', default=0.05, type=float,
                    help='column softmax temperature (default: 0.05)')
parser.add_argument('--use-amp', action='store_true',
                    help='use automatic mixed precision')
parser.add_argument('--use-bn', action='store_true',
                    help='use batch normalization layers in MLP')
parser.add_argument('--global-crops-scale', type=float, nargs='+', default=(0.4, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
                    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we 
                    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local-crops-number', type=int, default=6,
                    help="""Number of small local views to generate. 
                    Set this parameter to 0 to disable multi-crop training. 
                    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local-crops-scale', type=float, nargs='+', default=(0.05, 0.4),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image. 
                    Used for small local view cropping of multi-crop.""")
parser.add_argument('--eps', type=float, default=1e-8,
                    help='small value to avoid division by zero and log(0)')
parser.add_argument('--subset', default=0, type=int,
                    help='subset size')
parser.add_argument('--activation', type=str,
                    help='relu or leaky_relu in MLP layers', default='leaky_relu')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")
parser.add_argument("--wandb-entity", default=None, help="Specify WandB entity name")


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)

def main():
    args = parser.parse_args()

    if args.config is not None:
        with open(str(args.config), "r") as file:
            # safe load
            config = yaml.safe_load(file)
 
        update_args(args, config)
    if args.wandb:
        _wandb = vars(args)
        wandb.init(project=args.wandb, entity=args.wandb_entity, config=_wandb)
        # update_args(args, dict(run.config))
    
    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, args)


def main_worker(gpu,args):
    args.gpu = gpu
    
    model = Model(backbone=args.arch,
                  dim=args.dim,
                  hidden_dim=args.hidden_dim,
                  num_classes=args.cls_size,
                  num_layers_cls=args.num_hidden,
                  use_bn=args.use_bn,
                  #backbone_dim=backbone_dim,
                  activation_cls=args.activation,
                  #pretrained=args.pretrained,
                  )
    
    
    print(model)

    # nearest neighbor queue
    nn_queue = utils.NNQueue(args.queue_len, args.dim, args.gpu)

    if args.gpu is not None:
        print('-' * 150)
        print(" " * 50, "Using GPU: {} for training".format(args.gpu))
        print('-' * 150)
        #torch.autograd.set_detect_anomaly(True)
        torch.cuda.set_device('cuda:0') 
        # DataParallel will divide and allocate batch_size to all available GPUs
        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #    model.features = torch.nn.DataParallel(model.features)
        #    model.cuda()
        #else:   
        model = model.cuda()

    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)

    if args.lars:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            nn_queue = checkpoint['nn_queue']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True # Should make training faster
    
    traindir = os.path.join(args.data, 'train')
    transform = utils.DataAugmentation(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    dataset = utils.ImageFolderWithIndices(traindir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    n = len(dataset)

    #Train with subset of data 
    if args.subset > 0: 
        # generate random indices for each class in args.cls_size
        # and then concatenate them to get a list of indices
        # of length args.subset
        # this is done to ensure that the subset is balanced
        indices = []
        for i in range(args.cls_size):
            indices += random.sample(range(i*int(n/args.cls_size), (i+1)*int(n/args.cls_size)), int(args.subset/args.cls_size))
            #print("Class {} has {} samples".format(i, len(indices)))

        #idxs = np.random.choice(len(dataset), size=args.subset, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        

    if torch.cuda.is_available():
        criterion = Loss(row_tau=args.row_tau, col_tau=args.col_tau, eps=args.eps).cuda()  
    else:
        criterion = Loss(row_tau=args.row_tau, col_tau=args.col_tau, eps=args.eps)
       
      
    # schedulers
    lr_schedule = utils.cosine_scheduler_with_warmup(base_value=args.lr,
                                                     final_value=args.final_lr,
                                                     epochs=args.epochs,
                                                     niter_per_ep=len(loader),
                                                     warmup_epochs=args.warmup_epochs,
                                                     start_warmup_value=args.start_warmup)

    scaler = GradScaler(enabled=args.use_amp, init_scale=2. ** 14)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        loss_i, acc1 = train(loader, model, nn_queue, scaler, criterion, optimizer, lr_schedule, epoch, args)
        
        if args.wandb:
            wandb.log({"Train Loss": loss_i, "Train Acc": acc1})
        
        # Print to console training loss and accuracy
        print('-' * 100)
        print('Epoch: [{0}]\t'
                'Loss {loss:.4f}\t'
                'Acc@1 {top1:.3f}\t'.format(epoch, loss=loss_i, top1=acc1))
        print('-' * 100)


        # remember best acc@1 and save checkpoint
        is_best = True if epoch == 0 else loss_i < best_loss
        best_loss = loss_i if epoch == 0 else min(loss_i, best_loss)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'nn_queue': nn_queue,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, is_milestone=(epoch + 1) % 25 == 0,
            filename=os.path.join(args.save_path, 'model_last.pth.tar'))

def train(loader, model, nn_queue, scaler, criterion, optimizer, lr_schedule, epoch, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.6f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets, indices) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cos:
            # update learning rate
            adjust_lr(optimizer, lr_schedule, iteration=epoch * len(loader) + i)

        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = [x.cuda(args.gpu, non_blocking=True) for x in images]
            targets = targets.cuda(args.gpu, non_blocking=True)  # only used for monitoring progress, NOT for training
            indices = indices.cuda(args.gpu, non_blocking=True)

        with autocast(enabled=args.use_amp):
            # compute embeddings
            embds = model(images, return_embds=True)

            # view1 embeddings
            embds1 = embds[0].clone().detach()

            if nn_queue.full:
                #if not args.no_nn_aug:  # if queue is full and nn is enabled, replace view1 with view1-nn
                #    embds[0], nn_targets = nn_queue.get_nn(embds1, indices)
                #else:  # if nn augmentation is disabled do not replace, but use for monitoring progress
                _, nn_targets = nn_queue.get_nn(embds1, indices)

                # measure accuracy of nearest neighbor (for monitoring progress)
                acc1 = (targets.view(-1, ) == nn_targets.view(-1, )).float().mean().view(1, ) * 100.0
                # compute accuracy of all workers
                top1.update(acc1[0], targets.size(0))


            # push embeddings of view1 (all workers) into queue
            nn_queue.push(embds1, targets, indices)

            # compute probs
            probs = model(embds, return_embds=False)

            with autocast(enabled=False):
                # compute loss
                loss = criterion(probs)
            

        assert not torch.isnan(loss), 'loss is nan!'
        
        
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss.detach() 
        
        # measure elapsed time
        losses.update(loss.item(), probs[0][0].size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()



        if i % args.print_freq == args.print_freq - 1:
            progress.display(i)

    return losses.avg, top1.avg



def save_checkpoint(state, is_best, is_milestone, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
        print('Best model was saved.')
    if is_milestone:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_{}.pth.tar'.format(state['epoch'])))
        print('Milestone {} model was saved.'.format(state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_lr(optimizer, lr_schedule, iteration):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_schedule[iteration]



if __name__ == '__main__':
    main()
