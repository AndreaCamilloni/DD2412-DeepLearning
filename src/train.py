# Script for training the model

import argparse
import os
import random
import shutil
import time
import warnings
import utils
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torchvision import transforms, datasets

import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import numpy as np

#from torch import cifar10


from model.model import Model
from model.loss import Loss
from utils.utils import *
from data.loader import *




def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised Classification')
    parser.add_argument('--data_path', type=str, default='data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine', help='learning rate schedule')
    parser.add_argument('--lr_warmup_epochs', type=int, default=10, help='number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nesterov', type=bool, default=True, help='nesterov')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--num_views', type=int, default=2, help='number of views')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
    parser.add_argument('--pretrained', type=bool, default=True, help='pretrained')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--save_path', type=str, default='results', help='path to save results')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--resume', type=bool, default=False, help='resume')
    parser.add_argument('--resume_path', type=str, default='results', help='path to resume')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')
    #step size for step lr
    parser.add_argument('--step_size', type=int, default=10, help='step size for cosine annealing')
    #gamma for step lr
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma for step lr')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    # Set seed
    set_seed(args.seed)

    
    
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True)
    # Load dataset
    train_loader = get_dataset(args)

    # Load model
    model = Model()

    # Load optimizer
    optimizer = get_optimizer(args, model)

    # Load scheduler
    scheduler = get_scheduler(args, optimizer)

    # Load criterion
    criterion = Loss()

    # Train model
    train(args, model, train_loader, optimizer, scheduler, criterion)

    # Save model
    save_model(args, model)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(args):
    
   
    # data loading code
    traindir = os.path.join(args.data, 'train')
    transform = loader.Augmentation(loader.get_transform())
    dataset = loader.ImageFolderWithIndices(traindir, transform=transform)
    #import cifar10
    #train_dataset = cifar10.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader
def get_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    return optimizer

def get_scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    else:
        raise NotImplementedError

    return scheduler

def train(args, model, train_loader,optimizer, scheduler, criterion):
     
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

    # print loss and accuracy
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


def save_model(args, model):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))



if __name__ == '__main__':
    main()
