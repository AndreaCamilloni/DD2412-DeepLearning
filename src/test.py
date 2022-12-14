import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import math
import utils
import pickle
import yaml
from PIL import Image, ImageOps, ImageDraw

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from model.model import Model
from model.loss import Loss
from utils import utils

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as adjusted_nmi
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index
from scipy.optimize import linear_sum_assignment

#from robustness.tools.breeds_helpers import ClassHierarchy
#from robustness.tools.breeds_helpers import make_nonliving26, make_entity13, make_entity30, make_living17

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Self-Supervised Training')
parser.add_argument('--config', type=str,
                    help='path to config file', default="./configs/test.yaml")
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default='../saved/model_best.pth.tar', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--cls-size', type=int, default=10,
                    help='size of classification layer')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=3, type=int,
                    help='number of MLP hidden layers')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--use-bn', action='store_true',
                    help='use batch normalization layers in MLP')
parser.add_argument('--num-classes', default=10, type=int,
                    help='number of low entropy classes to visualize')
parser.add_argument('--num-samples-per-class', default=4, type=int,
                    help='number of samples per class to visualize (must be a square number)')
parser.add_argument('--cls-num', default=0, type=int,
                    help='index of classifier head to use (default: 0)')  # best SCAN head is 4.
parser.add_argument('--subset-file', default=None, type=str,
                    help='path to imagenet subset txt file')
parser.add_argument('--activation', type=str, default='relu',
                    help='use regular relu layers instead of leaky relu in MLP')
#TODO: remove arg model, because we are only using the self-classifier
parser.add_argument('--model', default='self-classifier', const='self-classifier', nargs='?',
                    choices=['self-classifier'],
                    help='type of pretrained model (default: %(default)s) - Only choice is self-classifier')
#parser.add_argument('--kmeans-cls', default=None, type=str,
#                    help='path to kmeans classifier')
#parser.add_argument('--superclass', default=None, const=None, nargs='?', type=str,
#                    choices=[None, '0', '1', '2', '3', '4', '5', '6', '7', '8',
#                             'entity13', 'entity30', 'living17', 'nonliving26'],
#                    help='type of superclass subset (default: %(default)s)')
#parser.add_argument('--imagenet-info-path', default='./imagenet_info/', type=str,
#                    help='includes dataset_class_info.json, class_hierarchy.txt, node_names.txt')

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
    
    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # save log file
    sys.stdout = utils.PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))
    print(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.model == 'self-classifier':
        # Import trained model from saved checkpoint
        #model = torch.hub.load('.....', '...')

        model = Model(backbone=args.arch,
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    num_classes=args.cls_size,
                    #num_cls=args.num_cls,
                    num_layers_cls=args.num_hidden,
                    use_bn=args.use_bn,
                    #backbone_dim=backbone_dim,
                    #fixed_cls=args.fixed_cls,
                    activation_cls=args.activation,
                    #pretrained=args.pretrained,
                    )
    #if args.model == 'self-classifier':
        #model = Model(base_model=models.__dict__[args.arch](),
        #              dim=args.dim,
        #              hidden_dim=args.hidden_dim,
        #              cls_size=args.cls_size,
        #              num_cls=args.num_cls,
        #              num_hidden=args.num_hidden,
        #              use_bn=args.use_bn,
        #              no_leaky=args.no_leaky)
    #elif args.model == 'scan' or args.model == 'sela':
    #    model = models.__dict__[args.arch]()
    #elif args.model == 'barlowtwins':
    #    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    #    model.fc = nn.Identity()
    #elif args.model == 'dino':
    #    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    #else:
    #    model = models.__dict__[args.arch]()
    #    model.fc = nn.Identity()
    print(model)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu") #map_location="cpu"??

            # load state dictionary
        #    if args.model == 'scan':
        #        state_dict = checkpoint['model']
        #    elif args.model == 'swav':
        #        state_dict = checkpoint
        #    elif args.model == 'obow':
        #        state_dict = checkpoint['network']
        #    else:
        #        state_dict = checkpoint['state_dict']
            state_dict = checkpoint['state_dict']

            # remove module. prefix
            for k in list(state_dict.keys()):

                #if args.model == 'scan':
                #    if k.startswith('backbone.'):
                #        # remove prefix
                #        state_dict[k[len("backbone."):]] = state_dict[k]
                #        del state_dict[k]
                #    elif k.startswith('cluster_head.{}'.format(args.cls_num)):
                #        # remove prefix
                #        state_dict['fc.' + k[len("cluster_head.{}.".format(args.cls_num)):]] = state_dict[k]
                #        del state_dict[k]
                #    else:
                #        del state_dict[k]  # delete other heads

                #elif args.model == 'mocov2':
                #    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                #        new_k = k[len('module.encoder_q.'):]
                #        state_dict[new_k] = state_dict[k]
                #        del state_dict[k]

                #elif args.model == 'swav':
                #    if k.startswith('module') and not k.startswith('module.projection_head'):
                #        new_k = k[len('module.'):]
                #        state_dict[new_k] = state_dict[k]
                #        del state_dict[k]

                #elif args.model == 'obow':
                #    if k.startswith('fc'):
                #        del state_dict[k]

                if args.model == 'self-classifier':
                    if k.startswith('module.'):
                        # remove prefix
                        state_dict[k[len("module."):]] = state_dict[k]
                        del state_dict[k]
                else: 
                    warnings.warn('Unknown model: {}'.format(args.model))
                    sys.exit()

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert len(msg.missing_keys) == 0, "missing_keys: {}".format(msg.missing_keys)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained,
                                                                       checkpoint['epoch'] if 'epoch' in checkpoint else 'NA'))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        print('=> using {} GPUs.'.format(torch.cuda.device_count()))
        # DataParallel will divide and allocate batch_size to all available GPUs
        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #    model.features = torch.nn.DataParallel(model.features)
        #    model.cuda()
        #else:
        #    model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()


    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


    
    superclass_mapping = None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True)

    # run inference
    targets, preds = validate(val_loader, model, args)
    # compute metrics
    max_acc_classes, acc_per_class, num_samples_per_class, reassignment = \
        compute_metrics(targets, preds, args.num_samples_per_class, superclass_mapping)

    # extract num_classes with max accuracy
    max_acc_classes = max_acc_classes[:args.num_classes]
    acc_per_class = acc_per_class[:args.num_classes]
    num_samples_per_class = num_samples_per_class[:args.num_classes]

    # sample num_samples_per_class from low entropy classes and save grid image
    for idx_label_i, label_i in enumerate(max_acc_classes):
        # extract all indices of current class
        sample_indices = np.where(preds == label_i)[0]
        # sample randomly num_samples_per_class
        np.random.seed(0)
        subset_sample_indices = np.random.choice(sample_indices, args.num_samples_per_class, replace=False)
        true_pos = targets[subset_sample_indices] == reassignment[label_i]
        #print("subset_sample_indices", subset_sample_indices)
        # get image paths
        subset_img_paths = [val_loader.dataset.imgs[idx][0] for idx in subset_sample_indices]

        #print("subset_img_paths", subset_img_paths)
        # get images
        subset_images = [load_image(x) for x in subset_img_paths]

        # get grid
        grid_i = image_grid(subset_images,
                            int(math.sqrt(args.num_samples_per_class)),
                            int(math.sqrt(args.num_samples_per_class)),
                            targets[subset_sample_indices])

        # save grid
        grid_i.save(os.path.join(args.save_path, 'grid_{}.pdf'.format(idx_label_i)))
        print('=> saved grid_{}.pdf, accuracy = {:.3f}, nsamples = {}'.format(idx_label_i,
                                                                              acc_per_class[idx_label_i],
                                                                              num_samples_per_class[idx_label_i]))
        print('=> grid {} labels: {}'.format(idx_label_i, targets[subset_sample_indices]))


def validate(val_loader, model, args):
    all_preds = np.zeros((len(val_loader.dataset.targets), ), dtype=int)
    all_targets = np.zeros((len(val_loader.dataset.targets), ), dtype=int)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')

    #if args.kmeans_cls is not None:
    #    kmeans = pickle.load(open(args.kmeans_cls, "rb"))

    # switch to evaluate mode
    model.eval()

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()

            # compute output
            #if args.model == 'self-classifier':
            output = model(images).detach().cpu().numpy() #cls_num=args.cls_num
            #else:
            #    output = model(images).detach().cpu().numpy()

            num_samples = output.shape[0]

            # save target
            all_targets[idx: idx + num_samples] = targets.numpy()

            # compute prediction
            #if args.kmeans_cls is not None:
            #    preds_i = kmeans.predict(output.astype(np.float64))
            #else:
            preds_i = output.argmax(1)
            all_preds[idx: idx + num_samples] = preds_i

            idx += num_samples

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return all_targets, all_preds


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


def compute_metrics(targets, preds, min_samples_per_class, superclass_mapping):
    val_nmi = nmi(targets, preds)
    val_adjusted_nmi = adjusted_nmi(targets, preds)
    val_adjusted_rand_index = adjusted_rand_index(targets, preds)
    print('=> number of samples: {}'.format(len(targets)))
    print('=> number of unique assignments: {}'.format(len(set(preds))))
    print('=> NMI: {:.3f}%'.format(val_nmi * 100.0))
    print('=> Adjusted NMI: {:.3f}%'.format(val_adjusted_nmi * 100.0))
    print('=> Adjusted Rand-Index: {:.3f}%'.format(val_adjusted_rand_index * 100.0))

    # compute accuracy
    num_classes = max(targets.max(), preds.max()) + 1
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for ii in range(preds.shape[0]):
        count_matrix[preds[ii], targets[ii]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]

    if len(np.unique(preds)) > len(np.unique(targets)):  # if using over-clustering, append remaining clusters to best option
        for cls_idx in np.unique(preds):
            if reassignment[cls_idx, 1] not in targets:
                reassignment[cls_idx, 1] = count_matrix[cls_idx].argmax()

    if superclass_mapping is not None:
        count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for ii in range(preds.shape[0]):
            count_matrix[preds[ii], superclass_mapping[targets[ii]]] += 1
        for ii in range(len(reassignment[:, 1])):
            reassignment[ii, 1] = superclass_mapping[reassignment[ii, 1]]
    acc = count_matrix[reassignment[:, 0], reassignment[:, 1]].sum().astype(np.float32) / preds.shape[0]
    print('=> Accuracy: {:.3f}%'.format(acc * 100.0))

    # extract max accuracy classes
    num_samples_per_class = count_matrix[reassignment[:, 0], :].sum(axis=1)
    acc_per_class = np.where(num_samples_per_class >= min_samples_per_class,
                             count_matrix[reassignment[:, 0], reassignment[:, 1]] / num_samples_per_class, 0)
    max_acc_classes = np.argsort(acc_per_class)[::-1]
    acc_per_class = acc_per_class[max_acc_classes]
    num_samples_per_class = num_samples_per_class[max_acc_classes]

    return max_acc_classes, acc_per_class, num_samples_per_class, reassignment[:, 1]


def load_image(infilename):
    img = Image.open(infilename)
    img = img.resize((128, 128))
    return img


def image_grid(imgs, rows, cols, label, eps=15, border=3):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w + (cols+1) * eps, rows * h + (rows+1) * eps), color=(255, 255, 255))
    # grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        # img_with_border = ImageOps.expand(img, border=border, fill='green' if true_pos[i] else 'red')
        img_with_border = ImageOps.expand(img, border=border, fill='black')
        # draw = ImageDraw.Draw(img_with_border)
        # txt_w, txt_h = draw.textsize(str(label[i]))
        # draw.text(((w - txt_w) / 2, (h - txt_h) / 2), str(label[i]), fill="black")
        grid.paste(img_with_border, box=(eps - border + i % cols * (w + eps), eps - border + i // cols * (h + eps)))
    return grid


if __name__ == '__main__':
    main()