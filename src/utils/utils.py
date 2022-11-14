
import os
import warnings
import numpy as np
import torch as th
import random
import math
import urllib.request
import torch.distributed as dist
from torchvision import transforms, datasets




class AllGather(th.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [th.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return th.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads



def keep_current(tensor):
    if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
        s = (tensor.shape[0] // dist.get_world_size()) * dist.get_rank()
        e = (tensor.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
        return tensor[s:e]
    return tensor
