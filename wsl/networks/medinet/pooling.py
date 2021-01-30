#!/usr/bin/python3
import sys
import random
import torch
import torch.nn as nn
from torch.autograd import Function


class WildcatPool2dFunction(Function):

    @staticmethod
    def get_positive_k(alpha, n):
        if alpha <= 0:
            return 0
        elif alpha < 1:
            return round(alpha * n)
        elif alpha > n:
            return int(n)
        else:
            return int(alpha)

    @staticmethod
    def forward(ctx, input, args):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions
        kmax = WildcatPool2dFunction.get_positive_k(args, n)
        ctx.k = random.randint(1, kmax + 1)

        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

        ctx.indices_max = indices.narrow(2, 0, ctx.k)
        output = sorted.narrow(2, 0, ctx.k).sum(2).div_(ctx.k)

        ctx.save_for_backward(input)
        return output.view(batch_size, num_channels)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, ctx.k)
        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, ctx.indices_max, grad_output_max).div_(ctx.k)

        return grad_input.view(batch_size, num_channels, h, w), None


class WildcatPool2d(nn.Module):
    def __init__(self, alpha=0.01):
        super(WildcatPool2d, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction.apply(input, self.alpha)

    def __repr__(self):
        return self.__class__.__name__ + f' (alpha={self.alpha})'


class ClassWisePoolFunction(Function):

    @staticmethod
    def forward(ctx, input, args):
        ctx.num_maps = args
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % ctx.num_maps != 0:
            print('Error in ClassWisePoolFunction. Number of channels is not multiple of number of maps.')
            sys.exit(-1)

        num_outputs = int(num_channels / ctx.num_maps)
        x = input.view(batch_size, num_outputs, ctx.num_maps, h, w)
        output = torch.sum(x, 2)
        ctx.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / ctx.num_maps

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_output = grad_output.view(batch_size, num_outputs, 1, h, w)
        grad_input = grad_output.expand(batch_size, num_outputs, ctx.num_maps, h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w), None


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction.apply(input, self.num_maps)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)
