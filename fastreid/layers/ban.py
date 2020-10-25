from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter


class _BatchAttNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchAttNorm, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * attention

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn


class BAN2d(_BatchAttNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
