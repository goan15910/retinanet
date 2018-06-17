import sys
sys.append('.')

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from losses import FocalLoss, SmoothedL1Loss


def test_focal_loss():
    loss = FocalLoss()

    inputT = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))

    print inputT
    print target

    output = loss(inputT, target)
    print(output)
    output.backward()


# TODO
def test_smoothed_l1_loss():
    raise NotImplementedError
