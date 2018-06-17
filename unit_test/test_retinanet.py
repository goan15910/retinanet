import time
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from retinanet import FeaturePyramid, RetinaNet


# TODO
def test_fpn():
    """test feature-pyramid-network"""
    pass


# TODO
def test_subnet():
    """test subnet"""
    pass


def test_retinanet():
    """test retinanet"""
    net = RetinaNet(classes=80)
    x = Variable(torch.rand(1, 3, 500, 500), volatile=True)

    now = time.time()
    net.cuda()
    predictions = net(x)
    later = time.time()

    print(later - now)

    for prediction in predictions:
        print(prediction.size())
