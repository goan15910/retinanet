import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Focal loss in focal loss for dense object detection.
    ref: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, inputT, target):
        logpt = - F.cross_entropy(inputT, target)
        pt    = torch.exp(logpt)

        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt

        return focal_loss


class SmoothedL1Loss(nn.Module):
    """
    Smoothed L1 Loss, proposed in Fast-RCNN.
    ref: https://arxiv.org/abs/1504.08083
    """
    # TODO: verify sigma of smoothed-L1-loss
    # default value 3. is from setting of keras-retinanet
    def __init__(self, sigma=3.):
        super(SmoothedL1Loss, self).__init__()
        self.sigma_sqr = sigma ** 2.
        self.thres = 1. / self.sigma_sqr
    

    def forward(self, inputT, target):
        diff = torch.abs(inputT - target)
        if diff < self.thres:
            smoothed_l1 = 0.5 * self.sigma_sqr * diff
        else:
            smoothed_l1 = (diff - 0.5) / self.sigma_sqr

        return smoothed_l1

