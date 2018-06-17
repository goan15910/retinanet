
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class TransformAnchors(nn.Module):
    """
    Generate transformed anchors, described in 
    Faster-RCNN / FPN / RetinaNet
    ref: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, scales=None, aspects=None):
        super(Anchor, self).__init__()
        if scales is None:
            self.scales = [1., 1.23, 1.51] # 2^0, 2^(1/3), 2^(2/3)
        else:
            self.scales = scales

        if aspects is None:
            self.aspects = [1., 0.5, 2.] # w to h
        else:
            self.aspects = aspects


    def forward(self, box_preds):
        _, _, h, w = inputT.size()
        boxes = []
        for scale in self.scales:
            for aspect in self.aspects:
                # TODO: transform box according to rules


class FilterAnchors(nn.Module):
    """
    Filter out anchors follow rules in RetinaNet
    ref: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, 
                 n_anchors=1000,
                 conf_thres=0.05,
                 nms_thres=0.5):
        super(FilterAnchors, self).__init__()
        self.n_anchors = n_anchors # number per pyramid
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres


    def forward(self, box_pred, cls_pred):
        # TODO: following below steps in order
        # 1. filter by conf
        # 2. filter by n_anchors
        # 3. filter using NMS
        pass
        

class AssignAnchors(nn.Module):
    """
    Assign label to anchors / ignore anchors
    ref: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, h_thres=0.5, l_thres=0.4):
        super(AssignAnchors, self).__init__()
        assert h_thres >= l_thres, \
            "h_thres {} must be no less than l_thres {}" \
                .format(h_thres, l_thres)
        self.h_thres = h_thres
        self.l_thres = l_thres


    def forward(self, inputT, target):
        # TODO:
        # 1. check the IOU of box_preds and box_targets
        # 2. if IOU in [0.5, 1], set as foreground
        # 3. if IOU in [0, 0.4), set as background
        # 4. if IOU in [0.4, 0.5), ignore it
        pass
