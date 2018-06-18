
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utils.images import iou, box2param, param2box
from utils.cython_bbox import bbox_overlaps


class FilterPreds(nn.Module):
  """
  Filter aug_preds following rules in RetinaNet.
  ref: https://arxiv.org/abs/1708.02002
  """
  def __init__(self,
               n_anchors=9,
               n_top=1000,
               conf_thres=0.05,
               nms_thres=0.5):
    super(FilterPreds, self).__init__()
    self.n_anchors = n_anchors
    self.n_top = n_top
    self.conf_thres = conf_thres
    self.nms_thres = nms_thres


  def forward(self, regs, cls):
    # reshape regs, cls
    A = self.n_anchors
    N, _, H, W = cls.size()
    confs = cls.view([N, -1, A, H, W])[:, 0] \
               .permute(0, 2, 3, 1) \
               .view([N, H*W*A]) # (N, H*W*A)
    regs = regs.permute(0, 2, 3, 1) \
               .view([N, H, W, A, -1]) \
               .view([N, H*W*A, -1]) # (N, H*W*A, 4)

    # TODO:
    # extract confs, and sorted indices
    _, indices = torch.sort(confs, dim=-1, descending=True)

    # top-k
    top_k_indices = indices[:, :n_top]

    # filter by confs
    conf_mask = conf[top_k_indices] > self.conf_thres

    # TODO:
    # transform top_k_indices to anchor mapping

    return regs[top_k_indices], conf_mask, top_k_indices


class AssignRegGt(nn.Module):
  """
  Assign regression target / ignore regression
  ref: https://arxiv.org/abs/1708.02002
  """
  def __init__(self, h_thres=0.5, l_thres=0.4):
    super(AssignRegGt, self).__init__()
    assert h_thres >= l_thres, \
      "h_thres {} must be no less than l_thres {}" \
        .format(h_thres, l_thres)
    self.h_thres = h_thres
    self.l_thres = l_thres


  def assign_labels_to_boxes(self, ):


  def forward(self, regs, targets, conf_mask, top_k_indices):
    # TODO:
    # 1. check the IOU of box_preds and box_targets
    # 2. if IOU in [0.5, 1], set as foreground
    # 3. if IOU in [0, 0.4), set as background
    # 4. if IOU in [0.4, 0.5), ignore it
    


class GenerateBoxes(nn.Module):
  """
  Generate boxes from anchors, 
  described in Faster-RCNN / FPN / RetinaNet
  ref: https://arxiv.org/abs/1708.02002
  """
  def __init__(self, scales=None, aspects=None):
    super(GenerateBoxes, self).__init__()
    if scales is None:
      self.scales = [1., 1.23, 1.51] # 2^0, 2^(1/3), 2^(2/3)
    else:
      self.scales = scales

    if aspects is None:
      self.aspects = [1., 0.5, 2.] # w to h
    else:
      self.aspects = aspects

    n_anchors = len(self.aspects) * len(self.scales)
    self.anchors = torch.zeros([n_anchors, 4], dtype=torch.float32)
    for scale in enumerate(self.scales):
      for aspects in enumerate(self.aspects):
        self.anchors[]


  def forward(self, regs, mask, indices):
    """
    Args:
      regs: top regression result
      mask: if it is confident enough to output
      indices: original indices before filtering
    """
    # transform indices to anchor box indices
    # TODO

    # transform to boxes
    boxes = torch.zeros(regs.size(), dtype=torch.float32)
    for scale in self.scales:
      for aspect in self.aspects:
        boxes[:, 0] = box_preds[:, 0]*