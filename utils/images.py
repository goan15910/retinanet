
import torch

from torchvision.transforms import Compose, Normalize, Scale, ToTensor

# TODO
# integrate this function into AssignAnchors
def box2param(box, anchor):
  param = torch.zeros([4, 1])
  param[0] = (box[0] - anchor[0]) / anchor[2]
  param[1] = (box[1] - anchor[1]) / anchor[3]
  param[2] = torch.log(box[2] / anchor[2])
  param[3] = torch.log(box[3] / anchor[3])
  return param


#TODO
# integrate this function into TransformAnchors
def param2box(param, anchor):
  box = torch.zeros([4, 1])
  box[0] = param[0] * anchor[2] + anchor[0]
  box[1] = param[1] * anchor[3] + anchor[1]
  box[2] = torch.exp(param[2]) * anchor[2]
  box[3] = torch.exp(param[3]) * anchor[3]
  return box


def iou(param, box_target, anchor):
  """Calculate IoU from param & box_target"""
  box = param2box(param, anchor)
  min_x = max(box[0]-box[2]/2, box_target[0]-box_target[2]/2)
  max_x = min(box[0]+box[2]/2, box_target[0]+box_target[2]/2)
  min_y = max(box[1]-box[3]/2, box_target[1]-box_target[3]/2)
  max_y = min(box[1]+box[3]/2, box_target[1]+box_target[3]/2)
  intersect_area = (max_x - min_x) * (max_y - min_y)
  box_area = box[2] * box[3]
  box_target_area = box_target[2] * box_target[3]
  union_area = box_area + box_target_area - intersect_area
  return float(intersect_area) / (union_area)


def iou_matrix(params, targets):