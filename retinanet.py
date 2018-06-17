import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from resnet_features import resnet50_features
from utils.layers import conv1x1, conv3x3


class FeaturePyramid(nn.Module):
    """
    Feature Pyramid Network, described in https://arxiv.org/abs/1612.03144.
    """

    def __init__(self, resnet):

        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.proj3 = conv1x1(512, 256)
        self.proj4 = conv1x1(1024, 256)
        self.proj5 = conv1x1(2048, 256)

        # both based around c5
        self.proj6 = conv3x3(2048, 256, padding=1, stride=2)
        self.proj7 = conv3x3(256, 256, padding=1, stride=2)

        # applied after upsampling
        self.upsample1 = conv3x3(256, 256, padding=1)
        self.upsample2 = conv3x3(256, 256, padding=1)


    # TODO: verify if this function can be replaced by simple bilinear upsample
    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :height, :width]

    def forward(self, x):
        # skip c2 features since it's too large
        _, c3, c4, c5 = self.resnet(x)

        # feature pyramid 6,7
        fp6 = self.proj6(c5)
        fp7 = self.proj7(F.relu(fp6))

        # feature pyramid 5
        fp5 = self.proj5(c5)

        # feature pyramid 4
        fp4 = self.proj4(c4)
        up5 = self._upsample(fp5, fp4)
        fp4 = self.upsample1(
            torch.add(up5, fp4)
        )

        # feature pyramid 3
        fp3 = self.proj3(c3)
        up4 = self._upsample(fp4, fp3)
        fp3 = self.upsample2(
            torch.add(up4, fp3)
        )

        return (fp3, fp4, fp5, fp6, fp7)


class SubNet(nn.Module):
    """
    Subnet, described in Focal Loss for Dense Object Detection.
    ref: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, mode, anchors, classes, depth,
                 base_activation,
                 output_activation):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation

        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1)
                                          for _ in range(depth)])

        if mode == 'boxes':
            self.subnet_output = conv3x3(256, 4 * self.anchors, padding=1)
        elif mode == 'classes':
            # an extra dim for confidence
            self.subnet_output = conv3x3(256, (1 + self.classes) * self.anchors, padding=1)
            self._output_layer_init(self.subnet_output.bias.data)
        else:
            raise ValueError("Invalid mode {}".format(mode))


    def _output_layer_init(self, tensor, pi=0.01):

        assert isinstance(tensor, Variable), \
            'input {} must be variable'.format(tensor)
        fill_constant = - math.log((1 - pi) / pi)

        return tensor.fill_(fill_constant)


    def forward(self, x):

        for layer in self.subnet_base:
            x = self.base_activation(layer(x))

        x = self.subnet_output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0),
                                                    x.size(2) * x.size(3) * self.anchors, -1)

        return x


class RetinaNet(nn.Module):
    """
    Retinanet, described in Focal Loss for Dense Object Detection.
    ref: https://arxiv.org/abs/1708.02002
    """
    def __init__(self,
                 classes,
                 backbone, 
                 pretrained=True,
                 n_anchors=9,
                 depth=4,
                 base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(RetinaNet, self).__init__()
        self.classes = classes
        self.fpn = FeaturePyramid(backbone(pretrained=pretrained))
        self.subnet_box = SubNet(mode='boxes',
                                 anchors=n_anchors,
                                 classes=classes,
                                 depth=depth,
                                 base_activation=base_activation,
                                 output_activation=output_activation)
        self.subnet_box = SubNet(mode='classes',
                                 anchors=n_anchors,
                                 classes=classes,
                                 depth=depth,
                                 base_activation=base_activation,
                                 output_activation=output_activation)

    def forward(self, x):

        fp_list = self.fpn(x)
        boxes = torch.cat([self.subnet_box(fp) for fp in fp_list])
        classes = torch.cat([self.subnet_cls(fp) for fp in fp_list])

        # TODO: verify if the boxes need to be transformed

        return boxes, classes

