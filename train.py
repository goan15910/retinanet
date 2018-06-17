import torch
import argparse

import torch.nn as nn

from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler

from losses import FocalLoss, SmoothedL1Loss
from retinanet import RetinaNet
from datasets.coco import CocoDetection
from resnet_features import resnet18_features, resnet34_features, \
    resnet50_features, resnet101_features, resnet152_features


parser = argparse.ArgumentParser("Train RetinaNet on dataset")
parser.add_argument('dataset', default='coco2017', help='name of dataset')
parser.add_argument('root_dir', default='../COCO/train2017', help='root directory of dataset')
parser.add_argument('--backbone', default='resnet50', help='backbone network')
parser.add_argument('--coco_anno', default='../COCO/annotations/instances_train2017.json', help='annotation json file for COCO')
parser.add_argument('--batch_size', default=16, help='batch size')
args = parser.parse_args()


# dataset
batch_size = args.batch_size
if args.dataset == 'coco2017':
    train_loader = torch.utils.data.DataLoader(
        CocoDetection(
            root=args.root_dir,
            annFile=args.coco_anno,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # normalization for imagenet-pretrained
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True)
elif args.dataset == 'voc2012':
    raise NotImplementedError
elif args.dataset == 'tmall':
    raise NotImplementedError
else:
    raise ValueError('Invalid dataset')


# model backbone
if args.backbone == 'resnet18':
    backbone = resnet18_features
elif args.backbone == 'resnet34':
    backbone = resnet34_features
elif args.backbone == 'resnet50':
    backbone = resnet50_features
elif args.backbone == 'resnet101':
    backbone = resnet101_features
elif args.backbone == 'resnet152':
    backbone = resnet152_features
else:
    raise ValueError('Invalid model backbone')


# RetinaNet model
model = RetinaNet(80, backbone, pretrained=True)

optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0001)

# TODO: enable more flexible lr profile
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     # Milestones are set assuming batch size is 16:
                                     # 60000 / batch_size = 3750
                                     # 80000 / batch_size = 5000
                                     milestones=[3750, 5000],
                                     gamma=0.1)


focal_loss = FocalLoss(80)
smoothed_l1_loss = SmoothedL1Loss(3.)


def train(model, cuda=True):

    if cuda:
        model.cuda()
        model = nn.DataParallel(model)
        focal_loss.cuda()
        smoothed_l1_loss.cuda()

    for batch_id, (images, box_targets, cls_targets) in enumerate(train_loader):

        if cuda:
            images.cuda()
            box_targets.cuda()
            cls_targets.cuda()

        # lr for this step
        scheduler.step()

        optimizer.zero_grad()

        images = Variable(images)
        box_preds, cls_preds = model(images)
        
        # TODO: 
        # 1. verify the assignment of cls and box to proposals
        # 2. normalize losses according to the number of 
        #    anchor boxes assigned to a ground-truth box
        cls_loss = focal_loss(cls_preds, cls_targets)
        loc_loss = smoothed_l1_loss(box_preds, box_targets)
        
        total_loss = cls_loss + loc_loss
        total_loss.backward()

        optimizer.step()

        print 'Batch: {0}, Class loss: {1}, Box loss: {2}, Total Loss: {3}' \
                 .format(batch_id, 
                         cls_loss / (current_batch+1),
                         box_loss / (current_batch+1),
                         total_loss / (current_batch+1))


if __name__ == '__main__':
    train(model)
