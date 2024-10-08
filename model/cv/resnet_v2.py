'''ResNet in Paddle.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from copy import deepcopy
import logging

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from utils.tool import *
import paddle.vision.transforms as transforms


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion * planes, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10, args=None, image_size=32, model_input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.args = args
        self.image_size = image_size

        if args.dataset == 'fmnist':
            self.init_bn = nn.BatchNorm2D(1)
        else:
            self.init_bn = nn.BatchNorm2D(3)
        self.conv1 = nn.Conv2D(model_input_channels, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.layers_name_map = {
            "classifier": "linear"
        }

        inplanes = [64, 64, 128, 256, 512]
        inplanes = [inplane * block.expansion for inplane in inplanes]
        logging.info(inplanes)

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_bn(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.linear(out.reshape([out.shape[0], -1]))

        return out

def ResNet10(args, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, args=args, **kwargs)


def ResNet18(args, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, args=args, **kwargs)


def ResNet34(args, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, args=args, **kwargs)


def ResNet50(args, num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, args=args, **kwargs)


def ResNet101(args, num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, args=args, **kwargs)


def ResNet152(args, num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, args=args, **kwargs)

