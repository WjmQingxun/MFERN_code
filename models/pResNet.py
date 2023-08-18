import torch
import torch.nn as nn
import math
# from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def make_conv_bn_relu(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=1):
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_linear_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_max_flat(out):
    flat = F.adaptive_max_pool2d(
        out, output_size=1)
    flat = flat.view(flat.size(0), -1)
    return flat


def make_avg_flat(out):
    flat = F.adaptive_avg_pool2d(out, output_size=1)
    flat = flat.view(flat.size(0), -1)
    return flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PyResNet(nn.Module):
    def __init__(self, block, layers, n_bands, num_classes):
        self.inplanes = 64

        super(PyResNet, self).__init__()
        in_channels = n_bands

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.fc2 = nn.Sequential(
            *make_linear_bn_relu(128 * block.expansion, 512),
            nn.Linear(512, num_classes),
        )
        self.fc3 = nn.Sequential(
            *make_linear_bn_relu(256 * block.expansion, 512),
            nn.Linear(512, num_classes),
        )
        self.fc4 = nn.Sequential(
            *make_linear_bn_relu(512 * block.expansion, 512),
            nn.Linear(512, num_classes),
        )

        # self.fc = nn.Sequential(
        #     *make_linear_bn_relu((128+256+512) * block.expansion, 1024),
        #     nn.Linear(1024, num_classes)
        # )
        #

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)  # 64, 64x64

        x = self.layer2(x)  #128, 32x32
        flat2 = make_max_flat(x)  ##make_avg_flat

        x = self.layer3(x)  #256, 16x16
        flat3 = make_max_flat(x)

        x = self.layer4(x)  #512,  8x8
        flat4 = make_max_flat(x)

        x = self.fc2(flat2) + self.fc3(flat3) + self.fc4(flat4)

        logit = x
        return logit


def PResNet(n_bands, num_classes):
    model = PyResNet(BasicBlock, [3, 4, 6, 3], n_bands, num_classes)
    return model