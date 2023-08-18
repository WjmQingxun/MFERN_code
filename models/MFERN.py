from utils import HPDM
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce

# ========= 多尺度空间特征提取模块（Multi-scale spatial feature extraction module）【MSFE module】 ============
class IGCModule(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, module, s):
        super(IGCModule, self).__init__()
        # 自实验
        self.s = s
        self.inchannel = in_channels
        self.width = int(math.floor((in_channels) / s))
        convs = []
        bns = []
        self.number = (s * s) - (3 * s) + 3
        self.short_num = int(((s * s) - (3 * s) + 4) / 2)
        self.M = s - 2
        for i in range(self.number):  # 3
            # 输入通道等于输出通道的卷积操作(3*3)
            if i <= (self.short_num - 1):
                convs.append(
                    nn.Conv2d(self.width, self.width, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
                bns.append(nn.BatchNorm2d(self.width))
            else:
                convs.append(nn.Conv2d(self.width * self.M, self.width * self.M, kernel_size=kernel_size, stride=1,
                                       padding=padding, bias=False))
                bns.append(nn.BatchNorm2d(self.width * self.M))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        d = int(self.width)  # 计算向量Z 的长度d

        self.out_channel = self.width * self.M

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(self.out_channel, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, self.out_channel * self.M, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

        self.relu = nn.ReLU(inplace=True)

        self.module = module

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        sp1 = spx[0]
        sp2 = spx[1]
        sp2 = self.convs[0](sp2)
        sp2 = self.relu(self.bns[0](sp2))

        if self.s > 2:
            for i in range(self.s - 2):
                if i == 0:
                    sp = self.convs[i + 1](spx[2 + i])
                    sp = self.relu(self.bns[i + 1](sp))
                else:
                    for j in range(i + 1):
                        if j == 0:
                            s1 = self.convs[i + 1 + j](spx[2 + i])
                            s1 = self.relu(self.bns[i + 1 + j](s1))
                        else:
                            s1 = self.convs[i + 1 + j](s1)
                            s1 = self.relu(self.bns[i + 1 + j](s1))
                    sp = torch.cat((sp, s1), 1)

            spp = []
            batch_size = x.size(0)
            outputs = []
            for i in range(self.s - 2):
                spp.append(sp)

            for i in range(self.s - 2):
                for j in range(i + 1):
                    if j == 0:
                        x1 = self.convs[i + self.short_num + j](spp[i])
                        x1 = self.relu(self.bns[i + self.short_num + j](x1))
                    else:
                        x1 = self.convs[i + self.short_num + j](x1)
                        x1 = self.relu(self.bns[i + self.short_num + j](x1))
                outputs.append(x1)


            U1 = reduce(lambda x, y: x + y, outputs)  # 逐元素相加生成 混合特征U
            s1 = self.global_pool(U1)
            z1 = self.fc1(s1)  # S->Z降维
            a_b1 = self.fc2(z1)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
            a_b1 = a_b1.reshape(batch_size, self.M, self.out_channel, -1)  # 调整形状，变为 两个全连接层的值
            a_b1 = self.softmax(a_b1)  # 使得两个全连接层对应位置进行softmax
            # the part of selection
            a_b1 = list(a_b1.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
            a_b1 = list(map(lambda x: x.reshape(batch_size, self.out_channel, 1, 1), a_b1))  # 将所有分块  调整形状，即扩展两维
            V1 = list(map(lambda x, y: x * y, outputs, a_b1))  # 权重与对应  不同卷积核输出的U 逐元素相乘
            V1 = reduce(lambda x, y: x + y, V1)  # 两个加权后的特征 逐元素相加

            if ((self.s * self.width) != self.inchannel):
                out = torch.cat((sp1, sp2, V1, spx[self.s]), 1)
            else:
                out = torch.cat((sp1, sp2, V1), 1)

        else: #self.s <= 2
            if ((self.s * self.width) != self.inchannel):
                out = torch.cat((sp1, sp2, spx[self.s]), 1)
            else:
                out = torch.cat((sp1, sp2), 1)

        return F.relu(out)


# ================= 组卷积 ======================
class IGC_G(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0, groups=1, module='', s=4):
        super(IGC_G, self).__init__()
        self.bands = int(in_channels / groups)
        IGC = []
        for i in range(groups):  # PA:5    IN、SA:11、11
            IGC.append(IGCModule(self.bands, kernel_size=kernel_size, padding=padding, module=module, s=s))
        self.IGC = nn.ModuleList(IGC)

        self.group = groups

    def forward(self, X):
        spx = torch.split(X, self.bands, 1)

        for i in range(self.group):
            if i == 0:
                spp = self.IGC[i](spx[i])
                out = spp
            else:
                spp = self.IGC[i](spx[i])
                out = torch.cat((out, spp), 1)

        out = F.relu(out)
        return out


# ================= 局部分支 ======================
class Res_1(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0, module='', groups=1, s=4):
        super(Res_1, self).__init__()
        self.IGC_G1 = IGC_G(in_channels, kernel_size=kernel_size, padding=padding, module=module, groups=groups, s=s)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.IGC_G1(X)

        return F.relu(X)


# ================= 全局分支 ======================
class Res_2(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, module='', s=4):
        super(Res_2, self).__init__()
        self.IGC_1 = IGCModule(in_channels, kernel_size=kernel_size, padding=padding, module=module, s=s)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.IGC_1(X)

        return F.relu(X)


# ================= 整个残差块 ======================
class Res(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, groups, module='', s=4):
        super(Res, self).__init__()
        self.res1 = Res_1(in_channels, kernel_size=(3, 3), padding=(1, 1), module=module, groups=groups, s=s)
        self.res2 = Res_2(in_channels, kernel_size=(3, 3), padding=(1, 1), module=module, s=s)

    def forward(self, X):
        Y = self.res1(X)
        Z = self.res2(X)

        return F.relu(X + Y + Z)


# ================= 模型 ======================
class mfern(nn.Module):
    def __init__(self, bands, classes, groups, groups_width, mod='', s=4):
        super(mfern, self).__init__()
        self.bands = bands
        self.classes = classes
        fc_planes = 128

        # pad the bands with final values
        new_bands = math.ceil(bands / groups) * groups
        pad_size = new_bands - bands
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))

        # SPRN
        self.conv1 = nn.Conv2d(new_bands, groups * groups_width, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(groups * groups_width)

        self.res0 = Res(groups * groups_width, kernel_size=1, padding=0, groups=groups, module=mod, s=s)
        self.res1 = Res(groups * groups_width, kernel_size=1, padding=0, groups=groups, module=mod, s=s)

        self.conv2 = nn.Conv2d(groups_width * groups, fc_planes, (1, 1))
        self.bn2 = nn.BatchNorm2d(fc_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_fc = nn.Linear(fc_planes, classes)

    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.pad(x).squeeze(1)

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.res0(x)
        x = self.res1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        x = self.out_fc(x)
        return x



def MFERN(dataset, bands, classes):
    if dataset == 'PaviaU': #32
        model = mfern(bands, classes, groups=5, groups_width=32, mod=dataset, s=4)
    elif dataset == 'IndianPines': #32
        model = mfern(bands, classes, groups=9, groups_width=32, mod=dataset, s=3)
    elif dataset == 'Salinas':  #27
        model = mfern(bands, classes, groups=11, groups_width=27, mod=dataset, s=4)
    return model

