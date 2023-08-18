# # Model
import torch
import torch.nn as nn
import math
# from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

############## Transformer #############

class MultiHeadDense(nn.Module):
    def __init__(self, d):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))

    def forward(self, x):
        # x:[b, h*w, d]
        # x = torch.bmm(x, self.weight)
        x = F.linear(x, self.weight)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel)
        self.key = MultiHeadDense(channel)
        self.value = MultiHeadDense(channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.positional_encoding_2d(c, h, w)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))  # [b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x


########################################


class HybridSN_network(nn.Module):
    def __init__(self, datasets, band, classes):
        super(HybridSN_network, self).__init__()
        self.datasets = datasets
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=2912,
                out_channels=64,
                kernel_size=(3, 3)),
            nn.ReLU(inplace=True))

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=6144,
                out_channels=64,
                kernel_size=(3, 3)),
            nn.ReLU(inplace=True))

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6016,
                out_channels=64,
                kernel_size=(3, 3)),
            nn.ReLU(inplace=True))

        self.MHSA = MultiHeadSelfAttention(64)

        self.dense1 = nn.Sequential(
            nn.Linear(18496, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense3 = nn.Sequential(
            nn.Linear(128, classes)
        )

    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        #print(x.shape)
        if self.datasets == 'PaviaU':
            x = self.conv4(x)
        elif self.datasets == 'Salinas':
            x = self.conv4_1(x)
        elif self.datasets == 'IndianPines':
            x = self.conv4_2(x)

        #         print(x.shape)
        # x = self.MHSA(x)
        #         print(x.shape)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)

        return out

def HybridSN(datasets, n_bands, num_classes):
    model = HybridSN_network(datasets, n_bands, num_classes)
    return model