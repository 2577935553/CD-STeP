import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SelfAtten', 'CrossAtten_spatial', 'CrossAtten_channel', 'Perceptron', 'SpatialGate', 'ChannelAtten']


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-1).unsqueeze(-1)


class ChannelAtten(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=256, pool_types=['avg', 'max'], dropout=0.2):
        super(ChannelAtten, self).__init__()
        self.gate_channels = gate_channels

        self.mlp = nn.Sequential(
            # Flatten(),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            # nn.Linear(gate_channels // reduction_ratio, gate_channels),
            # UnFlatten()
        )

        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        return scale * channel_att_sum


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ConvBlock(gate_channels, gate_channels // reduction_ratio, kernel_size=3, stride=1, padding=1, relu=True)
        # self.spatial = nn.Sequential(
        #     ConvBlock(2, 1, kernel_size=3, stride=1, padding=1, relu=True),
        #     ConvBlock(1, 1, kernel_size=3, stride=1, padding=1, relu=True),
        #     ConvBlock(1, 1, kernel_size=3, stride=1, padding=1, relu=False)
        # )

    def forward(self, x):
        x_compress = self.compress(x)
        # x_out = self.spatial(x_compress)
        # scale = F.sigmoid(x_compress) * x_compress  # broadcasting
        return x_compress


class SelfAtten(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=256, pool_types=['avg', 'max']):
        super(SelfAtten, self).__init__()
        self.ChannelGate = ChannelAtten(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(gate_channels, reduction_ratio)

    def forward(self, x):
        x_chann = self.ChannelGate(x)

        x_spatial = self.SpatialGate(x)

        return x_chann, x_spatial


class CrossAtten_channel(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=64, dropout=0.2):
        super(CrossAtten_channel, self).__init__()

        self.mlp = nn.Sequential(
            # Flatten(),
            nn.Conv2d(gate_channels, gate_channels, 1, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            # UnFlatten()
        )
        self.conv1 = ConvBlock(gate_channels, gate_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(gate_channels, gate_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(gate_channels, gate_channels, kernel_size=1, stride=1, padding=0, relu=False)

    def forward(self, x_chann, x_spatial):
        x_q = self.mlp(x_chann)
        x_k = self.conv1(x_spatial)
        x_qtk = F.softmax(x_q * x_k, dim=1)

        x_v = self.conv2(x_spatial)
        x_atten = self.conv3(x_qtk * x_v + x_spatial)
        return x_atten


class CrossAtten_spatial(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=64, dropout=0.2):
        super(CrossAtten_spatial, self).__init__()
        self.mlp1 = nn.Sequential(
            # Flatten(),
            nn.Conv2d(gate_channels, gate_channels, 1, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            # UnFlatten()
        )
        self.mlp2 = nn.Sequential(

            nn.Conv2d(gate_channels, gate_channels, 1, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),

        )
        self.conv1 = ConvBlock(gate_channels, gate_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(gate_channels, gate_channels, kernel_size=1, stride=1, padding=0, relu=False)

    def forward(self, x_chann, x_spatial):
        x_q = self.conv1(x_spatial)
        x_k = self.mlp1(x_chann)
        x_qtk = F.softmax(x_q * x_k, dim=1)

        x_v = self.mlp2(x_chann)
        x_atten = self.conv2(x_v * x_qtk + x_chann)
        return x_atten

class Perceptron(nn.Module):
    def __init__(self, in_planes, frame_number=20, reduction_ratio=64, pooling='avg', dropout=0.2):
        super(Perceptron, self).__init__()
        self.Pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_planes, frame_number),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x_chann, x_spatial):
        x_atten = self.Pool(x_chann * x_spatial)
        x_weight = self.mlp(x_atten)

        return self.activation(x_weight)


if __name__ == '__main__':
    x = torch.rand(2, 2048 * 20, 14, 14)
    selfatt = SelfAtten(2048 * 20)
    x_chann, x_spatial = selfatt(x)
    perceiver = Perceptron(2048 * 20)
    x_weight = perceiver(x_chann, x_spatial)
    print(x_weight.sum(1))
