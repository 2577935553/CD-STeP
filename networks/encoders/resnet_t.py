import functools

import torch
import torch.nn as nn
# from utils import load_model
import math
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock


__all__ = ['ResNet_t', 'resnet18_t', 'resnet34_t', 'resnet50_t', 'resnet101_t',
           'resnet152_t']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out


class FourierPositionalEncoding(nn.Module):
    """
    源自您代码的傅里叶位置编码，设计优良，予以保留。
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

class SpatioTemporalAttentionBlock(nn.Module):
    """
    一个集成了局部空间注意力和全局时间注意力的模块。
    
    工作流程:
    1.  **Spatial Attention (per frame)**: 对时间序列中的每一帧，使用Swin Transformer Block
        进行局部空间信息交互。这使得每个像素的特征都包含了其空间邻域的上下文。
    2.  **Temporal Attention (pixel-wise)**: 将经过空间增强的特征序列在时间维度上
        进行交互，捕捉每个空间位置的时间动态。
        
    输入: (B, T, C, H, W)
    输出: 
        - temporal_features: (B, T, C, H, W) - 每个时间步的时空交互特征
        - global_feature: (B, 1, C, H, W) - 全局时间上下文特征
    """
    def __init__(self, dim, num_heads,input_resolution, window_size=7, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        
        # 1. 空间注意力模块 (使用timm的SwinTransformerBlock)
        # 注意: Swin Block的输入需要是 (Batch, SeqLen, Channels)
        # self.spatial_attention = SwinTransformerBlock(
        #     dim=dim,
        #     input_resolution=input_resolution,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=True,
        #     proj_drop=dropout,
        #     attn_drop=dropout,
        #     drop_path=dropout, # 通常在堆叠多个块时使用drop_path，这里为保持接口一致
        # )
        self.spatial_attention=nn.Identity()

        self.context_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_encoding = FourierPositionalEncoding(dim)
        
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 非常重要！使得输入可以是 (Batch, SeqLen, Channels)
        )

    def forward(self, x,T):
        BT,C,H,W=x.shape
        x=x.view(BT//T,T,C,H,W)
        B=BT//T
        x_spatial = x.view(B * T, C, H, W)

        x_spatial = x_spatial.permute(0,2,3,1)  # (B*T, H,W, C)
        x_spatial = self.spatial_attention(x_spatial) # Swin v2 需要传入分辨率
        print(x_spatial.shape)

        x_spatial_enhanced = x_spatial.permute(0,3,1,2).view(B, T, C, H, W)

        # (B, T, C, H, W) -> (B, T, C, H*W) -> (B, H*W, T, C) -> (B*H*W, T, C)
        x_temporal = x_spatial_enhanced.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        
        # context_token shape: (1, 1, C) -> 扩展到 (B*H*W, 1, C)
        context = self.context_token.expand(B * H * W, -1, -1)
        x_with_context = torch.cat([x_temporal, context], dim=1) # (B*H*W, T+1, C)
        
        # 添加位置编码
        pos_enc = self.pos_encoding(T + 1) # (1, T+1, C)
        x_with_context = x_with_context + pos_enc
        
        # 应用时间注意力
        x_temporal_enhanced = self.temporal_attention(x_with_context) # (B*H*W, T+1, C)
        
        # (B*H*W, T+1, C) -> (B, H*W, T+1, C) -> (B, T+1, C, H*W) -> (B, T+1, C, H, W)
        output = x_temporal_enhanced.view(B, H * W, T + 1, C).permute(0, 2, 3, 1).reshape(B, T + 1, C, H, W)
        

        # 分离特征
        temporal_features = output[:, :T, :, :, :]  # (B, T, C, H, W)
        global_feature = output[:, T:, :, :, :]    # (B, 1, C, H, W)
        print(temporal_features.shape)
        print(BT,C,H,W)
        temporal_features=temporal_features.reshape(BT,C,H,W)
        
        return temporal_features, global_feature
        

class ResNet_t(nn.Module):

    def __init__(self, block, layers, out_channels, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True, in_channels=3,Times=15):
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.Seq_len=Times
        self.in_channels = in_channels
        self._out_channels = out_channels
        self._output_stride = 32
        self._out_channels[0] = self.inplanes
        super(ResNet_t, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=1,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        
        self.sta1=SpatioTemporalAttentionBlock(64, 4,(112,112), window_size=32, mlp_ratio=2.0, dropout=0.1)
        self.sta2=SpatioTemporalAttentionBlock(256, 4,(56,56), window_size=16, mlp_ratio=2.0, dropout=0.1)
        self.sta3=SpatioTemporalAttentionBlock(512, 4,(28,28), window_size=8, mlp_ratio=2.0, dropout=0.1)
        self.sta4=SpatioTemporalAttentionBlock(1024, 4,(14,14), window_size=4, mlp_ratio=2.0, dropout=0.1)
        self.sta5=SpatioTemporalAttentionBlock(2048, 4,(14,14), window_size=4, mlp_ratio=2.0, dropout=0.1)
        

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def get_out_channels(self):
        return self._out_channels

    def get_output_stride(self):
        return self._output_stride

    def forward(self, x):
        blocks = []
        contexts=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x,context1=self.sta1(x,self.Seq_len)
        blocks.append(x)
        contexts.append(context1)

        x = self.maxpool(x)
        x = self.layer1(x)
        x,context2=self.sta2(x,self.Seq_len)
        blocks.append(x)
        contexts.append(context2)

        x = self.layer2(x)
        x,context3=self.sta3(x,self.Seq_len)
        blocks.append(x)
        contexts.append(context3)

        x = self.layer3(x)
        x,context4=self.sta4(x,self.Seq_len)
        blocks.append(x)
        contexts.append(context4)

        x = self.layer4(x)
        blocks.append(x)
        x,context5=self.sta5(x,self.Seq_len)
        contexts.append(context5)

        return blocks,contexts


def resnet18_t(pretrained_model=None, **kwargs):
    model = ResNet_t(BasicBlock, [2, 2, 2, 2], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet34_t(pretrained_model=None, **kwargs):
    model = ResNet_t(BasicBlock, [3, 4, 6, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet50_t(pretrained_model=None, **kwargs):
    model = ResNet_t(Bottleneck, [3, 4, 6, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet101_t(pretrained_model=None, **kwargs):
    model = ResNet_t(Bottleneck, [3, 4, 23, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet152_t(pretrained_model=None, **kwargs):
    model = ResNet_t(Bottleneck, [3, 8, 36, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


if __name__ == '__main__':
    model = resnet50_t(None, in_channels=1, deep_stem=64, out_channels=[64, 256, 512, 1024, 2048],Times=15).cuda()
    x = torch.rand(2*15, 1, 224, 224).cuda()
    skips,contexts = model(x)
    print([item.shape for item in skips])
    print([item.shape for item in contexts])
