import torch
import torch.nn as nn
import torch.nn.functional as F

from base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        if (skip is not None) and (x.shape[2:] != skip.shape[2:]):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif skip is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            use_batchnorm=True):
        super(UnetDecoder, self).__init__()

        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.center = nn.Identity()

    def forward(self, *features,return_feats=False):
        features = features[::-1]

        head = features[0]
        skips = features[1:]
        out_feats=[]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            out_feats.append(x)
        if return_feats:
            return x,out_feats
        else:
            return x

class UnetDecoder_t(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            use_batchnorm=True):
        super(UnetDecoder_t, self).__init__()

        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.center = nn.Identity()

    def forward(self, *features):
        features = features[::-1]
        out_feats=[]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        out_feats.append(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            out_feats.append(x)
            
        return x,out_feats

