# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over channels only. Normalizes over C only.
    Input shape:  [B, C, T, F]
    """
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # [B, C, T, F] -> [B, T, F, C]
        x = x.permute(0, 2, 3, 1)
        # LayerNorm over C
        x = self.norm(x)
        # Back to [B, C, T, F]
        x = x.permute(0, 3, 1, 2)
        return x

def get_causal_padding_2d(kernel_size, dilation=(1,1)):
    """
    Causal padding only along time axis.
    Frequency axis uses symmetric padding.
    """
    pad_t =  kernel_size[0] * dilation[0] - dilation[0]  # all padding on left side
    pad_f = (kernel_size[1] * dilation[1] - dilation[1]) // 2
    return (pad_f, pad_f, pad_t, 0)  # (left, right, top, bottom)

def get_causal_padding_2d_FT(kernel_size, dilation=(1,1)):
    """
    Causal padding only along time axis.
    Frequency axis uses symmetric padding.
    """
    pad_t =  kernel_size[1] * dilation[1] - dilation[1]  # all padding on left side
    pad_f = (kernel_size[0] * dilation[0] - dilation[0]) // 2
    return (pad_t, 0, pad_f, pad_f)  # (left, right, top, bottom)

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d(padding, value=0.)
        self.out_channels = out_channels
        self.r = r
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, padding=0)
        

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.dense_block = nn.ModuleList()

        for i in range(depth):
            dil = 2 ** i 
            pad = get_causal_padding_2d(kernel_size, (dil, 1))
            dense_conv = nn.Sequential(
                nn.ConstantPad2d(pad, 0.0),
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=0),
                ChannelLayerNorm(self.hid_feature),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """
    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (3, 3)),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.dense_conv_1_2 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (3, 3)),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.dense_conv_1_3 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (3, 3)),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.dense_block = DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, x, number_ahead):
        x = F.pad(x, (1,1, 2-number_ahead, number_ahead), "constant", 0) # (pad_f, pad_f, pad_t, 0)  (left, right, top, bottom)
        if number_ahead==0:
            x = self.dense_conv_1_1(x)  # [batch, hid_feature, time, freq]
        elif number_ahead==1:
            x = self.dense_conv_1_2(x)  # [batch, hid_feature, time, freq]
        elif number_ahead==2: 
            x = self.dense_conv_1_3(x)  # [batch, hid_feature, time, freq]
        else:
            print('number_ahead not support!')

        x = self.dense_block(x)   # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x

class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """
    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        self.up_conv1 = nn.Sequential(
            SPConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), get_causal_padding_2d((1,3)), 2),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.up_conv2 = nn.Sequential(
            SPConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), get_causal_padding_2d_FT((1,3)), 1),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.final_conv = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x.permute(0,1,3,2)).permute(0,1,3,2)
        x = self.final_conv(x)

        return x

class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """
    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']

        self.up_conv1 = nn.Sequential(
            SPConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), get_causal_padding_2d((1,3)), 2),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.up_conv2 = nn.Sequential(
            SPConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), get_causal_padding_2d_FT((1,3)), 1),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature)
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x