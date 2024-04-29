import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from dorefanet import *
# from module import *
from typing import Type, Any, Callable, List, Optional, Tuple, Union
import math

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class SpatialSE(nn.Module):
    def __init__(self, height, width, ratio=4):
        super(SpatialSE, self).__init__()
        self.fch = nn.Sequential(
            nn.Linear(height, math.ceil(height / ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(math.ceil(height / ratio), height, bias=False),
            nn.Sigmoid()
        )

        self.fcw = nn.Sequential(
            nn.Linear(width, math.ceil(width / ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(math.ceil(width / ratio), width, bias=False),
            nn.Sigmoid()
        )
        self.height = height
    def forward(self, x):
        
        height_avg = torch.mean(x, dim=[1,3], keepdim=True)
        width_avg = torch.mean(x, dim=[1,2], keepdim=True)
        b, _, h, w = x.size()
        yh = height_avg.view(b, h)
        yw = width_avg.view(b, w)
        yh = self.fch(yh).view(b, 1, h, 1)
        yw = self.fcw(yw).view(b, 1, 1, w)
        return x.mul(yh).mul(yw) 

    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, w_bits=32, a_bits=32, output_height = 224, output_width = 224):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(inplanes)
        self.bn2 = norm_layer(inplanes)

        self.binary_3x3= QuantizationConv2d(inplanes, inplanes, 3, stride=stride, padding=1, 
          bias=False, w_bits=w_bits, output_height = output_height, output_width = output_width)
        self.prelu1 = nn.PReLU(inplanes)
        self.prelu2 = nn.PReLU(planes)

        self.move1a = SpatialSE(output_height * stride, output_width * stride)
        self.move1c = LearnableBias(inplanes)

        self.move2a = SpatialSE(output_height, output_width)
        self.move2c = LearnableBias(inplanes)

        self.move11 = LearnableBias(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.move21 = LearnableBias(planes)
        self.move22 = LearnableBias(planes)


        if inplanes == planes:
            self.binary_pw = QuantizationConv2d(inplanes, planes, 1, stride=1, 
                bias=False, w_bits=w_bits, output_height = output_height, output_width = output_width)
            self.bn2_1 = norm_layer(planes)
        else:
            self.binary_pw_down1 = QuantizationConv2d(inplanes, inplanes, 1, stride=1, 
                bias=False, w_bits=w_bits, output_height = output_height, output_width = output_width)
            self.binary_pw_down2 = QuantizationConv2d(inplanes, inplanes, 1, stride=1, 
                bias=False, w_bits=w_bits, output_height = output_height, output_width = output_width)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)


        self.binary_activation = QuantizationActivation(a_bits)
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):
        out1 = self.move1a(x) + 0.5 * x
        out1 = self.move1c(out1) 

        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)
        out1 = self.prelu1(out1)
        out1 = self.move11(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1
        out1 = self.move12(out1)

        out2 = self.move2a(out1) + 0.5 * out1
        out2 = self.move2c(out2) 

        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2_1(out2)
            out2 += out1
        else:
            assert self.planes == self.inplanes * 2
            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move21(out2)
        out2 = self.prelu2(out2)
        out2 = self.move22(out2)

        return out2


class xreactnet(nn.Module):
    def __init__(self, w_bits, a_bits, num_classes=1000, output_height=224, output_width=224):
        super(xreactnet, self).__init__()
        self.feature = nn.ModuleList()
        self.height = output_height
        self.width = output_width
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
                self.height = self.height // 2
                self.width = self.width // 2
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.height = self.height // 2
                self.width = self.width // 2
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2, w_bits=w_bits, a_bits=a_bits, output_height=self.height, output_width=self.width))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1, w_bits=w_bits, a_bits=a_bits, output_height=self.height, output_width=self.width))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x






