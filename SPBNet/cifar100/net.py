import torch.nn
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from dorefanet import *
from typing import Type, Any, Callable, List, Optional, Tuple, Union
import math

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class SpatialSE(nn.Module):
    def __init__(self, channel, height, width, ratio=4):
        super(SpatialSE, self).__init__()
        self.fch = nn.Sequential(
            nn.Linear(height, math.ceil(height / ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(math.ceil(height/ ratio), height, bias=False),
            nn.Sigmoid()
        )

        self.fcw = nn.Sequential(
            nn.Linear(width, math.ceil(width / ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(math.ceil(width/ ratio), width, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        height_avg = torch.mean(x, dim=[1,3], keepdim=True)
        width_avg = torch.mean(x, dim=[1,2], keepdim=True)
        b, _, h, w = x.size()
        yh = height_avg.view(b, h)
        yw = width_avg.view(b, w)
        yh = self.fch(yh).view(b, 1, h, 1)
        yw = self.fcw(yw).view(b, 1, 1, w)
        return x.mul(yh).mul(yw) 


class XBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, w_bits=32, a_bits=32, output_height = 32, output_width = 32, spatialratio=4):
        super(XBasicBlock, self).__init__()

        self.move1b = SpatialSE(inplanes, output_height * stride, output_width * stride, ratio=spatialratio)
        self.move1c = LearnableBias(inplanes)

        self.move2b = SpatialSE(planes, output_height, output_width, ratio=spatialratio)
        self.move2c = LearnableBias(planes)

        self.move11 = LearnableBias(planes)
        self.move12 = LearnableBias(planes)
        self.move21 = LearnableBias(planes)
        self.move22 = LearnableBias(planes)

        self.quant_conv1 = QuantizationConv2d(inplanes, planes, 3, stride=stride, padding=1, 
          bias=False, w_bits=w_bits, output_height = output_height, output_width = output_width)
        self.quant_conv2 = QuantizationConv2d(planes, planes, 3, padding=1, 
          bias=False, w_bits=w_bits, output_height = output_height, output_width = output_width)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU(planes)
        self.relu2 = nn.PReLU(planes)
        
        self.quant_activation = QuantizationActivation(a_bits)
        self.downsample= downsample


    def forward(self, x):
        residual1 = x 
        out = self.move1b(x) + 0.5 * x 
        out = self.move1c(out) 
        out = self.quant_activation(out)
        out = self.quant_conv1(out)
        out = self.bn1(out)
        out = self.relu1(out) 
        out = self.move11(out) 
        if self.downsample is not None:
          residual1 = self.downsample(x) 
        out += residual1
        out = self.move12(out) 
        residual2 = out
        out = self.move2b(out) + 0.5 * out 
        out = self.move2c(out)  
        out = self.quant_activation(out)
        out = self.quant_conv2(out)
        out = self.bn2(out)
        out += residual2
        out = self.move21(out) 
        out = self.relu2(out) 
        out = self.move22(out) 
        return out


class XResNet18(torch.nn.Module):
    def __init__(
        self,
        block: XBasicBlock,
        layers: List[int],
        w_bits: int,
        a_bits: int,
        num_classes: int=100,
        output_height = 32, 
        output_width = 32 
    ) -> None:
        super(XResNet18, self).__init__()

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.inplanes = 64

        self.conv = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.PReLU()

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, output_height = output_height//1, output_width = output_width//1, spatialratio=4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, output_height = output_height//2, output_width = output_width//2, spatialratio=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, output_height = output_height//4, output_width = output_width//4, spatialratio=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, output_height = output_height//8, output_width = output_width//8, spatialratio=4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.quant_activation = QuantizationActivation(a_bits)
        self.dropout = nn.Dropout()

    def _make_layer(self, block: Type[XBasicBlock], planes: int, num_blocks: int, stride: int, output_height: int, output_width: int, spatialratio: int) -> nn.Sequential:
        downsample = None
        layers = []

        if (stride != 1) or (self.inplanes != planes * block.expansion): 
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride), 
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )


        layers.append(block(self.inplanes, planes, stride, downsample, w_bits=self.w_bits, a_bits=self.a_bits, output_height = output_height, output_width = output_width, spatialratio=spatialratio))

        self.inplanes = planes * block.expansion

        for num_block in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, w_bits=self.w_bits, a_bits=self.a_bits, output_height = output_height, output_width = output_width))
        return nn.Sequential(*layers)


    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def xresnet18(w_bits, a_bits, **kwargs):
    model = XResNet18(XBasicBlock, [2, 2, 2, 2], w_bits, a_bits, **kwargs)
    return model
