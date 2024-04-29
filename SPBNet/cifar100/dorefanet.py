import torch
import torch.nn.functional as F
from torch import Tensor 

class Quantizer(torch.nn.Module):
    def __init__(self, k: int, name: str) -> None:
        super(Quantizer, self).__init__()
        self.bits = k
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        if self.name == 'weight' and self.bits == 1:
          return Quantize1WA.apply(x, self.bits, self.name)    
        elif self.name == "activation" and self.bits == 1:
          return Quantize1WA.apply(x, self.bits, self.name)    
        else: 
          return QuantizeNWA.apply(x, self.bits, self.name)    


class Quantize1WA(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor, k: int, name: str) -> Tensor:
        ctx.save_for_backward(x)
        r_o = torch.sign(x)
        r_o[r_o==0]=-1 
        return r_o

    @staticmethod
    def backward(ctx: object, grad_out: Tensor) -> Tensor:
        x, = ctx.saved_tensors
        mask = torch.abs(x) <= 1
        grad_input = grad_out * mask
        return grad_input, None, None


class QuantizeNWA(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor, k: int, name: str) -> Tensor:
        ctx.save_for_backward(x)
        q = 2 ** k -1 
        r_o = torch.round(q * x) / q
        return r_o

    @staticmethod
    def backward(ctx: object, grad_out: Tensor) -> Tensor:
        x, = ctx.saved_tensors
        mask = (0 <=x ) & (x <= 1)
        grad_input = grad_out * mask
        return grad_input, None, None


class QuantizationWeight(torch.nn.Module):
    def __init__(self, w_bits: int) -> None:
        super(QuantizationWeight, self).__init__()
        self.bits = w_bits
        self.quantizer = Quantizer(self.bits, name='weight')

    def forward(self, x: Tensor) -> Tensor:
        if self.bits == 1:
            weight = x
            mu = torch.mean(torch.abs(weight)).detach() 
            w_q = self.quantizer(x) * mu 
        elif self.bits == 32:
            w_q = x
        else:
            weight = x
            std_w = torch.std(weight).detach() 
            r_i = (weight / (2 * 1.645*std_w)) + 0.5 
            r_i = torch.clamp(r_i, 0, 1) 
            w_q = (2 * self.quantizer(r_i) - 1) * std_w

        return w_q


class QuantizationActivation(torch.nn.Module):
    def __init__(self, a_bits: int) -> None:
        super(QuantizationActivation, self).__init__()
        self.bits = a_bits
        self.quantizer = Quantizer(self.bits, name='activation')

    def forward(self, x: Tensor) -> Tensor:
        if self.bits == 32:
            a_q = x
        elif self.bits == 1:
            a_q = self.quantizer(torch.clamp(x, -1, 1))
        else:
            a_q = torch.clamp(x, -1, 1)
            a_q = (a_q / 2)  + 0.5
            a_q = 2 * self.quantizer(a_q) - 1

        return a_q


class QuantizationConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int=1,
        padding: int=0,
        dilation: int=1,
        groups: int=1,
        bias: bool=True,
        w_bits: int=1,
        output_height: int=224,
        output_width: int=224
    ) -> None:
        super(QuantizationConv2d, self).__init__(
            in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias)
        self.quantized_weight = QuantizationWeight(w_bits)
        self.alpha = torch.nn.Parameter(torch.ones(output_height).reshape(1,-1,1))
        self.beta = torch.nn.Parameter(torch.ones(output_width).reshape(1,1,-1))
        self.gamma = torch.nn.Parameter(torch.ones(out_channel).reshape(-1,1,1))
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.quantized_weight(self.weight)
        x = F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x.mul(self.gamma).mul(self.beta).mul(self.alpha)  


