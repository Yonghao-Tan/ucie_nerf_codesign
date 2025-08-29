from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import decimal
from decimal import Decimal
import numpy as np

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def trunc_pass(x):
    y = torch.trunc(x)
    y_grad = x
    return (y - y_grad).detach() + y_grad

def sign_pass(x):
    """Sign function with straight-through estimator for 1-bit quantization"""
    y = torch.sign(x)
    y_grad = x
    return (y - y_grad).detach() + y_grad

def clip(x, eps):
    x_clip = torch.where(x > eps.to(x.device), x, eps.to(x.device))
    return x - x.detach() + x_clip.detach()

def batch_frexp(inputs, bit=8):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """
    shape_of_input = inputs.size()

    # transform the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().detach().numpy())

    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** bit)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = 1.0*bit - output_e

    return torch.from_numpy(output_m).to(inputs.device).view(shape_of_input), \
           torch.from_numpy(output_e).to(inputs.device).view(shape_of_input)

def batch_frexp_new(inputs, bit=8):
    shape_of_input = inputs.size()
    # Flatten the input tensor
    inputs = inputs.view(-1)

    # Use PyTorch operations to decompose the input tensor
    abs_inputs = torch.abs(inputs)
    exponents = torch.floor(torch.log2(abs_inputs))
    mantissas = abs_inputs / (2 ** exponents)

    # Shift and round the mantissas to match Decimal rounding behavior
    mantissas_shifted = torch.round(mantissas * (2 ** bit)).to(torch.int32)

    # Ensure the mantissa fits within the specified bit-width
    overflow_mask = mantissas_shifted >= (2 ** bit)
    mantissas_shifted[overflow_mask] //= 2  # Divide by 2 if overflow occurs
    exponents[overflow_mask] += 1           # Increment exponent if overflow occurs

    # Adjust the exponents to ensure the correct scaling
    exponents = bit - exponents

    # Restore the sign of the mantissas
    mantissas_shifted = mantissas_shifted * torch.sign(inputs).to(torch.int32)

    # Reshape the outputs to the original shape
    output_m = mantissas_shifted.view(shape_of_input).to(inputs.device)
    output_e = exponents.view(shape_of_input).to(inputs.device)

    return output_m, output_e

def replace_linear_with_quantized(model: nn.Module,
                                  device,
                                num_bits: int = 1,
                                skip_layers: Optional[list] = None,
                                sparsity: float = None,
                                no_frexp=False) -> nn.Module:
    """
    Args:
        initial_scale_value: Value to initialize QuantizedLinear's scale parameter with.
    """
    if skip_layers is None:
        skip_layers = []
    
    def _replace_module(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear) and full_name not in skip_layers:
                quantized_layer = QuantizedLinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    bias=child_module.bias is not None,
                    num_bits=num_bits,
                    device=child_module.weight.device,
                    dtype=child_module.weight.dtype,
                    sparsity=sparsity,
                    no_frexp=no_frexp
                )
                
                with torch.no_grad():
                    quantized_layer.weight.copy_(child_module.weight)
                    if child_module.bias is not None:
                        quantized_layer.bias.copy_(child_module.bias)
                    quantized_layer._init()

                setattr(module, child_name, quantized_layer)
                # print(f"Replaced {full_name} with QuantizedLinear ({num_bits}-bit)")
            else:
                _replace_module(child_module, full_name)
    
    _replace_module(model)
    model.to(device)

def replace_conv2d_with_quantized(model: nn.Module,
                                  device,
                                num_bits: int = 1,
                                skip_layers: Optional[list] = None,
                                sparsity: float = None,
                                no_frexp=False) -> nn.Module:
    """
    Args:
        initial_scale_value: Value to initialize QuantizedConv2d's scale parameter with.
    """
    if skip_layers is None:
        skip_layers = []
    
    def _replace_module(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, nn.Conv2d) and full_name not in skip_layers:
                quantized_layer = QuantizedConv2d(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    num_bits=num_bits,
                    device=child_module.weight.device,
                    dtype=child_module.weight.dtype,
                    sparsity=sparsity,
                    no_frexp=no_frexp
                )

                with torch.no_grad():
                    quantized_layer.weight.copy_(child_module.weight)
                    if child_module.bias is not None:
                        quantized_layer.bias.copy_(child_module.bias)
                    quantized_layer._init()

                setattr(module, child_name, quantized_layer)
                # print(f"Replaced {full_name} with QuantizedConv2d ({num_bits}-bit)")
            else:
                _replace_module(child_module, full_name)
    
    _replace_module(model)
    model.to(device)
    
class QuantizedLinear(nn.Linear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 num_bits: int = 1,
                 device=None,
                 dtype=None,
                 sparsity=None,
                 no_frexp=False
                 ):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias, device, dtype)

        self.thd_neg = - 2 ** (num_bits - 1)
        self.thd_pos = 2 ** (num_bits - 1) - 1

        self.num_bits = num_bits
        self.group_size = None
        
        self.sparsity = sparsity
        self.no_frexp = no_frexp
    
        self.scale = None  # Will be initialized in init_from method
        
        self.scale_a = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_buffer('initialized_alpha', torch.zeros(1))

    def _init(self):
        # Initialize groupwise quantization parameters
        x = self.weight
        # For multi-bit quantization, use the original formula
        init_val = 2 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5)
            
        # Initialize parameter if not already done
        if self.scale is None:
            self.scale = torch.nn.Parameter(init_val, requires_grad=True)
        else:
            self.scale.data.copy_(init_val)
    
    def init_activation(self, x, modified_init=True):
        if not modified_init: 
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
        else:
            init_val_max = x.detach().abs().max() / self.thd_pos
            init_val_lsq = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            init_val = init_val_max if init_val_max < init_val_lsq else init_val_lsq
        self.scale_a.data.copy_(init_val)
        self.initialized_alpha.fill_(1) # only initialize once, first forward pass
        
    def _apply_weight_quantization(self, x):
        scale_factors = self.scale.view(-1, 1)
        
        s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5)
        scale_factors = grad_scale(scale_factors, s_grad_scale)
        
        if not self.no_frexp:
            scale_m, scale_e =  batch_frexp_new(scale_factors.detach(), bit=8)
            scale = (scale_m / torch.pow(2, scale_e)).type(torch.float32)
            scale_factors = (scale - scale_factors).detach() + scale_factors
                                                                       
        x_scaled = x / scale_factors
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_scaled = round_pass(x_scaled)
        x_quantized = x_scaled * scale_factors
            
        return x_quantized
    
    def _apply_activation_quantization(self, x):
        if self.initialized_alpha == 0:
            # print("init_input begin")
            self.init_activation(x)     
        scale_factors = self.scale_a.to(x.device)
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        scale_factors = grad_scale(scale_factors, s_grad_scale)
        
        if not self.no_frexp:
            pow = torch.round(torch.log2(scale_factors.detach()))
            clip_val = torch.pow(2, pow)
            scale_factors = (clip_val - scale_factors).detach() + scale_factors
                                                                       
        x_scaled = x / scale_factors
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_scaled = round_pass(x_scaled)
        x_quantized = x_scaled * scale_factors
            
        return x_quantized
    
    def apply_chn_pruning(self, weight, pruning_rate=0.5):
        num_rows = weight.size(0)
        num_cols = weight.size(1)
        num_rows_to_keep = torch.tensor(num_rows * (1 - pruning_rate))
        num_rows_to_keep = int(torch.clamp(num_rows_to_keep, 1, num_rows))
        _, indices = torch.topk(torch.abs(weight), k=num_rows_to_keep, dim=0)
        mask = torch.zeros_like(weight)
        mask.scatter_(0, indices, 1)
        # return weight * mask
        return (weight * mask - weight).detach() + weight
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_weight = self._apply_weight_quantization(self.weight)
        quantized_input = self._apply_activation_quantization(input)
        if self.sparsity is not None:
            sparse_weight = self.apply_chn_pruning(quantized_weight, self.sparsity)
            return F.linear(quantized_input, sparse_weight, self.bias)
        return F.linear(quantized_input, quantized_weight, self.bias)

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 num_bits: int = 1,
                 device=None,
                 dtype=None,
                 sparsity=None,
                 no_frexp=False
                 ):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.thd_neg = - 2 ** (num_bits - 1)
        self.thd_pos = 2 ** (num_bits - 1) - 1

        self.num_bits = num_bits
        self.group_size = None
        
        if self.groups != 1:
            self.sparsity = None
        else:
            self.sparsity = sparsity
        self.no_frexp = no_frexp
    
        self.scale = None  # Will be initialized in init_from method
        
        self.scale_a = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_buffer('initialized_alpha', torch.zeros(1))

    def _init(self):
        # Initialize groupwise quantization parameters
        x = self.weight
        # For multi-bit quantization, use the original formula
        init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5)
            
        # Initialize parameter if not already done
        if self.scale is None:
            self.scale = torch.nn.Parameter(init_val, requires_grad=True)
        else:
            self.scale.data.copy_(init_val)
    
    def init_activation(self, x, modified_init=True):
        if not modified_init: 
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
        else:
            init_val_max = x.detach().abs().max() / self.thd_pos
            init_val_lsq = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            init_val = init_val_max if init_val_max < init_val_lsq else init_val_lsq
        self.scale_a.data.copy_(init_val)
        self.initialized_alpha.fill_(1) # only initialize once, first forward pass
        
    def _apply_weight_quantization(self, x):
        scale_factors = self.scale
        
        s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-3]*x.shape[-2]* x.shape[-1]) ** 0.5)
        scale_factors = grad_scale(scale_factors, s_grad_scale)
        
        if not self.no_frexp:
            scale_m, scale_e =  batch_frexp_new(scale_factors.detach(), bit=8)
            scale = (scale_m / torch.pow(2, scale_e)).type(torch.float32)
            scale_factors = (scale - scale_factors).detach() + scale_factors
        
        scale_factors = scale_factors.view(scale_factors.shape[0],1,1,1)
                                                                       
        x_scaled = x / scale_factors
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_scaled = round_pass(x_scaled)
        x_quantized = x_scaled * scale_factors
            
        return x_quantized
    
    def _apply_activation_quantization(self, x):
        if self.initialized_alpha == 0:
            # print("init_input begin")
            self.init_activation(x)     
        scale_factors = self.scale_a.to(x.device)
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        scale_factors = grad_scale(scale_factors, s_grad_scale)
        
        if not self.no_frexp:
            pow = torch.round(torch.log2(scale_factors.detach()))
            clip_val = torch.pow(2, pow)
            scale_factors = (clip_val - scale_factors).detach() + scale_factors
                                                                       
        x_scaled = x / scale_factors
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_scaled = round_pass(x_scaled)
        x_quantized = x_scaled * scale_factors
            
        return x_quantized
    
    def apply_chn_pruning(self, weight, pruning_rate=0.5):
        num_rows = weight.size(0)
        num_cols = weight.size(1)
        num_rows_to_keep = torch.tensor(num_rows * (1 - pruning_rate))
        num_rows_to_keep = int(torch.clamp(num_rows_to_keep, 1, num_rows))
        _, indices = torch.topk(torch.abs(weight), k=num_rows_to_keep, dim=0)
        mask = torch.zeros_like(weight)
        mask.scatter_(0, indices, 1)
        # return weight * mask
        return (weight * mask - weight).detach() + weight
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_weight = self._apply_weight_quantization(self.weight)
        quantized_input = self._apply_activation_quantization(input)
        if self.sparsity is not None:
            sparse_weight = self.apply_chn_pruning(quantized_weight, self.sparsity)
            return F.conv2d(quantized_input, sparse_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(quantized_input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
