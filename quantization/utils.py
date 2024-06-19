import torch

import torch

"""
Based on:
https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py
"""

def _dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points

def group_quantize_round_to_nearest(x, bits=8, group_size=32):
    # assert group_size multiple of hidden dim
    x_shape = x.shape
    group_dim = x_shape[1]//group_size
    max_bin_value = 2**bits - 1
    
    x_reshaped = x.reshape(x_shape[0], group_dim, group_size)
    min_val, max_val = torch.aminmax(x_reshaped, dim=-1)
    zero_points = min_val
    scales = (max_val - min_val)/max_bin_value
    
    zero_points = zero_points.unsqueeze(-1)
    scales = scales.unsqueeze(-1)
    
    quantized_x = torch.round((x_reshaped - zero_points)/scales)
    quantized_x = quantized_x.to(dtype=torch.uint8)

    return quantized_x, zero_points, scales

def group_dequantize(x, scales, zero_points):
    to_reshape_x =  x * scales + zero_points
    return to_reshape_x
    #reshaped_x = to_reshape_x.reshape(x.shape[0], -1)
    #return reshaped_x