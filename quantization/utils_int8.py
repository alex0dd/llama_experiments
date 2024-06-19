import torch
from .utils import _dynamically_quantize_per_channel

def quantize_fp32_linear_to_int8(linear_orig_weights):
    int8_weight, scales, _ = _dynamically_quantize_per_channel(linear_orig_weights.float(), -128, 127, torch.int8)
    return int8_weight, scales.to(linear_orig_weights.dtype)