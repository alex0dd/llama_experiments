import torch
from .utils import _dynamically_quantize_per_channel

def quantize_fp32_linear_to_int8(linear_orig_weights):
    int8_weight, scales, _ = _dynamically_quantize_per_channel(linear_orig_weights.float(), -128, 127, torch.int8)
    return int8_weight, scales.to(linear_orig_weights.dtype)

def linear_int8(inputs, weight, scales, bias=None, original_shape=None):
    out = torch.nn.functional.linear(inputs, weight.to(dtype=inputs.dtype))
    if scales is not None:
        out*= scales
    if bias is not None:
        out+= bias
    return out

def embedding_int8(inputs, weights, scales=None, original_shape=None):
    embs = torch.nn.functional.embedding(inputs, weights)
    if scales is not None:
        embs = embs.to(scales.dtype) * scales
    return embs