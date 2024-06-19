import torch
from .utils import group_quantize_round_to_nearest, group_dequantize

"""
Based on:
https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py
"""

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)

def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros

##### weight only int4 per channel groupwise quantized code ######

def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    # https://github.com/pytorch/pytorch/blob/5ffb032be682a34b959c82ce289b457ea6c6e504/aten/src/ATen/native/LinearAlgebra.cpp#L3476
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros

def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


"""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )
"""
def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c

def quantize_fp32_linear_to_int4(weight, layer_name, groupsize: int = 128, inner_k_tiles=8, padding=True, device="cpu"):
    # https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py#L396
    assert groupsize in [32, 64, 128, 256]
    assert inner_k_tiles in [2, 4, 8]
    out_features = weight.shape[0]
    in_features = weight.shape[1]

    print(f"linear: {layer_name}, in={in_features}, out={out_features}")

    if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
        if padding:
            print(f"warning: {layer_name} is padded to satisfy in_features % 1024 == 0")
            padded_in_features = find_multiple(in_features, 1024)
            weight = torch.nn.functional.pad(weight, pad=(0, padded_in_features - in_features))
        else:
            print(f"warning: {layer_name} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
        "and that groupsize and inner_k_tiles*16 evenly divide into it")
            return
    weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
        weight.to(torch.bfloat16).to(device=device), groupsize, inner_k_tiles
    )
    return weight_int4pack, scales_and_zeros

########
# Adaptation of functions from https://github.com/mobiusml/hqq/
########

# 4-bit
################################################
from torch import Tensor, uint8

def pack_4bit_u8_last_dim(W_q: Tensor) -> Tensor:  # uint8 > uint8/2
    W_q = W_q.to(uint8)
    _step = int(W_q.shape[-1] / 2)

    return (W_q[..., :_step] << 4) | W_q[..., _step:]

@staticmethod
def unpack_4bit_u8_last_dim(W_q: Tensor, dtype=uint8) -> Tensor:  # uint8/2 > uint8
    _step = W_q.shape[-1]
    tmp = torch.empty(list(W_q.shape)[:-1] + [2 * _step], dtype=dtype, device=W_q.device)

    tmp[..., :_step] = (W_q & 0b11110000) >> 4
    tmp[..., _step:] = W_q & 0b00001111

    return tmp

"""
Layers operations
"""

def quantize_pack_embedding_table(embedding_table, group_size=32):
    """
    Quantizes an embedding table by taking groups on each row and packs the 4bit integers into a single byte.
    At inference time, the looked up indices will yield a very small portion of this quantized and packed table.
    """
    n_bits = 4
    # TODO: check if we can use here the quantization function from int8, which doesn't return zeros but only scales
    embedding_table_q, embedding_table_zero_points, embedding_table_scales = group_quantize_round_to_nearest(embedding_table, bits=n_bits, group_size=group_size)
    packed_embedding_table_q = pack_4bit_u8_last_dim(embedding_table_q)
    return packed_embedding_table_q, embedding_table_zero_points, embedding_table_scales


from .utils import _dynamically_quantize_per_channel
def pack_4bit_int8_last_dim(W_q: Tensor) -> Tensor:  # int8 > int8/2
    W_q = W_q.to(torch.int8)
    _step = int(W_q.shape[-1] / 2)
    return (W_q[..., :_step] << 4) | W_q[..., _step:]

def unpack_4bit_int8_last_dim(W_q: Tensor, dtype=torch.int8) -> Tensor:  # int8/2 > int8
    _step = W_q.shape[-1]
    tmp = torch.empty(list(W_q.shape)[:-1] + [2 * _step], dtype=dtype, device=W_q.device)
    tmp[..., :_step] = (W_q & 0b11110000) >> 4
    tmp[..., _step:] = W_q & 0b00001111
    return tmp

def quantize_pack_embedding_table_v2(embedding_table):
    int4_weight, scales, _ = _dynamically_quantize_per_channel(embedding_table.T.float(), -8, 7, torch.int8)
    int4_weight = pack_4bit_int8_last_dim(int4_weight.T)
    return int4_weight, scales

def lookup_on_quantized_and_packed_embedding_table(indices, embedding_table_q_p, embedding_table_zero_points, embedding_table_scales):
    indices_shape = list(indices.size())
    weights_shape = list(embedding_table_q_p.size()[1:])

    zeros_shape = list(embedding_table_zero_points.size()[1:])
    scales_shape = list(embedding_table_scales.size()[1:])
    # Selects rows using flattened indices, and reshapes to (B, S, Group, Hidden//Group//packing_factor)
    selected_rows = torch.index_select(embedding_table_q_p, 0, indices.reshape(-1)).view(indices_shape + weights_shape)

    selected_zeros = torch.index_select(embedding_table_zero_points, 0, indices.reshape(-1)).view(indices_shape + zeros_shape)
    selected_scales = torch.index_select(embedding_table_scales, 0, indices.reshape(-1)).view(indices_shape + scales_shape)

    # To avoid MPS error, we unpack on CPU
    orig_device = selected_rows.device
    selected_rows = unpack_4bit_u8_last_dim(selected_rows.to("cpu")).to(orig_device)

    # Results in shape (B, S, Group, Hidden//Group)
    embedding_weights_dequant = group_dequantize(selected_rows, selected_zeros, selected_scales)
    # Reshape to (B, S, Hidden)
    embedding_weights_dequant = embedding_weights_dequant.flatten(start_dim=-len(weights_shape))

    return embedding_weights_dequant