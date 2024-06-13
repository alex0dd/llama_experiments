import torch

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

def quantize_fp32_linear_to_int4(weight, groupsize: int = 128, inner_k_tiles=8, padding=True):
    # https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py#L396
    assert groupsize in [32, 64, 128, 256]
    assert inner_k_tiles in [2, 4, 8]
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    device = "cpu"

    fqn = "layer_name"
    print(f"linear: {fqn}, in={in_features}, out={out_features}")

    if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
        if padding:
            print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
            padded_in_features = find_multiple(in_features, 1024)
            weight = torch.nn.functional.pad(weight, pad=(0, padded_in_features - in_features))
        else:
            print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
        "and that groupsize and inner_k_tiles*16 evenly divide into it")
            return
    weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
        weight.to(torch.bfloat16).to(device=device), groupsize, inner_k_tiles
    )
    return weight_int4pack, scales_and_zeros