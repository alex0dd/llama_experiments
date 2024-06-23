from typing import Dict, Optional

import torch

from quantization.utils_int4 import embedding_int4, linear_int4
from quantization.utils_int8 import embedding_int8, linear_int8

from .rope import LLAMA3_PositionalEmbeddings, Phi3_PositionalEmbeddings
from .utils import (
    build_attention_mask,
    build_kv_caches,
    load_block_chunk,
    load_general_chunk,
    repeat_kv,
)

# TODO: clean this file and refactor
# TODO: make RMSNorm's weights int8 quantizable


def embedding_matrix(inputs, weights, scales=None, original_shape=None):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)


def linear_default(inputs, weight, scales=None, bias=None, original_shape=None):
    out = torch.nn.functional.linear(inputs, weight.to(dtype=inputs.dtype), bias=bias)
    return out


linears = {"default": linear_default, "int4": linear_int4, "int8": linear_int8}
embeddings = {
    "default": embedding_matrix,
    "int4": embedding_int4,
    "int8": embedding_int8,
}
prec = "default"
linear_quantized = linears[prec]
embedding_quantized = embeddings[prec]


@torch.inference_mode()
def functional_ffn_quantized(x: torch.Tensor, weights: Dict[str, torch.Tensor]):
    output = linear_quantized(
        x,
        weights["mlp.gate_proj.weight"],
        weights.get("mlp.gate_proj.weight_scales"),
        original_shape=weights.get("mlp.gate_proj.weight_orig_shape"),
        bias=weights.get("mlp.gate_proj.bias"),
    )
    output = torch.nn.functional.silu(output) * linear_quantized(
        x,
        weights["mlp.up_proj.weight"],
        weights.get("mlp.up_proj.weight_scales"),
        original_shape=weights.get("mlp.up_proj.weight_orig_shape"),
        bias=weights.get("mlp.up_proj.bias"),
    )
    output = linear_quantized(
        output,
        weights["mlp.down_proj.weight"],
        weights.get("mlp.down_proj.weight_scales"),
        original_shape=weights.get("mlp.down_proj.weight_orig_shape"),
        bias=weights.get("mlp.down_proj.bias"),
    )
    return output


@torch.inference_mode()
def functional_ffn_phi3_quantized(x: torch.Tensor, weights: Dict[str, torch.Tensor]):
    # TODO: refactor this with weights.get
    up_states = linear_quantized(
        x,
        weights["mlp.gate_up_proj.weight"],
        weights.get("mlp.gate_up_proj.weight_scales"),
        original_shape=weights.get("mlp.gate_up_proj.weight_orig_shape"),
    )
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = torch.nn.functional.silu(gate) * up_states
    output = linear_quantized(
        up_states,
        weights["mlp.down_proj.weight"],
        weights.get("mlp.down_proj.weight_scales"),
        original_shape=weights.get("mlp.down_proj.weight_orig_shape"),
    )
    return output


def _normalization(inputs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Assume inputs of shape (B, S, n)
    rms_a = inputs.pow(2).mean(-1, keepdim=True)
    rms_a = inputs * torch.rsqrt(rms_a + eps)
    return rms_a


@torch.inference_mode()
@torch.jit.script
def functional_rmsnorm(
    inputs: torch.Tensor, weights: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Root Mean Square Layer Normalization:
    https://arxiv.org/abs/1910.07467

    In the original paper inputs = a, weights = g (gain)
    """
    # Perform operation in fp32 precision
    output = _normalization(inputs.float()).type_as(inputs)
    return output * weights


def base_attn(q, k, v, mask, head_dim):
    normalize_fact = float(head_dim) ** -0.5
    scores = (q * normalize_fact) @ k.transpose(3, 2)
    if mask is not None:
        scores = scores + mask
    scores = torch.softmax(scores, axis=-1)
    output = (scores @ v).transpose(2, 1)
    return output


@torch.inference_mode()
def functional_gqa_quantized(
    x: torch.Tensor,
    start_pos: int,
    weights: Dict[str, torch.Tensor],
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    freqs_rope: torch.Tensor,
    n_rep: int,
    n_kv_heads: int,
    n_heads: int,
    head_dim: int,
    mask: Optional[torch.Tensor],
    model_type: str,
):
    bs, seq_len, _ = x.shape
    # Apply attention transformation matrices
    # (BS, S, dim) -> (BS, S, n_heads * head_dim)
    xq = linear_quantized(
        x,
        weights["self_attn.q_proj.weight"],
        weights.get("self_attn.q_proj.weight_scales"),
        original_shape=weights.get("self_attn.q_proj.weight_orig_shape"),
        bias=weights.get("self_attn.q_proj.bias"),
    )
    # (BS, S, dim) -> (BS, S, n_kv_heads * head_dim)
    xk = linear_quantized(
        x,
        weights["self_attn.k_proj.weight"],
        weights.get("self_attn.k_proj.weight_scales"),
        original_shape=weights.get("self_attn.k_proj.weight_orig_shape"),
        bias=weights.get("self_attn.k_proj.bias"),
    )
    xv = linear_quantized(
        x,
        weights["self_attn.v_proj.weight"],
        weights.get("self_attn.v_proj.weight_scales"),
        original_shape=weights.get("self_attn.v_proj.weight_orig_shape"),
        bias=weights.get("self_attn.v_proj.bias"),
    )
    # Reshapes
    # (BS, S, n_heads * head_dim) -> (BS, S, n_heads, head_dim)
    xq = xq.view(bs, seq_len, n_heads, head_dim)
    # (BS, S, n_kv_heads * head_dim) -> (BS, S, n_kv_heads, head_dim)
    xk = xk.view(bs, seq_len, n_kv_heads, head_dim)
    xv = xv.view(bs, seq_len, n_kv_heads, head_dim)
    # Apply positional embeddings
    if model_type == "phi3" or model_type == "granite-small":
        xq, xk = Phi3_PositionalEmbeddings.apply_rotary_emb(
            xq, xk, cos=freqs_rope[0], sin=freqs_rope[1]
        )
    else:
        xq, xk = LLAMA3_PositionalEmbeddings.apply_rotary_emb(xq, xk, freqs_rope)
    # Populate KV cache
    if cache_k is not None:
        cache_k[:bs, start_pos : start_pos + seq_len] = xk
        keys = cache_k[:bs, : start_pos + seq_len]
    else:
        # TODO: check if sequence length here is correct, given we start from pos=0
        keys = xk
    if cache_v is not None:
        cache_v[:bs, start_pos : start_pos + seq_len] = xv
        values = cache_v[:bs, : start_pos + seq_len]
    else:
        # TODO: check if sequence length here is correct, given we start from pos=0
        values = xv

    if n_kv_heads < n_heads:
        # Pad keys and values from n_kv_heads to n_heads, if needed (if n_kv_heads < n_heads)
        keys = repeat_kv(keys, n_rep)  # (BS, cache_len + S, n_heads, head_dim)
        values = repeat_kv(values, n_rep)  # (BS, cache_len + S, n_heads, head_dim)

    if xq.device.type == "mps":
        xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, keys, values))
        output = base_attn(xq, keys, values, mask, head_dim)
        output = output.reshape(bs, seq_len, -1)
    else:
        xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, keys, values))
        output = torch.nn.functional.scaled_dot_product_attention(
            xq, keys, values, attn_mask=mask, dropout_p=0.0
        )
        # # (BS, n_heads, S, head_dim) -> (BS, S, n_heads, head_dim) -> (BS, S, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    output = linear_quantized(
        output,
        weights["self_attn.o_proj.weight"],
        weights.get("self_attn.o_proj.weight_scales"),
        original_shape=weights.get("self_attn.o_proj.weight_orig_shape"),
        bias=weights.get("self_attn.o_proj.bias"),
    )
    return output


def functional_transformer_block(
    x: torch.Tensor,
    weights,
    cache_k,
    cache_v,
    start_pos: int,
    freqs_rope: torch.Tensor,
    config,
    mask: Optional[torch.Tensor],
):
    # Params needed for GQA
    n_rep = config["num_attention_heads"] // config["num_key_value_heads"]
    n_kv_heads = config["num_key_value_heads"]
    n_heads = config["num_attention_heads"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    model_type = config["model_type"]

    attended_x = functional_rmsnorm(x, weights["input_layernorm.weight"])
    hidden = x + functional_gqa_quantized(
        attended_x,
        start_pos,
        weights,
        cache_k,
        cache_v,
        freqs_rope,
        n_rep,
        n_kv_heads,
        n_heads,
        head_dim,
        mask,
        model_type,
    )
    attended_hidden = functional_rmsnorm(
        hidden, weights["post_attention_layernorm.weight"]
    )
    if model_type == "phi3":
        output = hidden + functional_ffn_phi3_quantized(attended_hidden, weights)
    else:
        output = hidden + functional_ffn_quantized(attended_hidden, weights)
    return output


def move_to_device_recursive(data, device):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                move_to_device_recursive(value, device)
            elif not key.endswith("_orig_shape"):
                data[key] = value.to(device)
    else:
        raise TypeError("Input must be a dictionary")


class Transformer:
    def __init__(self, model_dir, config, device="cpu"):
        model_dir = model_dir
        self.config = config
        self.device = device
        if self.config["model_type"] in ["llama", "mistral"]:
            self.freqs_rope = LLAMA3_PositionalEmbeddings.precompute_rope_constants(
                self.config["hidden_size"] // self.config["num_attention_heads"],
                self.config["max_position_embeddings"] * 2,
                self.config["rope_theta"],
            ).to(self.device)
        elif self.config["model_type"] in ["phi3", "granite-small"]:
            position_ids = torch.arange(
                0, self.config["max_position_embeddings"], dtype=torch.long
            )
            position_ids = position_ids.unsqueeze(0).view(
                -1, self.config["max_position_embeddings"]
            )
            self.freqs_rope = Phi3_PositionalEmbeddings.precompute_rope_constants(
                position_ids=position_ids,
                dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                base=self.config["rope_theta"],
            ).to(self.device, dtype=torch.bfloat16)  # TODO: unhardcode this precision
        self.max_seq_len = self.config["max_position_embeddings"]
        self.num_layers = self.config["num_hidden_layers"]
        self.caches_memory = build_kv_caches(self.config, device=self.device)

        self.chunk_weights = load_block_chunk(
            model_dir, 0
        )  # assume all weights are in single chunk
        self.general_chunk_weights = load_general_chunk(model_dir)
        move_to_device_recursive(self.chunk_weights, self.device)
        move_to_device_recursive(self.general_chunk_weights, self.device)
        if config["tie_word_embeddings"]:
            self.general_chunk_weights["lm_head.weight"] = self.general_chunk_weights[
                "model.embed_tokens.weight"
            ]

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # emb O(0.0001)s
        # all blocks O(0.25)s
        # head 0(0.0001)s
        seq_len = tokens.shape[-1]
        seq_embeddings = embedding_quantized(
            tokens,
            self.general_chunk_weights["model.embed_tokens.weight"],
            scales=self.general_chunk_weights.get("model.embed_tokens.weight_scales"),
        )
        if self.config["model_type"] in ["phi3", "granite-small"]:
            freqs_rope = self.freqs_rope[:, start_pos : start_pos + seq_len]
        else:
            freqs_rope = self.freqs_rope[start_pos : start_pos + seq_len]
        mask = build_attention_mask(
            seq_len, start_pos, device=tokens.device, dtype=seq_embeddings.dtype
        )
        h = seq_embeddings
        for layer_idx in range(self.num_layers):
            cache_k = self.caches_memory[layer_idx]["k"]
            cache_v = self.caches_memory[layer_idx]["v"]
            block_weights = self.chunk_weights[layer_idx]
            h = functional_transformer_block(
                h,
                block_weights,
                cache_k,
                cache_v,
                start_pos,
                freqs_rope,
                self.config,
                mask,
            )
        h = functional_rmsnorm(h, self.general_chunk_weights["model.norm.weight"])
        output = linear_quantized(
            h,
            self.general_chunk_weights.get("lm_head.weight"),
            scales=self.general_chunk_weights.get("lm_head.weight_scales"),
            original_shape=self.general_chunk_weights.get("lm_head.weight_orig_shape"),
        ).float()
        return output
