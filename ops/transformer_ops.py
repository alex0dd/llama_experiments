from collections import defaultdict
import torch
import math
from typing import Dict, Optional, Tuple

from .utils import build_kv_caches, build_attention_mask, remap_weights_if_needed, load_multiple_transformer_block_weights_and_remap
from .rope import apply_rotary_emb, precompute_rope_constants

from quantization.utils_int4 import linear_forward_int4, find_multiple

def linear_int8(inputs, weight, scales, original_shape=None):
    return torch.nn.functional.linear(inputs, weight.to(dtype=inputs.dtype)) * scales

def embedding_int8(inputs, weights, scales, original_shape=None):
    embs = torch.nn.functional.embedding(inputs, weights)
    return embs * scales #torch.embedding(input, weight) * scales

def linear_int4(inputs, weight, scales_and_zeros, original_shape, groupsize=128, padding=True):
    inputs = inputs.to(torch.bfloat16)

    out_features = original_shape[0]
    in_features = original_shape[1]

    origin_in_features = in_features

    in_features = find_multiple(in_features, 1024)

    assert out_features % 8 == 0, "require out_features % 8 == 0"
    
    # TODO: add behaviour if padding is needed (https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py#L483-L525)
    if padding:
        input = torch.nn.functional.pad(inputs, pad=(0, in_features - origin_in_features))
    return linear_forward_int4(inputs, weight, scales_and_zeros, out_features, groupsize)

def embedding_matrix(inputs, weights, scales=None, original_shape=None):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)

linear_quantized = linear_int8
#linear_quantized = linear_int4

embedding_quantized = embedding_int8
# embedding_quantized = embedding_matrix

#@torch.jit.script
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function is taken from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    It should be equivalent to torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

@torch.inference_mode()
#@torch.jit.script
def functional_ffn(x: torch.Tensor, weights: Dict[str, torch.Tensor]):
    output = torch.nn.functional.linear(x, weights["mlp.gate_proj.weight"])
    output = torch.nn.functional.silu(output) * torch.nn.functional.linear(x, weights["mlp.up_proj.weight"])
    output = torch.nn.functional.linear(output, weights["mlp.down_proj.weight"])
    return output

@torch.inference_mode()
#@torch.jit.script
def functional_ffn_quantized(x: torch.Tensor, weights: Dict[str, torch.Tensor]):
    output = linear_quantized(x, weights["mlp.gate_proj.weight"], weights["mlp.gate_proj.weight_scales"], weights["mlp.gate_proj.weight_orig_shape"])
    output = torch.nn.functional.silu(output) * linear_quantized(x, weights["mlp.up_proj.weight"], weights["mlp.up_proj.weight_scales"], weights["mlp.up_proj.weight_orig_shape"])
    output = linear_quantized(output, weights["mlp.down_proj.weight"], weights["mlp.down_proj.weight_scales"], weights["mlp.down_proj.weight_orig_shape"])
    return output

def _normalization(inputs: torch.Tensor, eps:float = 1e-5) -> torch.Tensor:
    # Assume inputs of shape (B, S, n)
    rms_a = inputs.pow(2).mean(-1, keepdim=True)
    rms_a = inputs * torch.rsqrt(rms_a + eps)
    return rms_a

@torch.inference_mode()
#@torch.jit.script
def functional_rmsnorm(inputs: torch.Tensor, weights: torch.Tensor, eps:float = 1e-5) -> torch.Tensor:
    """
    Root Mean Square Layer Normalization:
    https://arxiv.org/abs/1910.07467

    In the original paper inputs = a, weights = g (gain)
    """
    # TODO: make this conversion generalizable as a function
    # Perform operation in fp32 precision
    output = _normalization(inputs.float()).type_as(inputs)
    return output * weights

@torch.inference_mode()
##@torch.jit.script
def functional_gqa(
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
        mask: Optional[torch.Tensor]
    ):
    bs, seq_len, _ = x.shape
    
    # Apply attention transformation matrices
    # (BS, S, dim) -> (BS, S, n_heads * head_dim)
    xq = torch.nn.functional.linear(x, weights["self_attn.q_proj.weight"])
    # (BS, S, dim) -> (BS, S, n_kv_heads * head_dim)
    xk = torch.nn.functional.linear(x, weights["self_attn.k_proj.weight"])
    xv = torch.nn.functional.linear(x, weights["self_attn.v_proj.weight"])

    # Reshapes
    # (BS, S, n_heads * head_dim) -> (BS, S, n_heads, head_dim)
    xq = xq.view(bs, seq_len, n_heads, head_dim)
    # (BS, S, n_kv_heads * head_dim) -> (BS, S, n_kv_heads, head_dim)
    xk = xk.view(bs, seq_len, n_kv_heads, head_dim)
    xv = xv.view(bs, seq_len, n_kv_heads, head_dim)
    
    # Apply positional embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_rope)

    # Populate KV cache
    if cache_k is not None:
        cache_k[:bs, start_pos : start_pos + seq_len] = xk
        keys = cache_k[:bs, :start_pos + seq_len]
    else:
        # TODO: check if sequence length here is correct, given we start from pos=0
        keys = xk
    if cache_v is not None:
        cache_v[:bs, start_pos : start_pos + seq_len] = xv
        values = cache_v[:bs, :start_pos + seq_len]
    else:
        # TODO: check if sequence length here is correct, given we start from pos=0
        values = xv

    # Pad keys and values from n_kv_heads to n_heads, if needed (if n_kv_heads < n_heads)
    keys = repeat_kv(keys, n_rep)  # (BS, cache_len + S, n_heads, head_dim)
    values = repeat_kv(values, n_rep)  # (BS, cache_len + S, n_heads, head_dim)

    xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, keys, values))

    output = torch.nn.functional.scaled_dot_product_attention(xq, keys, values, attn_mask=mask, dropout_p=0.0)
    # # (BS, n_heads, S, head_dim) -> (BS, S, n_heads, head_dim) -> (BS, S, n_heads * head_dim)
    output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    output = torch.nn.functional.linear(output, weights["self_attn.o_proj.weight"])
    return output

@torch.inference_mode()
##@torch.jit.script
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
        mask: Optional[torch.Tensor]
    ):
    bs, seq_len, _ = x.shape
    # Apply attention transformation matrices
    # (BS, S, dim) -> (BS, S, n_heads * head_dim)
    xq = linear_quantized(x, weights["self_attn.q_proj.weight"], weights["self_attn.q_proj.weight_scales"], weights["self_attn.q_proj.weight_orig_shape"])
    # (BS, S, dim) -> (BS, S, n_kv_heads * head_dim)
    xk = linear_quantized(x, weights["self_attn.k_proj.weight"], weights["self_attn.k_proj.weight_scales"], weights["self_attn.k_proj.weight_orig_shape"])
    xv = linear_quantized(x, weights["self_attn.v_proj.weight"], weights["self_attn.v_proj.weight_scales"], weights["self_attn.v_proj.weight_orig_shape"])

    # Reshapes
    # (BS, S, n_heads * head_dim) -> (BS, S, n_heads, head_dim)
    xq = xq.view(bs, seq_len, n_heads, head_dim)
    # (BS, S, n_kv_heads * head_dim) -> (BS, S, n_kv_heads, head_dim)
    xk = xk.view(bs, seq_len, n_kv_heads, head_dim)
    xv = xv.view(bs, seq_len, n_kv_heads, head_dim)
    
    # Apply positional embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_rope)

    # Populate KV cache
    if cache_k is not None:
        cache_k[:bs, start_pos : start_pos + seq_len] = xk
        keys = cache_k[:bs, :start_pos + seq_len]
    else:
        # TODO: check if sequence length here is correct, given we start from pos=0
        keys = xk
    if cache_v is not None:
        cache_v[:bs, start_pos : start_pos + seq_len] = xv
        values = cache_v[:bs, :start_pos + seq_len]
    else:
        # TODO: check if sequence length here is correct, given we start from pos=0
        values = xv
    # Pad keys and values from n_kv_heads to n_heads, if needed (if n_kv_heads < n_heads)
    keys = repeat_kv(keys, n_rep)  # (BS, cache_len + S, n_heads, head_dim)
    values = repeat_kv(values, n_rep)  # (BS, cache_len + S, n_heads, head_dim)

    xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, keys, values))
    output = torch.nn.functional.scaled_dot_product_attention(xq, keys, values, attn_mask=mask, dropout_p=0.0)
    # # (BS, n_heads, S, head_dim) -> (BS, S, n_heads, head_dim) -> (BS, S, n_heads * head_dim)
    output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    output = linear_quantized(output, weights["self_attn.o_proj.weight"], weights["self_attn.o_proj.weight_scales"], weights["self_attn.o_proj.weight_orig_shape"])
    return output

def functional_transformer_block(x: torch.Tensor, weights, cache_k, cache_v, start_pos: int, freqs_rope: torch.Tensor, config, mask: Optional[torch.Tensor]):
    # Params needed for GQA
    n_rep = config["num_attention_heads"] // config["num_key_value_heads"]
    n_kv_heads = config["num_key_value_heads"]
    n_heads = config["num_attention_heads"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    attended_x = functional_rmsnorm(x, weights["input_layernorm.weight"])
    #hidden = x + functional_gqa(attended_x, start_pos, weights, cache_k, cache_v, freqs_rope, n_rep, n_kv_heads, n_heads, head_dim, mask)
    hidden = x + functional_gqa_quantized(attended_x, start_pos, weights, cache_k, cache_v, freqs_rope, n_rep, n_kv_heads, n_heads, head_dim, mask)
    attended_hidden = functional_rmsnorm(hidden, weights["post_attention_layernorm.weight"])
    #output = hidden + functional_ffn(attended_hidden, weights)
    output = hidden + functional_ffn_quantized(attended_hidden, weights)
    return output

import time

import pickle
def load_block_chunk(model_dir, block_chunk_idx):
    with open(f'{model_dir}/blocks_chunk_{block_chunk_idx}.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

def load_general_chunk(model_dir):
    with open(f'{model_dir}/general_chunk.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

def move_to_device(block_chunk, device):
    for layer_idx in block_chunk.keys():
        for layer_name in block_chunk[layer_idx].keys():
            if not layer_name.endswith("_orig_shape"):
                block_chunk[layer_idx][layer_name] = block_chunk[layer_idx][layer_name].to(device)

class Transformer:
    def __init__(self, config, device="cpu", preload_n_transformer_blocks = 16):
        model_dir = "LLAMA3-8B-PKL-int8"
        self.device=device
        self.freqs_rope = precompute_rope_constants(
            config["hidden_size"] // config["num_attention_heads"],
            config["max_position_embeddings"] * 2,
            config["rope_theta"],
        ).to(self.device)
        self.preload_n_transformer_blocks = preload_n_transformer_blocks
        self.config = config
        self.num_layers = config["num_hidden_layers"]
        self.caches_memory = build_kv_caches(config, device=self.device)

        # TODO: to support original precision models, for each weight add a dummy entry for "*_scales" and "orig_shapes" keys, and None value.
        self.chunk_weights = load_block_chunk(model_dir, 0) # assume all weights are in single chunk
        move_to_device(self.chunk_weights, self.device)
        self.general_chunk_weights = load_general_chunk(model_dir)
        
        # TODO: simplify this behavior (tie embeddings + scales/shapes assigning)
        self.embedding_weights = self.general_chunk_weights['model.embed_tokens.weight'].to(self.device)
        if 'model.embed_tokens.weight_scales' in self.general_chunk_weights:
            self.embedding_weights_scales = self.general_chunk_weights['model.embed_tokens.weight_scales'].to(self.device)
        else:
            self.embedding_weights_scales = None
        
        if config["tie_word_embeddings"]:
            self.output_embedding_weights = self.embedding_weights
            self.output_embedding_scales = self.embedding_weights_scales
        else:
            self.output_embedding_weights = self.general_chunk_weights['lm_head.weight'].to(self.device)
            if 'lm_head.weight_scales' in self.general_chunk_weights:
                self.output_embedding_scales = self.general_chunk_weights['lm_head.weight_scales'].to(self.device)
            else:
                self.output_embedding_scales = None
            if 'lm_head.weight_orig_shape' in self.general_chunk_weights:
                self.output_embedding_orig_shape = self.general_chunk_weights['lm_head.weight_orig_shape']
            else:
                self.output_embedding_orig_shape = None
        self.output_norm_weights = self.general_chunk_weights['model.norm.weight'].to(self.device)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        seq_len = tokens.shape[-1]
        seq_embeddings = embedding_quantized(tokens, self.embedding_weights, self.embedding_weights_scales)
        freqs_rope = self.freqs_rope[start_pos : start_pos + seq_len]
        mask = build_attention_mask(seq_len, start_pos, device=tokens.device, dtype=seq_embeddings.dtype)
        h = seq_embeddings
        for layer_idx in range(self.num_layers):
            cache_k = self.caches_memory[layer_idx]["k"]
            cache_v = self.caches_memory[layer_idx]["v"]
            block_weights = self.chunk_weights[layer_idx]
            h = functional_transformer_block(h, block_weights, cache_k, cache_v, start_pos, freqs_rope, self.config, mask)
        h = functional_rmsnorm(h, self.output_norm_weights)
        output = linear_quantized(h, self.output_embedding_weights, self.output_embedding_scales, self.output_embedding_orig_shape).float()
        return output