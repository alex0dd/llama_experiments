from collections import defaultdict
import torch
import math
from typing import Dict, Optional, Tuple

from .utils import build_kv_caches, build_attention_mask, remap_weights_if_needed, load_multiple_transformer_block_weights_and_remap
from .rope import apply_rotary_emb, precompute_rope_constants

def embedding_matrix(inputs, weights):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)

@torch.jit.script
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
@torch.jit.script
def functional_ffn(x: torch.Tensor, weights: Dict[str, torch.Tensor]):
    output = torch.nn.functional.linear(x, weights["mlp.gate_proj.weight"])
    output = torch.nn.functional.silu(output) * torch.nn.functional.linear(x, weights["mlp.up_proj.weight"])
    output = torch.nn.functional.linear(output, weights["mlp.down_proj.weight"])
    return output

def _normalization(inputs: torch.Tensor, eps:float = 1e-5) -> torch.Tensor:
    # Assume inputs of shape (B, S, n)
    rms_a = inputs.pow(2).mean(-1, keepdim=True)
    rms_a = inputs * torch.rsqrt(rms_a + eps)
    return rms_a

@torch.inference_mode()
@torch.jit.script
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
#@torch.jit.script
def functional_gqa(
        x: torch.Tensor, 
        start_pos: int, 
        weights: Dict[str, torch.Tensor], 
        cache_k: torch.Tensor, 
        cache_v: torch.Tensor, 
        freqs_rope: torch.Tensor, 
        config, 
        mask: Optional[torch.Tensor]
    ):
    n_rep = config["num_attention_heads"] // config["num_key_value_heads"]
    n_kv_heads = config["num_key_value_heads"]
    n_heads = config["num_attention_heads"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]
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
    xq, xk = apply_rotary_emb(xq, xk, freqs_rope=freqs_rope)

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

    # Reshapes
    xq = xq.transpose(1, 2)  # (BS, S, n_heads, head_dim) -> (bs, n_heads, S, head_dim)
    keys = keys.transpose(1, 2)  # (BS, cache_len + S, n_heads, head_dim) -> (BS, n_heads, cache_len + S, head_dim)
    values = values.transpose(1, 2)  # (BS, cache_len + S, n_heads, head_dim) -> (BS, n_heads, cache_len + S, head_dim)

    # Matmul -> (BS, n_heads, S, head_dim) @ (BS, n_heads, head_dim, cache_len + S) -> (BS, n_heads, S, cache_len + S)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)

    if mask is not None:
        scores = scores + mask

    scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
    # Matmul -> (BS, n_heads, S, cache_len + S) @ (BS, n_heads, cache_len + S, head_dim) -> (BS, n_heads, S, head_dim)
    output = torch.matmul(scores, values)
    # # (BS, n_heads, S, head_dim) -> (BS, S, n_heads, head_dim) -> (BS, S, n_heads * head_dim)
    output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    output = torch.nn.functional.linear(output, weights["self_attn.o_proj.weight"])
    return output

def functional_transformer_block(x: torch.Tensor, weights, cache_k, cache_v, start_pos: int, freqs_rope: torch.Tensor, config, mask: Optional[torch.Tensor]):
    attended_x = functional_rmsnorm(x, weights["input_layernorm.weight"])
    hidden = x + functional_gqa(attended_x, start_pos, weights, cache_k, cache_v, freqs_rope, config, mask)
    attended_hidden = functional_rmsnorm(hidden, weights["post_attention_layernorm.weight"])
    output = hidden + functional_ffn(attended_hidden, weights)
    return output

import time

class Transformer:
    def __init__(self, config, parser, device="cpu", preload_n_transformer_blocks = 16):
        self.device=device
        self.freqs_rope = precompute_rope_constants(
            config["hidden_size"] // config["num_attention_heads"],
            config["max_position_embeddings"] * 2,
            config["rope_theta"],
        ).to(self.device)
        self.preload_n_transformer_blocks = preload_n_transformer_blocks
        self.config = config
        self.num_layers = config["num_hidden_layers"]
        self.parser = parser
        self.caches_memory = build_kv_caches(config, device=self.device)

        self.embedding_weights = parser.get_tensor('model.embed_tokens.weight').to(self.device)
        if config["tie_word_embeddings"]:
            self.output_embedding_weights = self.embedding_weights
        else:
            self.output_embedding_weights = parser.get_tensor('lm_head.weight')
        self.output_embedding_weights = self.output_embedding_weights.to(self.device)
        self.output_norm_weights = parser.get_tensor('model.norm.weight').to(self.device)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        seq_len = tokens.shape[-1]
        seq_embeddings = embedding_matrix(tokens, self.embedding_weights)
        freqs_rope = self.freqs_rope[start_pos : start_pos + seq_len]
        mask = build_attention_mask(seq_len, start_pos, device=tokens.device, dtype=seq_embeddings.dtype)
        h = seq_embeddings

        current_transformer_blocks_loaded = None
        layer_idxs_to_load = []
        
        for layer_idx in range(self.num_layers):
            cache_k = self.caches_memory[layer_idx]["k"]
            cache_v = self.caches_memory[layer_idx]["v"]
            if len(layer_idxs_to_load) == 0:
                del current_transformer_blocks_loaded
                current_transformer_blocks_loaded = None
                layer_idxs_to_load = [layer_idx + i for i in range(self.preload_n_transformer_blocks)]
            if current_transformer_blocks_loaded is None:
                print(f"Beginning to load layers: {layer_idxs_to_load}")
                start_t = time.time()
                current_transformer_blocks_loaded = load_multiple_transformer_block_weights_and_remap(self.parser, self.config, layer_idxs_to_load, device=tokens.device)
                delta_t = time.time() - start_t
                print(f"Finished to load layers: {layer_idxs_to_load} in {delta_t} seconds.")
            block_weights = current_transformer_blocks_loaded[layer_idx]
            layer_idxs_to_load.pop(0) # remove index as "consumed"
            
            #block_weights = self._load_weights_for_block(self.config, layer_idx, device=tokens.device)
            h = functional_transformer_block(h, block_weights, cache_k, cache_v, start_pos, freqs_rope, self.config, mask)
            # delete weights after inference on that block
            del block_weights
            
        h = functional_rmsnorm(h, self.output_norm_weights)
        output = torch.nn.functional.linear(h, self.output_embedding_weights).float()
        return output