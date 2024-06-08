import torch
import math
from typing import Optional, Tuple

"""
'model.layers.{i}.input_layernorm.weight',
'model.layers.{i}.mlp.down_proj.weight',
'model.layers.{i}.mlp.gate_proj.weight',
'model.layers.{i}.mlp.up_proj.weight',
'model.layers.{i}.post_attention_layernorm.weight',
'model.layers.{i}.self_attn.k_proj.weight',
'model.layers.{i}.self_attn.o_proj.weight',
'model.layers.{i}.self_attn.q_proj.weight',
'model.layers.{i}.self_attn.v_proj.weight'
"""
"""
>>> tensor_parser.shape("model.layers.1.mlp.down_proj.weight")
[4096, 14336]
>>> tensor_parser.shape("model.layers.1.mlp.gate_proj.weight")
[14336, 4096]
>>> tensor_parser.shape("model.layers.1.mlp.up_proj.weight")
[14336, 4096]
"""

def embedding_matrix(inputs, weights):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    This function is taken and adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """ 
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

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

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_rope: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE:
        - https://blog.eleuther.ai/rotary-embeddings/
        - https://github.com/meta-llama/llama3/blob/main/llama/model.py
    This function is taken and adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    It applies rotatory positional embedding to query and key vectors, performing only one reshape of rope frequencies.
    """ 
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_rope = reshape_for_broadcast(freqs_rope, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_rope).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_rope).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_rope_constants(dim: int, end: int, theta: float = 10000.0):
    """
    RoPE:
        - https://blog.eleuther.ai/rotary-embeddings/
        - https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """ 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_rope = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_rope

class Transformer:
    def __init__(self, config):
        self.freqs_rope = precompute_rope_constants(
            config["hidden_size"] // config["num_attention_heads"],
            config["max_position_embeddings"] * 2,
            config["rope_theta"],
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        seq_len = tokens.shape[-1]
        seq_embeddings = embedding_matrix(tokens, embedding_weights)
        freqs_cis = self.freqs_rope[start_pos : start_pos + seq_len]

class TransformerBlock:
    def __init__(self):
        pass

class MLP:
    def __init__(self):
        pass

class RMSNorm:
    """
    Root Mean Square Layer Normalization:
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, eps=1e-05):
        #super().__init__()
        self.eps = eps
        pass

    def _normalization(self, inputs: torch.Tensor) -> torch.Tensor:
        # Assume inputs of shape (B, S, n)
        rms_a = inputs.pow(2).mean(-1, keepdim=True)
        rms_a = inputs * torch.rsqrt(rms_a + self.eps)
        return rms_a

    def forward(self, inputs: torch.Tensor, weights) -> torch.Tensor:
        """
        In the original paper inputs = a, weights = g (gain)
        """
        # TODO: make this conversion generalizable as a function
        # Perform operation in fp32 precision
        output = self._normalization(inputs.float()).type_as(inputs)
        return output * weights

class Attention:
    def __init__(self, config):
        self.config = config
        self.n_rep = config["num_attention_heads"] // config["num_key_value_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.n_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]
        """
        KV cache, memory occupation: MAX_BS*MAX_SEQ_LEN*N_KV_HEADS*HEAD_DIM*2 (2 because we have K and V)

        MAX_BS = 1
        MAX_SEQ_LEN = 2048 (can be up to 8192)
        N_KV_HEADS = 8
        HEAD_DIM = dim//n_heads=4096//32=128

        Total per layer = 1 * 2048 * 8 * 128 * 2 = 4194304 entries
        Bytes per layer = 4194304 * 4 (assuming float) = 16777216 bytes (16MB per layer)
        Bytes per model = 16777216 * NUM_LAYERS = 16777216 * 32 = 536870912 (512MB per model)
        """
        MAX_BS = 1
        MAX_SEQ_LEN = 2048
        # TODO: move this outside of the class, so we can store it when this layer gets deleted
        self.cache_k = torch.zeros((MAX_BS, MAX_SEQ_LEN, self.n_kv_heads, self.head_dim), dtype=self.config["torch_dtype"])
        self.cache_v = torch.zeros((MAX_BS, MAX_SEQ_LEN, self.n_kv_heads, self.head_dim), dtype=self.config["torch_dtype"])

    def get_cache(self):
        return (self.cache_k, self.cache_v)

    def set_cache(self, cache_k, cache_v):
        self.cache_k = cache_k
        self.cache_v = cache_v

    def forward(self, x: torch.Tensor, start_pos: int, weights, freqs_rope: torch.Tensor, mask: Optional[torch.Tensor]):
        bs, seq_len, _ = x.shape
        
        # Apply attention transformation matrices
        # (BS, S, dim) -> (BS, S, n_heads * head_dim)
        xq = torch.nn.functional.linear(x, weights["self_attn.q_proj.weight"])
        # (BS, S, dim) -> (BS, S, n_kv_heads * head_dim)
        xk = torch.nn.functional.linear(x, weights["self_attn.k_proj.weight"])
        xv = torch.nn.functional.linear(x, weights["self_attn.v_proj.weight"])

        # Reshapes
        # (BS, S, n_heads * head_dim) -> (BS, S, n_heads, head_dim)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        # (BS, S, n_kv_heads * head_dim) -> (BS, S, n_kv_heads, head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_rope=freqs_rope)

        # Populate KV cache
        if self.cache_k is not None:
            self.cache_k[:bs, start_pos : start_pos + seq_len] = xk
            keys = self.cache_k[:bs, :start_pos + seq_len]
        else:
            # TODO: check if sequence length here is correct, given we start from pos=0
            keys = xk
        if self.cache_v is not None:
            self.cache_v[:bs, start_pos : start_pos + seq_len] = xv
            values = self.cache_v[:bs, :start_pos + seq_len]
        else:
            # TODO: check if sequence length here is correct, given we start from pos=0
            values = xv

        # Pad keys and values from n_kv_heads to n_heads, if needed (if n_kv_heads < n_heads)
        keys = repeat_kv(keys, self.n_rep)  # (BS, cache_len + S, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (BS, cache_len + S, n_heads, head_dim)

        # Reshapes
        xq = xq.transpose(1, 2)  # (BS, S, n_heads, head_dim) -> (bs, n_heads, S, head_dim)
        keys = keys.transpose(1, 2)  # (BS, cache_len + S, n_heads, head_dim) -> (BS, n_heads, cache_len + S, head_dim)
        values = values.transpose(1, 2)  # (BS, cache_len + S, n_heads, head_dim) -> (BS, n_heads, cache_len + S, head_dim)

        # Matmul -> (BS, n_heads, S, head_dim) @ (BS, n_heads, head_dim, cache_len + S) -> (BS, n_heads, S, cache_len + S)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        # Matmul -> (BS, n_heads, S, cache_len + S) @ (BS, n_heads, cache_len + S, head_dim) -> (BS, n_heads, S, head_dim)
        output = torch.matmul(scores, values)
        # # (BS, n_heads, S, head_dim) -> (BS, S, n_heads, head_dim) -> (BS, S, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        output = torch.nn.functional.linear(output, weights["self_attn.o_proj.weight"])
        return output