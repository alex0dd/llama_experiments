import torch

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

    def _normalization(self, inputs):
        # Assume inputs of shape (B, S, n)
        rms_a = inputs.pow(2).mean(-1, keepdim=True)
        rms_a = inputs * torch.rsqrt(rms_a + self.eps)
        return rms_a

    def forward(self, inputs, weights):
        """
        In the original paper inputs = a, weights = g (gain)
        """
        # TODO: make this conversion generalizable as a function
        # Perform operation in fp32 precision
        output = self._normalization(inputs.float()).type_as(inputs)
        return output * weights

class Attention:
    def __init__(self):
        pass