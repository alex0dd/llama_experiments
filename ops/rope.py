import torch

from typing import Tuple

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    This function is taken and adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """ 
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def precompute_rope_constants(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
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

@torch.inference_mode()
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