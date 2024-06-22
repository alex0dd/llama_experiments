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

class Phi3_PositionalEmbeddings:

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.rotate_half
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    @torch.no_grad()
    def precompute_rope_constants(position_ids, base, dim):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return torch.cat([cos, sin])#.transpose(0, 1) # [2, seq_len, dim] -> [seq_len, 2, dim]

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (Phi3_PositionalEmbeddings.rotate_half(q) * sin)
        k_embed = (k * cos) + (Phi3_PositionalEmbeddings.rotate_half(k) * sin)
        return q_embed, k_embed

class LLAMA3_PositionalEmbeddings:
    @staticmethod
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

    @staticmethod
    @torch.inference_mode()
    @torch.jit.ignore
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
        #xq_shape = xq.shape
        #xk_shape = xk.shape
        #xq_ = torch.view_as_complex(xq.float().reshape(xq_shape[0], xq_shape[1], xq_shape[2], -1, 2))
        #xk_ = torch.view_as_complex(xk.float().reshape(xk_shape[0], xk_shape[1], xk_shape[2], -1, 2))
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_rope = reshape_for_broadcast(freqs_rope, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_rope).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_rope).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)