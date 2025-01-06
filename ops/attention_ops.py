import torch
from ops.apple_attention import split_einsum_v2, split_einsum

@torch.jit.script
def base_attn(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    normalization_factor: float,
    attn_logit_softcapping: float = 0.0
) -> torch.Tensor:
    scores = (q * normalization_factor) @ k.transpose(3, 2)
    if attn_logit_softcapping > 0.0:
        scores = scores / attn_logit_softcapping
        scores = torch.tanh(scores)
        scores = scores * attn_logit_softcapping
    scores = scores + mask
    scores = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = scores @ v
    return output

def base_attn_unopt(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    normalization_factor: float,
    attn_logit_softcapping: float = 0.0
) -> torch.Tensor:
    scores = (q * normalization_factor) @ k.transpose(3, 2)
    if attn_logit_softcapping > 0.0:
        scores = scores / attn_logit_softcapping
        scores = torch.tanh(scores)
        scores = scores * attn_logit_softcapping
    if mask is not None:
        scores = scores + mask
    scores = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = scores @ v
    return output

@torch.jit.script
def apple_attn_wrapper(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    head_dim: int
) -> torch.Tensor:
    # Q needs to become (bs, dim_head, heads, seq_len)
    # K needs to become (bs, dim_head, heads, seq_len) 
    #   but internally needs to be transposed to (bs, seq_len, heads, dim_head).
    #   So we change the internals and transpose it here directly to that format.
    # V needs to become (bs, dim_head, heads, seq_len)
    # Mask needs to become (bs, seq_len, n_heads, seq_len)

    # Q= torch.Size([1, 32, 13, 128]) -> [bs, heads, seq_len, dim_head]
    # K= torch.Size([1, 32, 13, 128]) -> [bs, heads, seq_len, dim_head]
    # V= torch.Size([1, 32, 13, 128]) -> [bs, heads, seq_len, dim_head]
    # Mask= torch.Size([13, 13]) -> [seq_len, seq_len]
    # output= torch.Size([1, 12, 32, 128])

    #import torch
    #from ops.apple_attention import split_einsum
    #q = torch.randn(1, 32, 13, 128)
    #k = torch.randn(1, 32, 13, 128)
    #v = torch.randn(1, 32, 13, 128)
    #mask = torch.randn(13, 13)
    #head_dim = 128
    heads = q.shape[1]
    perm_q = torch.permute(q, (0, 3, 1, 2))
    perm_k = torch.permute(k, (0, 2, 1, 3))
    perm_v = torch.permute(v, (0, 3, 1, 2))
    mask = mask.unsqueeze(1)
    attn_result = split_einsum_v2(perm_q, perm_k, perm_v, mask, heads, head_dim) # [1, 128, 32, 13]
    attn_result = torch.transpose(attn_result, 1, 3)
    attn_result = torch.transpose(attn_result, 1, 2)
    return attn_result