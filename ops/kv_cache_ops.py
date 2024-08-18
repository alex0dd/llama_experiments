import torch

from ops.utils import get_head_dim

def build_kv_caches(config, device, max_seq_len=2048, max_bs=4):
    """
    Builds N_layers KV caches of given size dictionary and places the caches on a given decide.

    KV cache, memory occupation: MAX_BS*MAX_SEQ_LEN*N_KV_HEADS*HEAD_DIM*2 (2 because we have K and V)

    MAX_BS = 1
    MAX_SEQ_LEN = 2048 (can be up to 8192)
    N_KV_HEADS = 8
    HEAD_DIM = dim//n_heads=4096//32=128

    Total per layer = 1 * 2048 * 8 * 128 * 2 = 4194304 entries
    Bytes per layer = 4194304 * 4 (assuming float) = 16777216 bytes (16MB per layer)
    Bytes per model = 16777216 * NUM_LAYERS = 16777216 * 32 = 536870912 (512MB per model)
    """
    if "torch." not in config["torch_dtype"]:
        torch_dtype = "torch." + config["torch_dtype"]
    else:
        torch_dtype = config["torch_dtype"]
    # We need this, as pytorch can't build a type instance from string
    dtype_map = {
        "torch.bfloat16": torch.bfloat16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.float16": torch.float16,
        "torch.half": torch.float16,
        "torch.float": torch.float32,
        "torch.float32": torch.float32,
    }

    caches_memory = {}
    n_kv_heads = config["num_key_value_heads"]
    head_dim = get_head_dim(config)
    for layer_idx in range(config["num_hidden_layers"]):
        caches_memory[layer_idx] = KVCache(max_bs, max_seq_len, n_kv_heads, head_dim).to(device)
    return caches_memory

class KVCache(torch.nn.Module):
    """
    Taken for adaptation from: https://github.com/pytorch/pytorch/blob/main/benchmarks/gpt_fast/model.py
    """
    
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        # TODO: fix this once input_pos is fixed to be [BS, 1]
        # assert input_pos.shape[0] == k_val.shape[2]

        bs = k_val.shape[0]
        seq_len = k_val.shape[1]
        
        self.k_cache[:bs, input_pos : input_pos + seq_len] = k_val
        self.v_cache[:bs, input_pos : input_pos + seq_len] = v_val

        k_out = self.k_cache[:bs, : input_pos + seq_len]
        v_out = self.v_cache[:bs, : input_pos + seq_len]

        return k_out, v_out