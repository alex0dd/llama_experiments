import torch

from collections import defaultdict

from typing import Dict, List

import pickle

def load_block_chunk(model_dir, block_chunk_idx):
    with open(f'{model_dir}/blocks_chunk_{block_chunk_idx}.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

def load_general_chunk(model_dir):
    with open(f'{model_dir}/general_chunk.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

def hf_undo_permute(w: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def remap_weights_if_needed(weights: torch.Tensor, param_name: str, config) -> torch.Tensor:
    """
    Important: HF checkpoint permutes the original weights for Q, K weight tensors
    https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/llama/convert_llama_weights_to_hf.py#L174-L182
    This function will take a block of attention weights and permute only the ones that need to be permuted
    """
    if "self_attn.q_proj.weight" in param_name:
        weights = hf_undo_permute(
            weights,
            n_heads=config["num_attention_heads"],
            dim1=config["hidden_size"],
            dim2=config["hidden_size"],
        )
    elif "self_attn.k_proj.weight" in param_name:
        # NOTE: for llama 3 70B this will need to be fixed further, as num_shards will be different
        num_shards = 1
        num_key_value_heads = config["num_key_value_heads"]
        n_heads = config["num_attention_heads"]
        n_heads_per_shard = n_heads // num_shards
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        
        weights = hf_undo_permute(
            weights,
            n_heads=config["num_key_value_heads"],
            dim1=config["hidden_size"] // num_local_key_value_heads,
            dim2=config["hidden_size"],
        )
    return weights

def get_all_layer_names_in_block(layer_idx: int, config) -> Dict[str, str]:
    """
    Returns a dictionary with keys being layer names for a given transformer block
    in HF format and values being their mapping to be used within Transformer model.
    """
    if config["model_type"] == "phi3":
        attn_letters = ['o', 'qkv']
        weight_remap = {
            f'model.layers.{layer_idx}.mlp.down_proj.weight': 'mlp.down_proj.weight',
            f'model.layers.{layer_idx}.mlp.gate_up_proj.weight': 'mlp.gate_up_proj.weight',
            f'model.layers.{layer_idx}.input_layernorm.weight': 'input_layernorm.weight',
            f'model.layers.{layer_idx}.post_attention_layernorm.weight': 'post_attention_layernorm.weight'
        }
    else:
        attn_letters = ['q', 'k', 'v', 'o'] 
        weight_remap = {
            f'model.layers.{layer_idx}.mlp.down_proj.weight': 'mlp.down_proj.weight',
            f'model.layers.{layer_idx}.mlp.gate_proj.weight': 'mlp.gate_proj.weight',
            f'model.layers.{layer_idx}.mlp.up_proj.weight': 'mlp.up_proj.weight',
            f'model.layers.{layer_idx}.input_layernorm.weight': 'input_layernorm.weight',
            f'model.layers.{layer_idx}.post_attention_layernorm.weight': 'post_attention_layernorm.weight'
        }
    
    for letter in attn_letters:
        orig_key = f'model.layers.{layer_idx}.self_attn.{letter}_proj.weight'
        new_key = f'self_attn.{letter}_proj.weight'
        weight_remap[orig_key] = new_key
    return weight_remap

def load_multiple_transformer_block_weights_and_remap(parser, config, layer_idxs, device):
    """
    Helper function that loads all the weights for given transformer block indices minimizing
    the number of file reads, and transfers them to the given decide.
    """
    num_layers = config["num_hidden_layers"]
    loaded_weights = defaultdict(dict)
    all_weights_remap = {}
    # Get all the layer names
    for idx in layer_idxs:
        if idx < num_layers:
            weight_remap = get_all_layer_names_in_block(idx, config=config)
            all_weights_remap.update(weight_remap)
    # Load all weights opening the F files exactly F times.
    all_loaded_weights = parser.get_tensors(list(all_weights_remap.keys()))
    # Assign the loaded weights to the appropriate loaded_weights dict fields.
    for weight_name in all_loaded_weights.keys():
        layer_idx = int(weight_name.split(".")[2])
        remapped_weight_name = all_weights_remap[weight_name]
        # Move the original loaded weight with original weight name, into the loaded_weights dict (and repermute it if needed, e.g. in attn q, k cases)
        final_weights = remap_weights_if_needed(all_loaded_weights[weight_name], weight_name, config)
        loaded_weights[layer_idx][remapped_weight_name] = final_weights.to(device)         
    return loaded_weights

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
    if "torch." not in config["torch_dtype"]: torch_dtype = "torch."+config["torch_dtype"]
    else: torch_dtype = config["torch_dtype"]
    # We need this, as pytorch can't build a type instance from string
    dtype_map ={
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
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    for layer_idx in range(config["num_hidden_layers"]):
        caches_memory[layer_idx] = {}
        caches_memory[layer_idx]["k"] = torch.zeros((max_bs, max_seq_len, n_kv_heads, head_dim), dtype=dtype_map[torch_dtype], device=device)
        caches_memory[layer_idx]["v"] = torch.zeros((max_bs, max_seq_len, n_kv_heads, head_dim), dtype=dtype_map[torch_dtype], device=device)
    return caches_memory

@torch.inference_mode()
def build_attention_mask(seq_len, start_pos, device, dtype):
    """
    Builds a sequence mask tensor for attention modules.
    """
    mask = None
    if seq_len > 1:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        if device.type == "mps":
            # https://github.com/pytorch/pytorch/issues/100005
            mask = torch.nan_to_num(mask, nan=0.0)
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seq_len, start_pos), device=device), mask]
        ).to(dtype)
    return mask

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