import pickle
from collections import defaultdict
from typing import Dict

import torch


def load_block_chunk(model_dir, block_chunk_idx):
    with open(f"{model_dir}/blocks_chunk_{block_chunk_idx}.pkl", "rb") as handle:
        b = pickle.load(handle)
    return b


def load_general_chunk(model_dir):
    with open(f"{model_dir}/general_chunk.pkl", "rb") as handle:
        b = pickle.load(handle)
    return b


def hf_undo_permute(
    w: torch.Tensor, n_heads: int, dim1: int, dim2: int
) -> torch.Tensor:
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )

def hf_undo_permute_gemma2(
    w: torch.Tensor, n_heads: int, dim1: int, dim2: int
) -> torch.Tensor:
    return (
        w.view(n_heads, 2, dim1 // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1 * n_heads, dim2)
    )


def remap_weights_if_needed(
    weights: torch.Tensor, param_name: str, config
) -> torch.Tensor:
    """
    Important: HF checkpoint permutes the original weights for Q, K weight tensors
    https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/llama/convert_llama_weights_to_hf.py#L174-L182
    This function will take a block of attention weights and permute only the ones that need to be permuted
    """
    if "self_attn.q_proj.weight" in param_name:
        if "gemma2" in config["model_type"]:
            # Gemma2 config has a different number of heads that's not inferrable form the other shapes
            weights = hf_undo_permute_gemma2(
                weights,
                n_heads=config["num_attention_heads"],
                dim1=config["hidden_size"] // (config["num_attention_heads"] + 1),
                dim2=config["hidden_size"],
            )
        else:
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

        if "gemma2" in config["model_type"]:
            weights = hf_undo_permute_gemma2(
                weights,
                n_heads=config["num_key_value_heads"],
                dim1=config["hidden_size"] // (config["num_attention_heads"] + 1),
                dim2=config["hidden_size"],
            )
        else:
            weights = hf_undo_permute(
                weights,
                n_heads=config["num_key_value_heads"],
                dim1=config["hidden_size"] // num_local_key_value_heads,
                dim2=config["hidden_size"],
            )
            #hf_undo_permute_gemma2(weights, n_heads=config["num_key_value_heads"], dim1=config["hidden_size"] // (config["num_attention_heads"] + 1),dim2=config["hidden_size"],)
    return weights


def get_all_layer_names_in_block(layer_idx: int, config) -> Dict[str, str]:
    """
    Returns a dictionary with keys being layer names for a given transformer block
    in HF format and values being their mapping to be used within Transformer model.
    """
    weight_remap = {}
    if config["model_type"] == "phi3":
        attn_letters = ["o", "qkv"]
        mlp_names = ["down_proj", "gate_up_proj"]
    else:
        attn_letters = ["q", "k", "v", "o"]
        mlp_names = ["down_proj", "gate_proj", "up_proj"]
    if "gemma2" in config["model_type"]:
        norm_names = ["input_layernorm", "post_attention_layernorm", "post_feedforward_layernorm", "pre_feedforward_layernorm"]
    else:
        norm_names = ["input_layernorm", "post_attention_layernorm"]
    for letter in attn_letters:
        weight_remap[f"model.layers.{layer_idx}.self_attn.{letter}_proj.weight"] = (
            f"self_attn.{letter}_proj.weight"
        )
        if "attention_bias" in config and config["attention_bias"]:
            weight_remap[f"model.layers.{layer_idx}.self_attn.{letter}_proj.bias"] = (
                f"self_attn.{letter}_proj.bias"
            )
    for mlp in mlp_names:
        weight_remap[f"model.layers.{layer_idx}.mlp.{mlp}.weight"] = f"mlp.{mlp}.weight"
        if "mlp_bias" in config and config["mlp_bias"]:
            weight_remap[f"model.layers.{layer_idx}.mlp.{mlp}.bias"] = f"mlp.{mlp}.bias"
    for norm in norm_names:
        weight_remap[f"model.layers.{layer_idx}.{norm}.weight"] = f"{norm}.weight"
    return weight_remap


def load_multiple_transformer_block_weights_and_remap(
    parser, config, layer_idxs, device, disable_llama_qk_remap=False
):
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
        if not disable_llama_qk_remap:
            final_weights = remap_weights_if_needed(
                all_loaded_weights[weight_name], weight_name, config
            )
        else:
            final_weights = all_loaded_weights[weight_name]

        if config["model_type"] in ["phi3"] and "qkv_proj" in remapped_weight_name:
            # Split qkv into 3 separate tensors
            qkv_tensor = torch.split(final_weights, final_weights.shape[0] // 3)
            q_tensor = qkv_tensor[0]
            k_tensor = qkv_tensor[1]
            v_tensor = qkv_tensor[2]
            loaded_weights[layer_idx]["self_attn.q_proj.weight"] = q_tensor.to(device)
            loaded_weights[layer_idx]["self_attn.k_proj.weight"] = k_tensor.to(device)
            loaded_weights[layer_idx]["self_attn.v_proj.weight"] = v_tensor.to(device)
        else:
            loaded_weights[layer_idx][remapped_weight_name] = final_weights.to(device)
    return loaded_weights


@torch.inference_mode()
def build_attention_mask(seq_len, start_pos, device, dtype, ignore_kv=False, fill_value=float("-inf")):
    """
    Builds a sequence mask tensor for attention modules.
    """
    mask = None
    if seq_len >= 1:
        mask = torch.full((seq_len, seq_len), fill_value, device=device)
        mask = torch.triu(mask, diagonal=1)
        if device.type == "mps":
            # https://github.com/pytorch/pytorch/issues/100005
            mask = torch.nan_to_num(mask, nan=0.0)
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        if not ignore_kv:
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=device), mask]
            ).to(dtype)
    return mask

@torch.inference_mode()
def build_attention_mask_gemma2(seq_len, min_seq_len, start_pos, device, dtype, ignore_kv=False):
    """
    Builds a sequence mask tensor for attention modules.
    """
    mask = None
    #breakpoint()
    if seq_len >= 1:
        # TODO: this is slow, bring this mask creation outside in the generator, if this function will work
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        if start_pos != 0:
            if min_seq_len <= start_pos + 1:
                mask = mask[min_seq_len:start_pos+1]#.unsqueeze(0)
            else:
                mask = mask[start_pos:min_seq_len]
        else:
            mask = mask[:min_seq_len]

        #print(start_pos, min_seq_len, seq_len, mask.shape)
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

def get_head_dim(config):
    if "head_dim" in config:
        return config["head_dim"]
    else:
        return config["hidden_size"] // config["num_attention_heads"]