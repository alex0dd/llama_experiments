from pprint import pprint
from parsers import ModelParser

import pickle

import torch
import time
import os

from ops.utils import load_multiple_transformer_block_weights_and_remap
from quantization.utils_int8 import quantize_fp32_linear_to_int8

def is_linear_weight(layer_name):
    return "mlp." in layer_name and ".weight" in layer_name

def is_attention_linear(layer_name):
    return "self_attn." in layer_name and "proj.weight" in layer_name

def quantize_all_mlps(blocks_chunk):
    for layer_idx in blocks_chunk.keys():
        blocks_update_dict = {}
        for layer_name in blocks_chunk[layer_idx].keys():
            if is_linear_weight(layer_name) or is_attention_linear(layer_name):
                block_int8, block_scale = quantize_fp32_linear_to_int8(blocks_chunk[layer_idx][layer_name])
                blocks_chunk[layer_idx][layer_name] = block_int8
                blocks_update_dict[layer_name+"_scales"] = block_scale
        blocks_chunk[layer_idx].update(blocks_update_dict)

quantize_to_int8 = True

model_parser = ModelParser([
    "./Meta-Llama-3-8B/model-00001-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00002-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00003-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00004-of-00004.safetensors",
])

config = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  #"max_position_embeddings": 8192,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "rope_theta": 500000.0,
  "tie_word_embeddings": False,
  "torch_dtype": torch.bfloat16,
  "transformers_version": "4.40.0.dev0",
  "use_cache": True,
  "vocab_size": 128256
}

converted_model_path = "LLAMA3-8B-PKL-INT8"

num_layers = config["num_hidden_layers"]
preload_n_transformer_blocks = 32

current_transformer_blocks_loaded = None
layer_idxs_to_load = []
current_chunk = -1
for layer_idx in range(num_layers):
    if len(layer_idxs_to_load) == 0:
        current_chunk += 1
        current_transformer_blocks_loaded = None
        layer_idxs_to_load = [layer_idx + i for i in range(preload_n_transformer_blocks)]
    if current_transformer_blocks_loaded is None:
        print(f"Beginning to load layers: {layer_idxs_to_load}")
        start_t = time.time()
        current_transformer_blocks_loaded = load_multiple_transformer_block_weights_and_remap(model_parser, config, layer_idxs_to_load, device="cpu")
        if quantize_to_int8:
            quantize_all_mlps(current_transformer_blocks_loaded)
        delta_t = time.time() - start_t
        print(f"Finished to load layers: {layer_idxs_to_load} in {delta_t} seconds.")
        with open(os.path.join(converted_model_path, f"blocks_chunk_{current_chunk}.pkl"), "wb") as f:
            pickle.dump(current_transformer_blocks_loaded, f)
    layer_idxs_to_load.pop(0) # remove index as "consumed"

embedding_weights = model_parser.get_tensor('model.embed_tokens.weight')
output_norm_weights = model_parser.get_tensor('model.norm.weight')
if quantize_to_int8:
    output_embedding_weights = model_parser.get_tensor('lm_head.weight')
    

general_chunk_dict = {
    'model.embed_tokens.weight': embedding_weights,
    'model.norm.weight': output_norm_weights,
}
if quantize_to_int8:
    output_embedding_weights, output_embedding_scales = quantize_fp32_linear_to_int8(output_embedding_weights)
    general_chunk_dict['lm_head.weight'] = output_embedding_weights
    general_chunk_dict['lm_head.weight_scales'] = output_embedding_scales
else:
    general_chunk_dict['lm_head.weight'] = output_embedding_weights

with open(os.path.join(converted_model_path, f"general_chunk.pkl"), "wb") as f:
    pickle.dump(general_chunk_dict, f)