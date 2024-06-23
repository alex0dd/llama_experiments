import argparse

from pprint import pprint
from parsers import ModelParser

import pickle

import torch
import time
import os

from ops.utils import load_multiple_transformer_block_weights_and_remap
from quantization.utils_int4 import quantize_fp32_linear_to_int4, quantize_pack_embedding_table_v2
from quantization.utils_int8 import quantize_fp32_linear_to_int8

from utils.utils import get_all_safetensors_model_files, load_json

def is_linear_weight(layer_name):
    return "mlp." in layer_name and ".weight" in layer_name

def is_attention_linear(layer_name):
    return "self_attn." in layer_name and "proj.weight" in layer_name

def quantize_all_mlps(blocks_chunk, quant_type="int8", device="cpu"):
    for layer_idx in blocks_chunk.keys():
        blocks_update_dict = {}
        for layer_name in blocks_chunk[layer_idx].keys():
            if is_linear_weight(layer_name) or is_attention_linear(layer_name):
                if "qkv_proj" in layer_name:
                    # We need to split into q, k, v
                    # Support only int8 for now
                    # The layer will be named: self_attn.qkv_proj.weight
                    qkv_tensor = blocks_chunk[layer_idx][layer_name]
                    qkv_tensor = torch.split(
                        qkv_tensor,
                        qkv_tensor.shape[0]//3
                    )
                    q_tensor = qkv_tensor[0]
                    k_tensor = qkv_tensor[1]
                    v_tensor = qkv_tensor[2]

                    if quant_type == "int8":
                        q_quantized, q_scale = quantize_fp32_linear_to_int8(q_tensor)
                        k_quantized, k_scale = quantize_fp32_linear_to_int8(k_tensor)
                        v_quantized, v_scale = quantize_fp32_linear_to_int8(v_tensor)

                    # For later code compatibility
                    q_orig_shapes = q_tensor.shape
                    k_orig_shapes = k_tensor.shape
                    v_orig_shapes = v_tensor.shape

                    blocks_chunk[layer_idx][layer_name] = None
                    q_layer_name = layer_name.replace("qkv", "q")
                    k_layer_name = layer_name.replace("qkv", "k")
                    v_layer_name = layer_name.replace("qkv", "v")

                    blocks_update_dict[q_layer_name] = q_quantized
                    blocks_update_dict[q_layer_name+"_scales"] = q_scale
                    blocks_update_dict[q_layer_name+"_orig_shape"] = q_orig_shapes

                    blocks_update_dict[k_layer_name] = k_quantized
                    blocks_update_dict[k_layer_name+"_scales"] = k_scale
                    blocks_update_dict[k_layer_name+"_orig_shape"] = k_orig_shapes

                    blocks_update_dict[v_layer_name] = v_quantized
                    blocks_update_dict[v_layer_name+"_scales"] = v_scale
                    blocks_update_dict[v_layer_name+"_orig_shape"] = v_orig_shapes
                else:
                    if quant_type == "int8":
                        block_quantized, block_scale = quantize_fp32_linear_to_int8(blocks_chunk[layer_idx][layer_name])
                    elif quant_type == "int4":
                        block_quantized, block_scale = quantize_fp32_linear_to_int4(blocks_chunk[layer_idx][layer_name], layer_name, device=device)                    
                    # We'll always store original shapes for later code compatibility
                    orig_shapes = blocks_chunk[layer_idx][layer_name].shape
                    blocks_update_dict[layer_name] = block_quantized
                    blocks_update_dict[layer_name+"_scales"] = block_scale
                    blocks_update_dict[layer_name+"_orig_shape"] = orig_shapes
            else:
                # Copy the norm layers just as they are
                blocks_update_dict[layer_name] = blocks_chunk[layer_idx][layer_name]
        # Remove all items
        blocks_chunk[layer_idx].clear()
        blocks_chunk[layer_idx].update(blocks_update_dict)

def parse_all_args():
    # Create the argument parser
    arg_parser = argparse.ArgumentParser(description="Conversion script arguments.")
    arg_parser.add_argument(
        "--quantization_type",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Specify the quantization type: int4 or int8. Defaults to None if not specified."
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Specify the device: cpu or cuda. Defaults to 'cpu' if not specified."
    )
    arg_parser.add_argument(
        "--base_model_dir",
        type=str,
        default="Phi-3-mini-4k-instruct",
        help="Specify the path of base model directory containing Phi3. Defaults to 'Phi-3-mini-4k-instruct' if not specified."
    )
    arg_parser.add_argument(
        "--output_model_dir",
        type=str,
        default="PHI3-MINI-4K-PKL",
        help="Specify the path of the output model directory that will contain the converted checkpoint. Defaults to 'PHI3-MINI-4K-PKL'+quantization_type if not specified."
    )
    
    # Parse the arguments
    args = arg_parser.parse_args()
    return args

args = parse_all_args()
print(args)
# Convert "None" string to actual None type if necessary
quantization_type = args.quantization_type
device = args.device

base_model_dir = args.base_model_dir
output_model_dir = args.output_model_dir

model_files = get_all_safetensors_model_files(base_model_dir)
model_parser = ModelParser(model_files)
config = load_json(f"./{base_model_dir}/config.json")

converted_model_path = output_model_dir+("" if not quantization_type else f"-{quantization_type}")

if not os.path.exists(converted_model_path):
    os.makedirs(converted_model_path)

num_layers = int(config["num_hidden_layers"])
preload_n_transformer_blocks = num_layers # preload all (since we're dealing with not so big models)

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
        current_transformer_blocks_loaded = load_multiple_transformer_block_weights_and_remap(model_parser, config, layer_idxs_to_load, device=device)
        if quantization_type:
            quantize_all_mlps(current_transformer_blocks_loaded, quant_type=quantization_type, device=device)
        delta_t = time.time() - start_t
        print(f"Finished to load layers: {layer_idxs_to_load} in {delta_t} seconds.")
        with open(os.path.join(converted_model_path, f"blocks_chunk_{current_chunk}.pkl"), "wb") as f:
            pickle.dump(current_transformer_blocks_loaded, f)
    layer_idxs_to_load.pop(0) # remove index as "consumed"

embedding_weights = model_parser.get_tensor('model.embed_tokens.weight')
output_norm_weights = model_parser.get_tensor('model.norm.weight')
if quantization_type:
    output_embedding_weights = model_parser.get_tensor('lm_head.weight')

general_chunk_dict = {
    'model.norm.weight': output_norm_weights.to(device),
}
if quantization_type:
    orig_shapes = output_embedding_weights.shape
    if quantization_type == "int8":
        output_embedding_weights, output_embedding_scales = quantize_fp32_linear_to_int8(output_embedding_weights)
    elif quantization_type == "int4":
        output_embedding_weights, output_embedding_scales = quantize_fp32_linear_to_int4(output_embedding_weights, 'lm_head.weight', device=device)
    general_chunk_dict['lm_head.weight'] = output_embedding_weights.to(device)
    general_chunk_dict['lm_head.weight_scales'] = output_embedding_scales.to(device)
    general_chunk_dict['lm_head.weight_orig_shape'] = orig_shapes

    # Embed also input embeddings (by default use int8 now, as int4 is still tricky)
    embedding_weights, embedding_scales = quantize_fp32_linear_to_int8(embedding_weights.T) # pay attention to transpose here
    # Int4 quantize and pack convert
    #embedding_weights, embedding_scales = quantize_pack_embedding_table_v2(embedding_weights)
    general_chunk_dict['model.embed_tokens.weight'] = embedding_weights.T.to(device)
    general_chunk_dict['model.embed_tokens.weight_scales'] = embedding_scales.to(device)
else:
    general_chunk_dict['lm_head.weight'] = output_embedding_weights.to(device)
    general_chunk_dict['model.embed_tokens.weight'] = embedding_weights.to(device)

with open(os.path.join(converted_model_path, f"general_chunk.pkl"), "wb") as f:
    pickle.dump(general_chunk_dict, f)