import argparse
import json
import os
import pickle
import shutil
import time

import torch

from ops.utils import load_multiple_transformer_block_weights_and_remap
from parsers import ModelParser
from quantization.utils_int4 import quantize_fp32_linear_to_int4
from quantization.utils_int8 import quantize_fp32_linear_to_int8
from utils.utils import get_all_keyword_files, load_json


def is_linear_weight(layer_name):
    return "mlp." in layer_name and ".weight" in layer_name


def is_attention_linear(layer_name):
    return "self_attn." in layer_name and "proj.weight" in layer_name


def is_bias(param_name):
    return param_name.endswith(".bias")


def split_qkv(qkv_tensor, layer_name, quant_type):
    out_dict = {}
    # We need to split into q, k, v
    # Support only int8 for now
    # The layer will be named: self_attn.qkv_proj.weight
    qkv_tensor = torch.split(qkv_tensor, qkv_tensor.shape[0] // 3)
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

    q_layer_name = layer_name.replace("qkv", "q")
    k_layer_name = layer_name.replace("qkv", "k")
    v_layer_name = layer_name.replace("qkv", "v")

    out_dict[q_layer_name] = q_quantized
    out_dict[q_layer_name + "_scales"] = q_scale
    out_dict[q_layer_name + "_orig_shape"] = q_orig_shapes

    out_dict[k_layer_name] = k_quantized
    out_dict[k_layer_name + "_scales"] = k_scale
    out_dict[k_layer_name + "_orig_shape"] = k_orig_shapes

    out_dict[v_layer_name] = v_quantized
    out_dict[v_layer_name + "_scales"] = v_scale
    out_dict[v_layer_name + "_orig_shape"] = v_orig_shapes

    return out_dict


def quantize_all_mlps(blocks_chunk, quant_type="int8", device="cpu"):
    for layer_idx in blocks_chunk.keys():
        blocks_update_dict = {}
        remove_layers = []
        for layer_name in blocks_chunk[layer_idx].keys():
            param_instance = blocks_chunk[layer_idx][layer_name]
            if is_linear_weight(layer_name) or is_attention_linear(layer_name):
                if "qkv_proj" in layer_name:
                    qkv_update_dict = split_qkv(
                        blocks_chunk[layer_idx][layer_name], layer_name, quant_type
                    )
                    blocks_update_dict.update(qkv_update_dict)
                    remove_layers.append(layer_name)
                else:
                    if quant_type == "int8":
                        block_quantized, block_scale = quantize_fp32_linear_to_int8(
                            param_instance
                        )
                    elif quant_type == "int4":
                        block_quantized, block_scale = quantize_fp32_linear_to_int4(
                            param_instance, layer_name, device=device
                        )
                    # We'll always store original shapes for later code compatibility
                    orig_shapes = param_instance.shape
                    blocks_chunk[layer_idx][layer_name] = block_quantized
                    blocks_update_dict[layer_name + "_scales"] = block_scale
                    blocks_update_dict[layer_name + "_orig_shape"] = orig_shapes
        # Potentially remove keys that need to be removed (e.g., qkv if it was split into q/k/v)
        for k in remove_layers:
            blocks_update_dict.pop(k, None)
        blocks_chunk[layer_idx].update(blocks_update_dict)


def parse_all_args():
    # Create the argument parser
    arg_parser = argparse.ArgumentParser(description="Conversion script arguments.")
    arg_parser.add_argument(
        "--quantization_type",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Specify the quantization type: int4 or int8. Defaults to None if not specified.",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Specify the device: cpu or cuda. Defaults to 'cpu' if not specified.",
    )
    arg_parser.add_argument(
        "--base_model_dir",
        type=str,
        default="Meta-Llama-3-8B",
        help="Specify the path of base model directory containing LLAMA3. Defaults to 'Meta-Llama-3-8B' if not specified.",
    )
    arg_parser.add_argument(
        "--output_model_dir",
        type=str,
        default="LLAMA3-8B-PKL",
        help="Specify the path of the output model directory that will contain the converted checkpoint. Defaults to 'LLAMA3-8B-PKL'+quantization_type if not specified.",
    )
    arg_parser.add_argument(
        "--custom_model_type",
        type=str,
        choices=["granite-small", None],
        default=None,
        help="Define custom model type for conversion. Defaults to None if not specified. For Granite3B/8B models, this should be set to 'granite-small'.",
    )
    arg_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="If specified, this maximum sequence length will be used for the conversion.",
    )
    arg_parser.add_argument(
        "--quantize_embeddings",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If specified, input embeddings will be quantized.",
    )
    arg_parser.add_argument(
        "--disable-llama-qk-remap",
        action="store_true",
        help="If specified, will disable the remapping of QK weights when loading them (for Granite3/8B models this needs to be disabled).",
    )
    arg_parser.add_argument(
        "--force-tie-word-embeddings",
        action="store_true",
        help="If specified, will force the usage of tie word embeddings in case this field is missing in config file (e.g., in Gemma-2 case).",
    )

    # Parse the arguments
    args = arg_parser.parse_args()
    return args


args = parse_all_args()
print(args)
# Convert "None" string to actual None type if necessary
quantization_type = args.quantization_type
device = args.device
quantize_embeddings = args.quantize_embeddings
disable_llama_qk_remap = args.disable_llama_qk_remap
force_tie_word_embeddings = args.force_tie_word_embeddings
custom_model_type = args.custom_model_type
max_seq_len = int(args.max_seq_len) if args.max_seq_len else None

base_model_dir = args.base_model_dir
output_model_dir = args.output_model_dir

config_file_path = f"./{base_model_dir}/config.json"
model_files = get_all_keyword_files(base_model_dir, "safetensors", mode="endswith")
model_parser = ModelParser(model_files)
config = load_json(config_file_path)
if max_seq_len:
    assert max_seq_len <= config["max_position_embeddings"], "max-seq-len must be less than or equal to the original model's max_position_embeddings"
    config["max_position_embeddings"] = max_seq_len
tie_word_embeddings = config["tie_word_embeddings"] if "tie_word_embeddings" in config else force_tie_word_embeddings

output_model_dir = output_model_dir + (
    "" if not quantization_type else f"-{quantization_type}"
)

if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)

num_layers = int(config["num_hidden_layers"])
preload_n_transformer_blocks = (
    num_layers  # preload all (since we're dealing with not so big models)
)

current_transformer_blocks_loaded = None
layer_idxs_to_load = []
current_chunk = -1
for layer_idx in range(num_layers):
    if len(layer_idxs_to_load) == 0:
        current_chunk += 1
        current_transformer_blocks_loaded = None
        layer_idxs_to_load = [
            layer_idx + i for i in range(preload_n_transformer_blocks)
        ]
    if current_transformer_blocks_loaded is None:
        print(f"Beginning to load layers: {layer_idxs_to_load}")
        start_t = time.time()
        current_transformer_blocks_loaded = (
            load_multiple_transformer_block_weights_and_remap(
                model_parser,
                config,
                layer_idxs_to_load,
                device=device,
                disable_llama_qk_remap=disable_llama_qk_remap,
            )
        )
        if quantization_type:
            quantize_all_mlps(
                current_transformer_blocks_loaded,
                quant_type=quantization_type,
                device=device,
            )
        delta_t = time.time() - start_t
        print(f"Finished to load layers: {layer_idxs_to_load} in {delta_t} seconds.")
        with open(
            os.path.join(output_model_dir, f"blocks_chunk_{current_chunk}.pkl"), "wb"
        ) as f:
            pickle.dump(current_transformer_blocks_loaded, f)
    layer_idxs_to_load.pop(0)  # remove index as "consumed"

output_norm_weights = model_parser.get_tensor("model.norm.weight")
general_chunk_dict = {
    "model.norm.weight": output_norm_weights.to(device),
}

embedding_weights = model_parser.get_tensor("model.embed_tokens.weight")
if quantization_type and quantize_embeddings:
    # Embed also input embeddings (by default use int8 now, as int4 is still tricky)
    embedding_weights, embedding_scales = quantize_fp32_linear_to_int8(
        embedding_weights.T
    )  # pay attention to transpose here
    # Int4 quantize and pack convert
    # embedding_weights, embedding_scales = quantize_pack_embedding_table_v2(embedding_weights)
    general_chunk_dict["model.embed_tokens.weight"] = embedding_weights.T.to(device)
    general_chunk_dict["model.embed_tokens.weight_scales"] = embedding_scales.to(device)
else:
    general_chunk_dict["model.embed_tokens.weight"] = embedding_weights.to(device)

if not tie_word_embeddings:
    output_embedding_weights = model_parser.get_tensor("lm_head.weight")
else:
    # We'll still need to quantize the output embedding weights, even if tied.
    output_embedding_weights = embedding_weights

if quantization_type:
    orig_shapes = output_embedding_weights.shape
    if quantization_type == "int8":
        output_embedding_weights, output_embedding_scales = (
            quantize_fp32_linear_to_int8(output_embedding_weights)
        )
    elif quantization_type == "int4":
        output_embedding_weights, output_embedding_scales = (
            quantize_fp32_linear_to_int4(
                output_embedding_weights, "lm_head.weight", device=device
            )
        )
    general_chunk_dict["lm_head.weight"] = output_embedding_weights.to(device)
    general_chunk_dict["lm_head.weight_scales"] = output_embedding_scales.to(device)
    general_chunk_dict["lm_head.weight_orig_shape"] = orig_shapes
else:
    general_chunk_dict["lm_head.weight"] = output_embedding_weights.to(device)

with open(os.path.join(output_model_dir, "general_chunk.pkl"), "wb") as f:
    pickle.dump(general_chunk_dict, f)

# Copy config file
shutil.copy2(config_file_path, output_model_dir)
config_fpath = os.path.join(output_model_dir, "config.json")
# Modify the config file to include conversion data
with open(config_fpath, "r") as file:
    data = json.load(file)

data["conversion_config"] = {
    "precision": "default" if not quantization_type else quantization_type
}

if custom_model_type:
    data["model_type"] = custom_model_type

with open(config_fpath, "w") as file:
    json.dump(data, file, indent=4)
# Copy all tokenizer files
tokenizer_files = get_all_keyword_files(base_model_dir, "token", mode="contains")
for f in tokenizer_files:
    shutil.copy2(f, output_model_dir)
