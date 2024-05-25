from safetensors.numpy import safe_open

import json
import os
import sys
from pprint import pprint

import numpy as np

# Parse manually safetensors
import struct

def bf16_to_fp32(bf16_val):
    # https://docs.python.org/3/library/struct.html#format-characters
    # https://stackoverflow.com/questions/77881835/how-to-get-the-exact-memory-representation-of-fp16-fp32-bf16-int8-and-so-on
    bytes_list = [bf16_val, b"0", b"0"]
    bytes_conc = b"".join(bytes_list)
    return struct.unpack("<f", bytes_conc)

def parse_bf16_tensor(tensor, shape):
    parsed_vals = []
    # bf16 takes 2 bytes in memory
    for i in range(0, len(tensor), 2):
        current_entry = tensor[i:i+2]
        current_entry = bf16_to_fp32(current_entry)
        parsed_vals.append(current_entry)
    parsed_vals = np.array(parsed_vals).reshape(shape)
    return parsed_vals

def bytes_to_uint(b):
    return int.from_bytes(b, byteorder='little', signed=False)

def build_offsetted_getter(data, base_offset):
    def get_tensor(tensor_metadata_entry):
        begin_offset = tensor_metadata_entry["data_offsets"][0] + base_offset
        end_offset = tensor_metadata_entry["data_offsets"][1] + base_offset
        return data[begin_offset: end_offset]
    return get_tensor

class TensorParser:

    def __init__(self, data_raw):
        self._tensor_metadata, self._metadata, self._tensor_memory = self._parse_structure(data_raw)

    def _parse_structure(self, data):
        """
        File structure
        8 bytes: N=u64 int containing header size
        N bytes: JSON utf-8 string representing header
        Rest of file: data
        """
        header_base_offset = 8
        header_size = bytes_to_uint(data[0:header_base_offset]) # 0-7
        data_base_offset = header_base_offset+header_size
        header = data[header_base_offset:data_base_offset] # 8 - 8 + N
        header = json.loads(header)
        tensor_metadata, metadata = self._parse_header(header)
        tensor_memory = data[data_base_offset:] # until end of file
        return tensor_metadata, metadata, tensor_memory

    def _parse_header(self, header):
        """
        Parses header bytes, returns tensor metadata and header's metadata.
        """
        tensor_metadata = {}
        metadata = None
        for key, value in header.items():
            if key == "__metadata__": metadata = value; continue
            tensor_metadata[key] = value
        return tensor_metadata, metadata

    def _get_tensor_data_raw(self, name):
        begin_offset = self._tensor_metadata[name]["data_offsets"][0]
        end_offset = self._tensor_metadata[name]["data_offsets"][1]
        return self._tensor_memory[begin_offset: end_offset]
    
    def get_tensor(self, name):
        shape = self._tensor_metadata[name]["shape"]
        current_tensor_raw = self._get_tensor_data_raw(name)
        current_tensor = parse_bf16_tensor(current_tensor_raw, shape)
        # TODO: add parsing cache, so that if a tensor was parsed, it can be offloaded to HDD or in RAM
        return current_tensor


# Opening the binary file in binary mode as rb(read binary)
f = open("./Meta-Llama-3-8B/model-00001-of-00004.safetensors", mode="rb")
# Reading file data with read() method
data = f.read()
f.close()

#def parse_file(data):
"""
File structure
8 bytes: N=u64 int containing header size
N bytes: JSON utf-8 string representing header
Rest of file: data
"""
header_base_offset = 8
header_size = bytes_to_uint(data[0:header_base_offset]) # 0-7
data_base_offset = header_base_offset+header_size
header = data[header_base_offset:data_base_offset] # 8 - 8 + N
header = json.loads(header)

tensor_metadata = {}

for key, value in header.items():
    if key == "__metadata__": continue
    tensor_metadata[key] = value
    print(key)

get_tensor = build_offsetted_getter(data, base_offset=data_base_offset)

tensor_name = "model.layers.8.self_attn.k_proj.weight"
current_tensor_metadata = tensor_metadata[tensor_name]
shape = current_tensor_metadata["shape"]
current_tensor_raw = get_tensor(current_tensor_metadata)
current_tensor = parse_bf16_tensor(current_tensor_raw, shape)

#struct.unpack("<e", tensor[:2])

raise

def safetensors_metadata_parser(file_path):
    header_size = 8
    meta_data = {}
    if os.stat(file_path).st_size > header_size:
        with open(file_path, "rb") as f:
            b8 = f.read(header_size)
            if len(b8) == header_size:
                header_len = int.from_bytes(b8, 'little', signed=False)
                headers = f.read(header_len)
                if len(headers) == header_len:
                    meta_data = sorted(json.loads(headers.decode("utf-8")).get("__metadata__", meta_data).items())
    return meta_data

file_path = "./Meta-Llama-3-8B/model-00001-of-00004.safetensors"

meta_data = safetensors_metadata_parser(file_path)
#result = {}
#with safe_open(file_path, framework="np") as f:
#    for k in f.keys():
#        result[k] = f.get_tensor(k)