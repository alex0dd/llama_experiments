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

def load_tensor_from_memory(file_name, begin_pos, n_of_bytes):
    """
    Returns n_of_bytes raw bytes from a binary file, starting from begin_pos position.
    """
    raw_data = None
    with open(file_name, "rb") as file:
        file.seek(begin_pos)
        raw_data = file.read(n_of_bytes)
    return raw_data

def deallocate_tensor(tensor):
    del tensor

class SafeTensorParser:

    #def __init__(self, data_raw):
        #self._tensor_metadata, self._metadata, self._tensor_memory = self._parse_structure(data_raw)
    def __init__(self, file_path):
        self._file_path = file_path
        self._tensor_metadata, self._metadata, self._data_base_offset = self._parse_structure(self._file_path)

    @property
    def tensor_names(self):
        return list(self._tensor_metadata.keys())
    
    def shape(self, name):
        return self._tensor_metadata[name]["shape"]

    def _parse_structure(self, file_path):
        """
        Inputs:
            - file_path: path of safetensors file
        
        File structure
        8 bytes: N=u64 int containing header size
        N bytes: JSON utf-8 string representing header
        Rest of file: data
        """
        with open(file_path, "rb") as file:
            header_size_field_length = 8
            header_size = file.read(header_size_field_length) # header size
            header_size = bytes_to_uint(header_size) # 0-7
            data_base_offset = header_size_field_length + header_size
            header = file.read(header_size) # 8 - 8 + N (read header_size bytes)
            header = json.loads(header)
            tensor_metadata, metadata = self._parse_header(header)

            return tensor_metadata, metadata, data_base_offset

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
        # Loads the tensor from secondary memory, only when requested
        start_tensor_offset = self._tensor_metadata[name]["data_offsets"][0]
        end_tensor_offset = self._tensor_metadata[name]["data_offsets"][1]
        begin_pos = start_tensor_offset + self._data_base_offset
        n_of_bytes = end_tensor_offset - start_tensor_offset
        # read the data
        data_raw = load_tensor_from_memory(self._file_path, begin_pos, n_of_bytes)
        return data_raw
    
    def get_tensor(self, name):
        shape = self._tensor_metadata[name]["shape"]
        current_tensor_raw = self._get_tensor_data_raw(name)
        current_tensor = parse_bf16_tensor(current_tensor_raw, shape)
        # TODO: add parsing cache, so that if a tensor was parsed, it can be offloaded to HDD or in RAM
        return current_tensor

tensor_parser = SafeTensorParser("./Meta-Llama-3-8B/model-00001-of-00004.safetensors")
print(tensor_parser.tensor_names)
tensor = tensor_parser.get_tensor("model.layers.8.post_attention_layernorm.weight")
tensor = tensor_parser.get_tensor("model.layers.8.self_attn.v_proj.weight")