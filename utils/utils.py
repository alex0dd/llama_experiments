import math
import struct

import numpy as np
try:
    import torch
except:
    print("PyTorch is not available")

def bf16_to_fp32(bf16_val):
    # https://docs.python.org/3/library/struct.html#format-characters
    # https://stackoverflow.com/questions/77881835/how-to-get-the-exact-memory-representation-of-fp16-fp32-bf16-int8-and-so-on
    bytes_list = [bf16_val, b"0", b"0"]
    bytes_conc = b"".join(bytes_list)
    return struct.unpack("<f", bytes_conc)[0]

def parse_bf16_tensor(tensor, shape):
    parsed_vals = []
    # bf16 takes 2 bytes in memory
    for i in range(0, len(tensor), 2):
        current_entry = tensor[i:i+2]
        current_entry = bf16_to_fp32(current_entry)
        parsed_vals.append(current_entry)
    parsed_vals = np.array(parsed_vals).reshape(shape)
    return parsed_vals

def parse_bf16_tensor_v2(tensor, shape):
    parsed_vals = np.zeros(shape=shape, dtype=np.float32)
    # bf16 takes 2 bytes in memory
    for i in range(0, len(tensor), 2):
        current_entry = tensor[i:i+2]
        current_entry = bf16_to_fp32(current_entry)
        col, row = math.floor((i // 2) / shape[0]), (i // 2) % shape[0]
        try:
            parsed_vals[row - 1, col] = current_entry
        except:
            print(i, i//2, row, col)
            raise
    return parsed_vals

def parse_bf16_tensor_v3(tensor, shape):
    tensor = bytearray(tensor)
    parsed_vals = torch.frombuffer(tensor, dtype=torch.bfloat16).reshape(shape)
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