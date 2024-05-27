import json

from utils import bytes_to_uint, load_tensor_from_memory
from utils import parse_bf16_tensor_v3 as parse_bf16_tensor

class SafeTensorsParser:

    def __init__(self, file_path):
        self._file_path = file_path
        self._tensor_metadata, self._metadata, self._data_base_offset = self._parse_structure(self._file_path)

    @property
    def tensor_names(self):
        return list(self._tensor_metadata.keys())
    
    @property
    def tensor_shapes(self):
        return [(name, self._tensor_metadata[name]["shape"]) for name in self._tensor_metadata.keys()]
    
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
        assert name in self.tensor_names, f"Tensor {name} is not in list of tensor names loaded from {self._file_path}"
        shape = self._tensor_metadata[name]["shape"]
        current_tensor_raw = self._get_tensor_data_raw(name)
        current_tensor = parse_bf16_tensor(current_tensor_raw, shape)
        # TODO: add parsing cache, so that if a tensor was parsed, it can be offloaded to HDD or in RAM
        return current_tensor