from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np

from .safetensors_parser import SafeTensorsParser

class ModelParser:

    def __init__(self, file_names: List[str]) -> None:
        self._file_names = file_names
        self._parsers_dict = self._load_parsers(self._file_names)
        self._tensors_parsers_map = self._build_tensors_parsers_map(self._parsers_dict)

    @property
    def tensor_names(self) -> List[str]:
        return list(chain(*[parser.tensor_names for parser in self._parsers_dict.values()]))
    
    @property
    def tensor_shapes(self) -> List[Tuple[str, List[int]]]:
        return list(chain(*[parser.tensor_shapes for parser in self._parsers_dict.values()]))
    
    def get_tensor(self, tensor_name: str) -> np.array:
        assert tensor_name in self.tensor_names, f"Tensor {tensor_name} is not present in loaded tensors"
        parser = self._tensors_parsers_map[tensor_name]
        return parser.get_tensor(tensor_name)

    def get_tensors(self, tensor_names: List[str]):
        for tensor_name in tensor_names: tensor_name in self.tensor_names, f"Tensor {tensor_name} is not present in loaded tensors"
        output_dict = {}
        parser_tensor_read_map = defaultdict(list)
        # Get all the parsers and their associated tensor names
        for tensor_name in tensor_names:
            parser = self._tensors_parsers_map[tensor_name]
            parser_tensor_read_map[parser].append(tensor_name)
        # For each parser, read all the necessary tensors from it at once.
        for parser in parser_tensor_read_map.keys():
            tensors_dict = parser.get_tensors_without_closing_file(parser_tensor_read_map[parser])
            output_dict.update(tensors_dict)
        return output_dict

    def _build_tensors_parsers_map(self, parsers_dict: Dict[str, SafeTensorsParser]) -> Dict[str, SafeTensorsParser]:
        """
        Associates a parser reference to each tensor name for later quick mapping of parsers.
        """
        tensors_parsers_map = {}
        for parser in parsers_dict.values():
            for tensor_name in parser.tensor_names:
                tensors_parsers_map[tensor_name] = parser
        return tensors_parsers_map

    def _load_parsers(self, file_names: List[str]) -> Dict[str, SafeTensorsParser]:
        parsers_dict = {}
        for fname in file_names:
            parsers_dict[fname] = SafeTensorsParser(fname)
        return parsers_dict