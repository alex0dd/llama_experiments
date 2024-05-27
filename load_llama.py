from pprint import pprint
from parsers import SafeTensorsParser

tensor_parser = SafeTensorsParser("./Meta-Llama-3-8B/model-00001-of-00004.safetensors")
print(tensor_parser.tensor_names)
tensor = tensor_parser.get_tensor("model.layers.8.post_attention_layernorm.weight")
tensor = tensor_parser.get_tensor("model.layers.8.self_attn.v_proj.weight")