from pprint import pprint
from parsers import ModelParser

model_parser = ModelParser([
    "./Meta-Llama-3-8B/model-00001-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00002-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00003-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00004-of-00004.safetensors",
])
pprint(model_parser.tensor_names)
tensor = model_parser.get_tensor("model.layers.8.self_attn.v_proj.weight")

pprint(model_parser.tensor_shapes)

emb_table_input = model_parser.get_tensor('model.embed_tokens.weight')
#emb_table_output = model_parser.get_tensor('lm_head.weight')