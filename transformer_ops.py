import numpy as np

"""
'model.layers.{i}.input_layernorm.weight',
'model.layers.{i}.mlp.down_proj.weight',
'model.layers.{i}.mlp.gate_proj.weight',
'model.layers.{i}.mlp.up_proj.weight',
'model.layers.{i}.post_attention_layernorm.weight',
'model.layers.{i}.self_attn.k_proj.weight',
'model.layers.{i}.self_attn.o_proj.weight',
'model.layers.{i}.self_attn.q_proj.weight',
'model.layers.{i}.self_attn.v_proj.weight'
"""
"""
>>> tensor_parser.shape("model.layers.1.mlp.down_proj.weight")
[4096, 14336]
>>> tensor_parser.shape("model.layers.1.mlp.gate_proj.weight")
[14336, 4096]
>>> tensor_parser.shape("model.layers.1.mlp.up_proj.weight")
[14336, 4096]
"""

class Transformer:
    def __init__(self):
        pass

class MLP:
    def __init__(self):
        pass

class LayerNorm:
    def __init__(self):
        pass

class Attention:
    def __init__(self):
        pass