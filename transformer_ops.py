import torch

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

def embedding_matrix(inputs, weights):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)

class Transformer:
    def __init__(self):
        pass

    def forward(self, tokens):
        seq_embeddings = embedding_matrix(tokens, embedding_weights)

class TransformerBlock:
    def __init__(self):
        pass

class MLP:
    def __init__(self):
        pass

class RMSNorm:
    """
    Root Mean Square Layer Normalization:
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, eps=1e-05):
        #super().__init__()
        self.eps = eps
        pass

    def _normalization(self, inputs):
        # Assume inputs of shape (B, S, n)
        rms_a = inputs.pow(2).mean(-1, keepdim=True)
        rms_a = inputs * torch.rsqrt(rms_a + self.eps)
        return rms_a

    def forward(self, inputs, weights):
        """
        In the original paper inputs = a, weights = g (gain)
        """
        # TODO: make this conversion generalizable as a function
        # Perform operation in fp32 precision
        output = self._normalization(inputs.float()).type_as(inputs)
        return output * weights

class Attention:
    def __init__(self):
        pass