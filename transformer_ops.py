from collections import defaultdict
import torch
import math
from typing import Dict, Optional, Tuple

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

def hf_undo_permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def remap_weights_if_needed(weights, param_name, config):
    """
    Important: HF checkpoint permutes the original weights for Q, K weight tensors
    https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/llama/convert_llama_weights_to_hf.py#L174-L182
    This function will take a block of attention weights and permute only the ones that need to be permuted
    """
    if "self_attn.q_proj.weight" in param_name:
        weights = hf_undo_permute(
            weights,
            n_heads=config["num_attention_heads"],
            dim1=config["hidden_size"],
            dim2=config["hidden_size"],
        )
    elif "self_attn.k_proj.weight" in param_name:
        # NOTE: for llama 3 70B this will need to be fixed further, as num_shards will be different
        num_shards = 1
        num_key_value_heads = config["num_key_value_heads"]
        n_heads = config["num_attention_heads"]
        n_heads_per_shard = n_heads // num_shards
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        
        weights = hf_undo_permute(
            weights,
            n_heads=config["num_key_value_heads"],
            dim1=config["hidden_size"] // num_local_key_value_heads,
            dim2=config["hidden_size"],
        )
    return weights
        
    

def embedding_matrix(inputs, weights):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    This function is taken and adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """ 
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function is taken from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    It should be equivalent to torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_rope: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE:
        - https://blog.eleuther.ai/rotary-embeddings/
        - https://github.com/meta-llama/llama3/blob/main/llama/model.py
    This function is taken and adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    It applies rotatory positional embedding to query and key vectors, performing only one reshape of rope frequencies.
    """ 
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_rope = reshape_for_broadcast(freqs_rope, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_rope).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_rope).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_rope_constants(dim: int, end: int, theta: float = 10000.0):
    """
    RoPE:
        - https://blog.eleuther.ai/rotary-embeddings/
        - https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """ 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_rope = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_rope

def get_all_layer_names_in_block(layer_idx):
    weight_remap = {
        f'model.layers.{layer_idx}.mlp.down_proj.weight': 'mlp.down_proj.weight',
        f'model.layers.{layer_idx}.mlp.gate_proj.weight': 'mlp.gate_proj.weight',
        f'model.layers.{layer_idx}.mlp.up_proj.weight': 'mlp.up_proj.weight',
        f'model.layers.{layer_idx}.input_layernorm.weight': 'input_layernorm.weight',
        f'model.layers.{layer_idx}.post_attention_layernorm.weight': 'post_attention_layernorm.weight'
    }
    for letter in ['q', 'k', 'v', 'o']:
        orig_key = f'model.layers.{layer_idx}.self_attn.{letter}_proj.weight'
        new_key = f'self_attn.{letter}_proj.weight'
        weight_remap[orig_key] = new_key
    return weight_remap

def load_multiple_transformer_block_weights_and_remap(parser, config, layer_idxs, device):
    """
    Helper function that loads all the weights for given transformer block indices minimizing
    the number of file reads, and transfers them to the given decide.
    """
    num_layers = config["num_hidden_layers"]
    loaded_weights = defaultdict(dict)
    all_weights_remap = {}
    # Get all the layer names
    for idx in layer_idxs:
        if idx < num_layers:
            weight_remap = get_all_layer_names_in_block(idx)
            all_weights_remap.update(weight_remap)
            #loaded_weights[idx] = self._load_weights_for_block(self.config, idx, device=device)
    # Load all weights opening the F files exactly F times.
    all_loaded_weights = parser.get_tensors(list(all_weights_remap.keys()))
    # Assign the loaded weights to the appropriate loaded_weights dict fields.
    for weight_name in all_loaded_weights.keys():
        layer_idx = int(weight_name.split(".")[2])
        remapped_weight_name = all_weights_remap[weight_name]
        # Move the original loaded weight with original weight name, into the loaded_weights dict (and repermute it if needed, e.g. in attn q, k cases)
        final_weights = remap_weights_if_needed(all_loaded_weights[weight_name], weight_name, config)
        loaded_weights[layer_idx][remapped_weight_name] = final_weights.to(device)         
    return loaded_weights

# TODO: rewrite majority of layers using functional style, since we want to avoid holding internal state. This will make code more elegant
class Transformer:
    def __init__(self, config, parser, device="cpu"):
        self.device=device
        self.freqs_rope = precompute_rope_constants(
            config["hidden_size"] // config["num_attention_heads"],
            config["max_position_embeddings"] * 2,
            config["rope_theta"],
        ).to(self.device)
        self.preload_n_transformer_blocks = 6
        self.config = config
        self.num_layers = config["num_hidden_layers"]
        self.parser = parser
        self.caches_memory = self._build_kv_caches(config, device=self.device)
        self.transformer_block = TransformerBlock(config)

        self.embedding_weights = parser.get_tensor('model.embed_tokens.weight').to(self.device)
        if config["tie_word_embeddings"]:
            self.output_embedding_weights = self.embedding_weights
        else:
            self.output_embedding_weights = parser.get_tensor('lm_head.weight')
        self.output_embedding_weights = self.output_embedding_weights.to(self.device)
        self.output_norm = RMSNorm()
        self.output_norm_weights = parser.get_tensor('model.norm.weight').to(self.device)

    def _build_kv_caches(self, config, device, max_seq_len=2048, max_bs=4):
        """
        KV cache, memory occupation: MAX_BS*MAX_SEQ_LEN*N_KV_HEADS*HEAD_DIM*2 (2 because we have K and V)

        MAX_BS = 1
        MAX_SEQ_LEN = 2048 (can be up to 8192)
        N_KV_HEADS = 8
        HEAD_DIM = dim//n_heads=4096//32=128

        Total per layer = 1 * 2048 * 8 * 128 * 2 = 4194304 entries
        Bytes per layer = 4194304 * 4 (assuming float) = 16777216 bytes (16MB per layer)
        Bytes per model = 16777216 * NUM_LAYERS = 16777216 * 32 = 536870912 (512MB per model)
        """
        caches_memory = {}
        n_kv_heads = config["num_key_value_heads"]
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        for layer_idx in range(config["num_hidden_layers"]):
            caches_memory[layer_idx] = {}
            caches_memory[layer_idx]["k"] = torch.zeros((max_bs, max_seq_len, n_kv_heads, head_dim), dtype=config["torch_dtype"], device=device)
            caches_memory[layer_idx]["v"] = torch.zeros((max_bs, max_seq_len, n_kv_heads, head_dim), dtype=config["torch_dtype"], device=device)
        return caches_memory

    def _build_mask(self, seq_len, start_pos, device, dtype):
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=device), mask]
            ).to(dtype)
        return mask

    def _load_weights_for_block(self, config, layer_idx, device):
        # Loads all weights for an attention module, renaming and remapping weights if needed
        weights = {}
        for letter in ['q', 'k', 'v', 'o']:
            orig_key = f'model.layers.{layer_idx}.self_attn.{letter}_proj.weight'
            new_key = f'self_attn.{letter}_proj.weight'
            weights[new_key] = remap_weights_if_needed(self.parser.get_tensor(orig_key), param_name=orig_key, config=config)

        # Loads all the remaining (MLP+normalization) weights
        weights['mlp.down_proj.weight'] = self.parser.get_tensor(f'model.layers.{layer_idx}.mlp.down_proj.weight')
        weights['mlp.gate_proj.weight'] = self.parser.get_tensor(f'model.layers.{layer_idx}.mlp.gate_proj.weight')
        weights['mlp.up_proj.weight'] = self.parser.get_tensor(f'model.layers.{layer_idx}.mlp.up_proj.weight')
        weights['input_layernorm.weight'] = self.parser.get_tensor(f'model.layers.{layer_idx}.input_layernorm.weight')
        weights['post_attention_layernorm.weight'] = self.parser.get_tensor(f'model.layers.{layer_idx}.post_attention_layernorm.weight')
        for key in weights.keys():
            weights[key] = weights[key].to(device)
        return weights

    def _bulk_weights_loader(self, config, layer_idxs, device):
        loaded_weights = {}
        for idx in layer_idxs:
            if idx < self.num_layers:
                loaded_weights[idx] = self._load_weights_for_block(self.config, idx, device=device)
        return loaded_weights
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        seq_len = tokens.shape[-1]
        seq_embeddings = embedding_matrix(tokens, self.embedding_weights)
        freqs_rope = self.freqs_rope[start_pos : start_pos + seq_len]
        mask = self._build_mask(seq_len, start_pos, device=tokens.device, dtype=seq_embeddings.dtype)
        h = seq_embeddings

        current_transformer_blocks_loaded = None
        layer_idxs_to_load = []
        
        for layer_idx in range(self.num_layers):
            cache_k = self.caches_memory[layer_idx]["k"]
            cache_v = self.caches_memory[layer_idx]["v"]
            if len(layer_idxs_to_load) == 0:
                del current_transformer_blocks_loaded
                current_transformer_blocks_loaded = None
                layer_idxs_to_load = [layer_idx + i for i in range(self.preload_n_transformer_blocks)]
            if current_transformer_blocks_loaded is None:
                current_transformer_blocks_loaded = load_multiple_transformer_block_weights_and_remap(self.parser, self.config, layer_idxs_to_load, device=tokens.device)
            block_weights = current_transformer_blocks_loaded[layer_idx]
            layer_idxs_to_load.pop(0) # remove index as "consumed"
            
            #block_weights = self._load_weights_for_block(self.config, layer_idx, device=tokens.device)
            h = self.transformer_block.forward(h, block_weights, cache_k, cache_v, start_pos, freqs_rope, mask)
            # delete weights after inference on that block
            del block_weights
            
        h = self.output_norm.forward(h, self.output_norm_weights)
        output = torch.nn.functional.linear(h, self.output_embedding_weights).float()
        return output

class TransformerBlock:
    def __init__(self, config):
        self.attention=Attention(config)
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.ffn = FFN()

    def forward(self, x: torch.Tensor, weights, cache_k, cache_v, start_pos: int, freqs_rope: torch.Tensor, mask: Optional[torch.Tensor]):
        attended_x = self.attention_norm.forward(x, weights["input_layernorm.weight"])
        hidden = x + self.attention.forward(attended_x, start_pos, weights, cache_k, cache_v, freqs_rope, mask)
        attended_hidden = self.ffn_norm.forward(hidden, weights["post_attention_layernorm.weight"])
        output = hidden + self.ffn.forward(attended_hidden, weights)
        return output

class FFN():
    def __init__(
        self,
    ):
        pass

    def forward(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]):
        output = torch.nn.functional.linear(x, weights["mlp.gate_proj.weight"])
        output = torch.nn.functional.silu(output) * torch.nn.functional.linear(x, weights["mlp.up_proj.weight"])
        output = torch.nn.functional.linear(output, weights["mlp.down_proj.weight"])
        return output

class RMSNorm:
    """
    Root Mean Square Layer Normalization:
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, eps=1e-05):
        #super().__init__()
        self.eps = eps
        pass

    def _normalization(self, inputs: torch.Tensor) -> torch.Tensor:
        # Assume inputs of shape (B, S, n)
        rms_a = inputs.pow(2).mean(-1, keepdim=True)
        rms_a = inputs * torch.rsqrt(rms_a + self.eps)
        return rms_a

    def forward(self, inputs: torch.Tensor, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        In the original paper inputs = a, weights = g (gain)
        """
        # TODO: make this conversion generalizable as a function
        # Perform operation in fp32 precision
        output = self._normalization(inputs.float()).type_as(inputs)
        return output * weights

class Attention:
    def __init__(self, config, max_seq_len=2048):
        self.config = config
        self.n_rep = config["num_attention_heads"] // config["num_key_value_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.n_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]
        #self.max_seq_len = max_seq_len
        """
        KV cache, memory occupation: MAX_BS*MAX_SEQ_LEN*N_KV_HEADS*HEAD_DIM*2 (2 because we have K and V)

        MAX_BS = 1
        MAX_SEQ_LEN = 2048 (can be up to 8192)
        N_KV_HEADS = 8
        HEAD_DIM = dim//n_heads=4096//32=128

        Total per layer = 1 * 2048 * 8 * 128 * 2 = 4194304 entries
        Bytes per layer = 4194304 * 4 (assuming float) = 16777216 bytes (16MB per layer)
        Bytes per model = 16777216 * NUM_LAYERS = 16777216 * 32 = 536870912 (512MB per model)
        """
        MAX_BS = 4
        # TODO: move this outside of the class, so we can store it when this layer gets deleted
        #self.cache_k = torch.zeros((MAX_BS, self.max_seq_len, self.n_kv_heads, self.head_dim), dtype=self.config["torch_dtype"])
        #self.cache_v = torch.zeros((MAX_BS, self.max_seq_len, self.n_kv_heads, self.head_dim), dtype=self.config["torch_dtype"])

    """
    def get_cache(self):
        return (self.cache_k, self.cache_v)

    def set_cache(self, cache_kv: Tuple[torch.Tensor, torch.Tensor]):
        self.cache_k = cache_kv[0]
        self.cache_v = cache_kv[1]
    """

    # TODO: change order of start_pos and weights arguments
    def forward(self, x: torch.Tensor, start_pos: int, weights: Dict[str, torch.Tensor], cache_k, cache_v, freqs_rope: torch.Tensor, mask: Optional[torch.Tensor]):
        bs, seq_len, _ = x.shape
        
        # Apply attention transformation matrices
        # (BS, S, dim) -> (BS, S, n_heads * head_dim)
        xq = torch.nn.functional.linear(x, weights["self_attn.q_proj.weight"])
        # (BS, S, dim) -> (BS, S, n_kv_heads * head_dim)
        xk = torch.nn.functional.linear(x, weights["self_attn.k_proj.weight"])
        xv = torch.nn.functional.linear(x, weights["self_attn.v_proj.weight"])

        # Reshapes
        # (BS, S, n_heads * head_dim) -> (BS, S, n_heads, head_dim)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        # (BS, S, n_kv_heads * head_dim) -> (BS, S, n_kv_heads, head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_rope=freqs_rope)

        # Populate KV cache
        if cache_k is not None:
            cache_k[:bs, start_pos : start_pos + seq_len] = xk
            keys = cache_k[:bs, :start_pos + seq_len]
        else:
            # TODO: check if sequence length here is correct, given we start from pos=0
            keys = xk
        if cache_v is not None:
            cache_v[:bs, start_pos : start_pos + seq_len] = xv
            values = cache_v[:bs, :start_pos + seq_len]
        else:
            # TODO: check if sequence length here is correct, given we start from pos=0
            values = xv

        # Pad keys and values from n_kv_heads to n_heads, if needed (if n_kv_heads < n_heads)
        keys = repeat_kv(keys, self.n_rep)  # (BS, cache_len + S, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (BS, cache_len + S, n_heads, head_dim)

        # Reshapes
        xq = xq.transpose(1, 2)  # (BS, S, n_heads, head_dim) -> (bs, n_heads, S, head_dim)
        keys = keys.transpose(1, 2)  # (BS, cache_len + S, n_heads, head_dim) -> (BS, n_heads, cache_len + S, head_dim)
        values = values.transpose(1, 2)  # (BS, cache_len + S, n_heads, head_dim) -> (BS, n_heads, cache_len + S, head_dim)

        # Matmul -> (BS, n_heads, S, head_dim) @ (BS, n_heads, head_dim, cache_len + S) -> (BS, n_heads, S, cache_len + S)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        # Matmul -> (BS, n_heads, S, cache_len + S) @ (BS, n_heads, cache_len + S, head_dim) -> (BS, n_heads, S, head_dim)
        output = torch.matmul(scores, values)
        # # (BS, n_heads, S, head_dim) -> (BS, S, n_heads, head_dim) -> (BS, S, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        output = torch.nn.functional.linear(output, weights["self_attn.o_proj.weight"])
        return output