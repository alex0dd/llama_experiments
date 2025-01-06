import logging
from typing import Dict, Optional

import torch

import os
import pickle

from quantization.utils_int4 import embedding_int4, linear_int4
from quantization.utils_int8 import embedding_int8, linear_int8

from ops.rope import LLAMA3_PositionalEmbeddings, Phi3_PositionalEmbeddings
from ops.kv_cache_ops import build_kv_caches
from ops.attention_ops import apple_attn_wrapper, base_attn, base_attn_unopt

from ops.utils import (
    build_attention_mask,
    build_attention_mask_gemma2,
    load_general_chunk,
    repeat_kv,
    get_head_dim
)

logger = logging.getLogger(__name__)


def embedding_matrix(inputs, weights, scales=None, original_shape=None):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    return torch.nn.functional.embedding(inputs, weights)


def linear_default(inputs, weight, scales=None, bias=None, original_shape=None):
    out = torch.nn.functional.linear(inputs, weight.to(dtype=inputs.dtype), bias=bias)
    return out


linears = {"default": linear_default, "int4": linear_int4, "int8": linear_int8}
embeddings = {
    "default": embedding_matrix,
    "int4": embedding_int4,
    "int8": embedding_int8,
}

class WeightlessFFN(torch.nn.Module):
    def __init__(self, linear_fn, activation_fn=torch.nn.functional.silu):
        super(WeightlessFFN, self).__init__()
        self.linear_fn = linear_fn
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]):
        output = self.linear_fn(
            x,
            weights["mlp.gate_proj.weight"],
            weights.get("mlp.gate_proj.weight_scales"),
            original_shape=weights.get("mlp.gate_proj.weight_orig_shape"),
            bias=weights.get("mlp.gate_proj.bias"),
        )
        output = self.activation_fn(output) * self.linear_fn(
            x,
            weights["mlp.up_proj.weight"],
            weights.get("mlp.up_proj.weight_scales"),
            original_shape=weights.get("mlp.up_proj.weight_orig_shape"),
            bias=weights.get("mlp.up_proj.bias"),
        )
        output = self.linear_fn(
            output,
            weights["mlp.down_proj.weight"],
            weights.get("mlp.down_proj.weight_scales"),
            original_shape=weights.get("mlp.down_proj.weight_orig_shape"),
            bias=weights.get("mlp.down_proj.bias"),
        )
        return output


class WeightlessPhi3FFN(WeightlessFFN):
    def __init__(self, linear_fn):
        super(WeightlessPhi3FFN, self).__init__(linear_fn=linear_fn)

    def forward(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]):
        up_states = self.linear_fn(
            x,
            weights["mlp.gate_up_proj.weight"],
            weights.get("mlp.gate_up_proj.weight_scales"),
            original_shape=weights.get("mlp.gate_up_proj.weight_orig_shape"),
        )
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = torch.nn.functional.silu(gate) * up_states
        output = self.linear_fn(
            up_states,
            weights["mlp.down_proj.weight"],
            weights.get("mlp.down_proj.weight_scales"),
            original_shape=weights.get("mlp.down_proj.weight_orig_shape"),
        )
        return output

@torch.jit.script
def _normalization(inputs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    rms_a = inputs.pow(2).mean(-1, keepdim=True)
    rms_a = inputs * torch.rsqrt(rms_a + eps)
    return rms_a

class WeightlessRMSNorm(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, add_unit_offset: bool = False):
        super(WeightlessRMSNorm, self).__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset

    @torch.inference_mode()
    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        output = _normalization(inputs.float(), eps=self.eps)
        if self.add_unit_offset:
            output = output * (1 + weights.float())
            return output.type_as(inputs)
        else:
            return output * weights


class WeightlessGQA(torch.nn.Module):
    def __init__(
        self,
        linear_fn,
        model_type: str,
        n_rep: int,
        n_kv_heads: int,
        n_heads: int,
        head_dim: int,
        attn_type: str = "vanilla",
        attn_logit_softcapping: float = None
    ):
        super(WeightlessGQA, self).__init__()
        assert attn_type in ["vanilla", "apple", "sliding"]
        self.linear_fn = linear_fn
        self.model_type = model_type
        self.n_rep = n_rep
        self.n_kv_heads = n_kv_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_type = attn_type
        self.attn_logit_softcapping = attn_logit_softcapping
        self.normalization_factor = float(self.head_dim) ** -0.5

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        input_pos: int,
        weights: Dict[str, torch.Tensor],
        kv_cache: torch.nn.Module,
        freqs_rope: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bs, seq_len, _ = x.shape

        x = x.bfloat16()

        xq = self.linear_fn(
            x,
            weights["self_attn.q_proj.weight"],
            weights.get("self_attn.q_proj.weight_scales"),
            original_shape=weights.get("self_attn.q_proj.weight_orig_shape"),
            bias=weights.get("self_attn.q_proj.bias"),
        )
        xk = self.linear_fn(
            x,
            weights["self_attn.k_proj.weight"],
            weights.get("self_attn.k_proj.weight_scales"),
            original_shape=weights.get("self_attn.k_proj.weight_orig_shape"),
            bias=weights.get("self_attn.k_proj.bias"),
        )
        xv = self.linear_fn(
            x,
            weights["self_attn.v_proj.weight"],
            weights.get("self_attn.v_proj.weight_scales"),
            original_shape=weights.get("self_attn.v_proj.weight_orig_shape"),
            bias=weights.get("self_attn.v_proj.bias"),
        )

        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)

        if self.model_type in ["phi3", "granite-small", "gemma2", "qwen2"]:
            xq, xk = Phi3_PositionalEmbeddings.apply_rotary_emb(
                xq, xk, cos=freqs_rope[0], sin=freqs_rope[1]
            )
        else:
            xq, xk = LLAMA3_PositionalEmbeddings.apply_rotary_emb(xq, xk, freqs_rope)

        if kv_cache is not None:
            keys, values = kv_cache.update(input_pos, xk, xv)
        
        if self.n_kv_heads < self.n_heads:
            keys = repeat_kv(keys, self.n_rep)
            values = repeat_kv(values, self.n_rep)

        xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, keys, values))
        if xq.device.type == "mps":
            if mask is not None:
                mask = mask[:, : keys.shape[-2]]
                match self.attn_type:
                    case "apple":
                        output = apple_attn_wrapper(xq, keys, values, mask, self.normalization_factor)
                    case "sliding":
                        sliding_window_size = 4096
                        min_dtype = torch.finfo(xq.dtype).min
                        sliding_window_mask = torch.tril(
                            torch.ones_like(mask, dtype=torch.bool), diagonal=-sliding_window_size
                        )
                        mask = torch.where(sliding_window_mask, min_dtype, mask)
                        if mask.shape[-1] <= 1:  # when decoding
                            mask = mask[:, -sliding_window_size :]
                        output = base_attn(xq, keys, values, mask, self.normalization_factor, attn_logit_softcapping=self.attn_logit_softcapping)
                    case _:
                        output = base_attn(xq, keys, values, mask, self.normalization_factor)
            else:
                # Trick: since torch.jit won't trace if mask is none, 
                # we'll call the unoptimized version just for first iteration
                output = base_attn_unopt(xq, keys, values, None, self.normalization_factor, attn_logit_softcapping=self.attn_logit_softcapping)
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, keys, values, attn_mask=mask, dropout_p=0.0
            )
        
        if self.attn_type != "apple":
            output = output.transpose(1, 2).contiguous().view(bs, output.shape[2], -1)
        else:
            # TODO: later fix this specific apple attn case with transposing
            output = output.contiguous().view(bs, output.shape[2], -1)
        output = self.linear_fn(
            output,
            weights["self_attn.o_proj.weight"],
            weights.get("self_attn.o_proj.weight_scales"),
            original_shape=weights.get("self_attn.o_proj.weight_orig_shape"),
            bias=weights.get("self_attn.o_proj.bias"),
        )
        return output


class WeightlessTransformerBlock(torch.nn.Module):
    def __init__(self, linear_fn, config, attn_type = "vanilla"):
        super(WeightlessTransformerBlock, self).__init__()
        self.linear_fn = linear_fn
        self.config = config
        self.n_rep = config["num_attention_heads"] // config["num_key_value_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.n_heads = config["num_attention_heads"]
        self.head_dim = get_head_dim(config)
        self.model_type = config["model_type"]
        self.attn_type = attn_type
        self.attention_module = WeightlessGQA(
            linear_fn, 
            model_type=self.model_type, 
            n_rep=self.n_rep, 
            n_kv_heads=self.n_kv_heads, 
            n_heads=self.n_heads, 
            head_dim=self.head_dim,
            attn_type=self.attn_type,
            attn_logit_softcapping=config.get("attn_logit_softcapping", None)
        )
        self.hidden_activation = self._get_act_fn(config=self.config)
        self.ffn_module = WeightlessFFN(linear_fn, activation_fn=self.hidden_activation) if self.model_type != "phi3" else WeightlessPhi3FFN(linear_fn)
        self.layernorm = WeightlessRMSNorm(add_unit_offset="gemma2" in self.model_type)

    def _get_act_fn(self, config):
        from functools import partial
        """
        Parses config entry for the activation function, and matches it with torch's activation function instance.
        """
        act_dict = {
            "silu": torch.nn.functional.silu,
            "gelu_pytorch_tanh": partial(torch.nn.functional.gelu, approximate="tanh")
        }
        if "hidden_act" in config:
            hidden_activation_str = config["hidden_act"]
        elif "hidden_activation" in config:
            hidden_activation_str = config["hidden_activation"]
        else:
            hidden_activation_str = "silu"
        return act_dict[hidden_activation_str]

    def forward(
        self,
        x: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        cache_kv: torch.nn.Module,
        input_pos: int,
        freqs_rope: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        if "gemma2" in self.config["model_type"]:
            # Self-attention
            residual = x
            hidden = self.layernorm(x, weights["input_layernorm.weight"])
            hidden = self.attention_module(
                hidden,
                input_pos,
                weights,
                cache_kv,
                freqs_rope,
                mask,
            )
            hidden = self.layernorm(
                hidden, weights["post_attention_layernorm.weight"]
            )
            hidden = residual + hidden
            # NOTE: THIS WAS THE ERROR I WAS GETTING ALL THIS TIME, I DIDN'T PUT THIS LINE!
            residual = hidden
            # MLP
            hidden = self.layernorm(
                hidden, weights["pre_feedforward_layernorm.weight"]
            )
            hidden = self.ffn_module(hidden, weights)
            hidden = self.layernorm(
                hidden, weights["post_feedforward_layernorm.weight"]
            )
            output = residual + hidden
            return output
        else:
            attended_x = self.layernorm(x, weights["input_layernorm.weight"])
            hidden = x + self.attention_module(
                attended_x,
                input_pos,
                weights,
                cache_kv,
                freqs_rope,
                mask,
            )
            attended_hidden = self.layernorm(
                hidden, weights["post_attention_layernorm.weight"]
            )
            output = hidden + self.ffn_module(attended_hidden, weights)
            return output


def move_to_device_recursive(data, device):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                move_to_device_recursive(value, device)
            elif not key.endswith("_orig_shape"):
                data[key] = value.to(device)
    else:
        raise TypeError("Input must be a dictionary")


class Transformer:
    def __init__(self, model_dir, config, device="cpu", cache_max_seq_len=4096, cache_max_bs=1, output_hidden_states=False):
        self.model_dir = model_dir
        self.config = config
        self.device = device
        self.head_dim = get_head_dim(config)
        self.model_type = self.config["model_type"]
        if self.model_type in ["llama", "mistral"]:
            self.freqs_rope = LLAMA3_PositionalEmbeddings.precompute_rope_constants(
                self.head_dim,
                self.config["max_position_embeddings"] * 2,
                self.config["rope_theta"],
            ).to(self.device)
        elif self.model_type in ["phi3", "granite-small", "gemma2", "qwen2"]:
            position_ids = torch.arange(
                0, self.config["max_position_embeddings"], dtype=torch.long
            )
            position_ids = position_ids.unsqueeze(0).view(
                -1, self.config["max_position_embeddings"]
            )
            self.freqs_rope = Phi3_PositionalEmbeddings.precompute_rope_constants(
                position_ids=position_ids,
                dim=self.head_dim,
                base=self.config["rope_theta"],
            ).to(self.device, dtype=torch.bfloat16)  # TODO: unhardcode this precision
        self.max_seq_len = self.config["max_position_embeddings"]
        self.num_layers = self.config["num_hidden_layers"]
        self.caches_memory = build_kv_caches(
            self.config, 
            device=self.device, 
            max_seq_len=cache_max_seq_len if cache_max_seq_len > 0 else self.max_seq_len,
            max_bs=cache_max_bs
        )
        logger.info(f"Cache shape: {self.caches_memory[0].k_cache.shape}")

        #self.chunk_weights = load_block_chunk(
        #    model_dir, 0
        #)  # assume all weights are in single chunk
        
        self.general_chunk_weights = load_general_chunk(self.model_dir)
        #move_to_device_recursive(self.chunk_weights, self.device)
        move_to_device_recursive(self.general_chunk_weights, self.device)

        self.conversion_config = self.config.get("conversion_config", {})

        # Get all the chunking info from conversion config, falling back
        # to single chunk if needed.
        chunking_info = self.conversion_config.get("chunking_info", {})
        self.num_chunks = chunking_info.get("num_chunks", 1)
        self.chunk_size = chunking_info.get(
            "chunk_size",
            config["num_hidden_layers"]
        )

        self.precision = self.conversion_config.get("precision", "default")
        self.precision_embeddings = self.conversion_config.get("precision_embeddings", "default")
        self.linear_fn = linears[self.precision]
        self.embedding_fn = embeddings[self.precision_embeddings]
        if "gemma2" in self.model_type:
            self.transformer_block_fns = [
                WeightlessTransformerBlock(
                    self.linear_fn, self.config, attn_type="sliding" if idx % 2 == 0 else "vanilla"
                ) for idx in range(self.num_layers)
            ]
        else:
            self.transformer_block_fns = [
                WeightlessTransformerBlock(self.linear_fn, self.config) for _ in range(self.num_layers)
            ]
        self.layernorm = WeightlessRMSNorm(add_unit_offset="gemma2" in self.model_type)

        self.cur_max_seq_len = -1

        self.output_hidden_states = output_hidden_states

        self.final_logit_softcapping=config.get("final_logit_softcapping", None)

        self.current_chunk_idx = None
        self.current_chunk_weights = None

    def _get_chunk_idx_for_layer(self, layer_idx: int) -> int:
        """
        Determine which chunk file contains a given layer index.
        """
        return layer_idx // self.chunk_size
    
    def load_swap_chunk(self, needed_chunk_idx: int):
        """
        Loads the specified chunk from disk if it's not already loaded.
        Unloads the previously loaded chunk to save memory.
        """
        if needed_chunk_idx == self.current_chunk_idx:
            # Already loaded
            return

        logger.info(f"Swapping chunk {self.current_chunk_idx} with {needed_chunk_idx}.")
        # Unload the old chunk
        self.current_chunk_weights = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()  # optional, to free GPU memory

        # Load new chunk
        chunk_file = os.path.join(self.model_dir, f"blocks_chunk_{needed_chunk_idx}.pkl")
        with open(chunk_file, "rb") as f:
            chunk_data = pickle.load(f)

        # Move weights to device if needed
        move_to_device_recursive(chunk_data, self.device)

        self.current_chunk_idx = needed_chunk_idx
        self.current_chunk_weights = chunk_data
    
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, input_pos: int, max_seq_len: int = None, min_seq_len: int = None):
        # TODO: remove max_seq_len or change it, as it's needed for gemma2 correct masking
        if self.output_hidden_states: self.hidden_states = []
        # TODO: make input_pos a tensor of shape [B, 1], containing positions for each batch element
        
        # emb O(0.0001)s
        # all blocks O(0.25)s
        # head 0(0.0001)s
        seq_len = tokens.shape[-1]
        seq_embeddings = self.embedding_fn(
            tokens,
            self.general_chunk_weights["model.embed_tokens.weight"],
            scales=self.general_chunk_weights.get("model.embed_tokens.weight_scales"),
        )
        if self.model_type in ["phi3", "granite-small", "gemma2", "qwen2"]:
            freqs_rope = self.freqs_rope[:, input_pos : input_pos + seq_len]
        else:
            freqs_rope = self.freqs_rope[input_pos : input_pos + seq_len]
        if not self.model_type in ["gemma2"]:
            mask = build_attention_mask(
                seq_len, input_pos, device=tokens.device, dtype=seq_embeddings.dtype,
            )
        else:
            mask = build_attention_mask_gemma2(
                max_seq_len, min_seq_len, input_pos, device=tokens.device, dtype=seq_embeddings.dtype
            )
        h = seq_embeddings
        if self.model_type in ["gemma2"]:
            normalizer = torch.tensor(self.config["hidden_size"]**0.5, dtype=torch.bfloat16)
            h = h * normalizer
        if self.output_hidden_states: self.hidden_states.append(h)
        for layer_idx in range(self.num_layers):
            cache_kv = self.caches_memory[layer_idx]

            needed_chunk_idx = self._get_chunk_idx_for_layer(layer_idx)
            self.load_swap_chunk(needed_chunk_idx)
            
            block_weights = self.current_chunk_weights[layer_idx]

            h = self.transformer_block_fns[layer_idx](
                h,
                block_weights,
                cache_kv,
                input_pos,
                freqs_rope,
                mask,
            )
            if self.output_hidden_states: self.hidden_states.append(h)
        h = self.layernorm(h, self.general_chunk_weights["model.norm.weight"])
        if self.output_hidden_states: self.hidden_states.append(h)
        logits = self.linear_fn(
            h,
            self.general_chunk_weights.get("lm_head.weight"),
            scales=self.general_chunk_weights.get("lm_head.weight_scales"),
            original_shape=self.general_chunk_weights.get("lm_head.weight_orig_shape"),
        ).float()

        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        if self.output_hidden_states:
            return logits, self.hidden_states
        else:
            return logits
