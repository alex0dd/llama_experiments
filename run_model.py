#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pprint import pprint
from parsers import ModelParser

import torch
from transformers import AutoTokenizer

model_parser = ModelParser([
    "./Meta-Llama-3-8B/model-00001-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00002-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00003-of-00004.safetensors",
    "./Meta-Llama-3-8B/model-00004-of-00004.safetensors",
])


# In[2]:


model_parser._parsers_dict['./Meta-Llama-3-8B/model-00002-of-00004.safetensors'].tensor_names


# In[3]:


pprint(model_parser.tensor_names)


# ## Prepare text and embeddings

# In[4]:


device="cpu"


# In[5]:


tokenizer = AutoTokenizer.from_pretrained("./Meta-Llama-3-8B/")


# In[6]:


"""
messages = [
    {"role": "system", "content": "You are a programmer chatbot who always responds clearly and concisely!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)#.to(model.device)
"""
input_ids = tokenizer(
    ["I believe the meaning of life is"],
    return_tensors="pt"
)["input_ids"].to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# ## Forward passes

# In[8]:


config = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  #"max_position_embeddings": 8192,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "rope_theta": 500000.0,
  "tie_word_embeddings": False,
  "torch_dtype": torch.bfloat16,
  "transformers_version": "4.40.0.dev0",
  "use_cache": True,
  "vocab_size": 128256
}


# In[9]:


from ops.transformer_ops import Transformer


# In[10]:


model = Transformer(config, model_parser, device=device, preload_n_transformer_blocks=6)


# In[11]:


max_seq_len = 2048
max_gen_len = 32
min_prompt_len = min(len(t) for t in input_ids)
max_prompt_len = max(len(t) for t in input_ids)
assert max_prompt_len <= max_seq_len
total_len = min(max_seq_len, max_gen_len + max_prompt_len)


# In[12]:


pad_id = tokenizer.eos_token_id
batch_size = 1
prev_pos = 0

tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
for k, t in enumerate(input_ids):
    tokens[k, : len(t)] = t

eos_reached = torch.tensor([False] * batch_size, device=device)
input_text_mask = tokens != pad_id


# In[13]:


stop_tokens = torch.tensor([13], device="cpu") # 13=.


# In[14]:


#from tqdm import tqdm


# In[15]:


#for cur_pos in tqdm(range(min_prompt_len, total_len)):
for cur_pos in range(min_prompt_len, total_len):
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    # assume temperature 0
    next_token = torch.argmax(logits[:, -1], dim=-1)
    next_token = next_token.reshape(-1)
    print(f"Decoded token={tokenizer.decode(next_token)}")
    # only replace token if prompt has already been generated
    next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    tokens[:, cur_pos] = next_token
    """
    Needs to be on CPU:
    NotImplementedError: The operator 'aten::isin.Tensor_Tensor_out' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
    """
    is_in = torch.isin(next_token.cpu(), stop_tokens).to(device)
    eos_reached |= (~input_text_mask[:, cur_pos]) & (
        is_in
    )
    prev_pos = cur_pos
    if all(eos_reached):
        break

