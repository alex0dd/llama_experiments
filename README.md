```
conda create -n llama_exploration python=3.10
```
```
conda activate llama_exploration 
```
```
pip install -r requirements.txt
```

```
python -m cProfile -s time run_model.py > profile.text 2>&1
```

## Conversion scripts

### LLAMA3 8B

```
python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --quantization_type int8 --base_model_dir original_models/Meta-Llama-3-8B --output_model_dir converted_models/LLAMA-3-8B-PKL

python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --quantization_type int8 --base_model_dir original_models/Meta-Llama-3-8B-Instruct --output_model_dir converted_models/LLAMA-3-8B-INSTRUCT-PKL
```


### LLAMA 3.1 8B

```
python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --quantization_type int8 --base_model_dir original_models/Meta-Llama-3.1-8B --output_model_dir converted_models/LLAMA-3.1-8B-PKL

python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --quantization_type int8 --base_model_dir original_models/Meta-Llama-3.1-8B-Instruct --output_model_dir converted_models/LLAMA-3.1-8B-INSTRUCT-PKL
```

### Phi3

```
python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --quantization_type int8 --base_model_dir original_models/Phi-3-mini-4k-instruct --output_model_dir converted_models/PHI3-MINI-4K-PKL
```

### Mistral 7B v0.3

```
python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --quantization_type int8 --base_model_dir original_models/Mistral-7B-Instruct-v0.3 --output_model_dir converted_models/MISTRAL-7B-PKL
```

### Granite 3B/8B

```
python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --base_model_dir original_models/granite-3b-code-instruct --output_model_dir converted_models/GRANITE-3B-CODE-INSTRUCT-PKL --disable-llama-qk-remap --custom_model_type granite-small

python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --base_model_dir original_models/granite-3b-code-instruct --output_model_dir converted_models/GRANITE-3B-CODE-INSTRUCT-PKL --quantization_type int8 --no-quantize_embeddings --disable-llama-qk-remap --custom_model_type granite-small

python -m conversion_scripts.convert_safetensors_decoder_model_to_pkl --base_model_dir original_models/granite-8b-code-instruct --output_model_dir converted_models/GRANITE-8B-CODE-INSTRUCT-PKL --quantization_type int8 --no-quantize_embeddings --disable-llama-qk-remap --custom_model_type granite-small
```

## Running models

```
python -m example_notebooks.model_runner --model-dir ${PWD}/converted_models/LLAMA-3-8B-INSTRUCT-PKL-int8 --interaction-type chat

python -m example_notebooks.model_runner --model-dir ${PWD}/converted_models/LLAMA-3.1-8B-PKL-int8 --interaction-type completion

python -m example_notebooks.model_runner --model-dir ${PWD}/converted_models/LLAMA-3.1-8B-INSTRUCT-PKL-int8 --max-gen-len 4096
```

## Supported and tested models

The following models have been converted, quantized to at least int8 and tested for inference on PyTorch MPS:
* [LLAMA-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and [LLAMA-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* [LLAMA-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) and [LLAMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
* [Phi3-mini-4k](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
* [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [Granite-3b Base](https://huggingface.co/ibm-granite/granite-3b-code-base) and [Granite-3b Instruct](https://huggingface.co/ibm-granite/granite-3b-code-instruct)
* [Granite-8b Base](https://huggingface.co/ibm-granite/granite-8b-code-base) and [Granite-8b Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)


## Misc Resources
Safetensors format: https://huggingface.co/docs/safetensors/index
BF16: https://www.johndcook.com/blog/2018/11/15/bfloat16/

Quantization: 
https://cdn.files.pg.edu.pl/eti/KASK/Intel_HPML/05%20-%20LowPrecisionDL.pdf
https://www.reddit.com/r/LocalLLaMA/comments/1c7no52/psa_if_you_quant_your_llama_3_model_from_f16_you/
https://github.com/pytorch/ao/blob/main/torchao/dtypes/uint4.py
https://github.com/pytorch/pytorch/issues/74627
https://github.com/pytorch/ao/issues/47
https://mobiusml.github.io/hqq_blog/



Transformer:
https://towardsdatascience.com/deep-dive-into-llama-3-by-hand-%EF%B8%8F-6c6b23dc92b2