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

```
python convert_safetensors_llama3_model_to_pkl.py --quantization_type int8
python convert_safetensors_phi3_model_to_pkl.py --quantization_type int8
```

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