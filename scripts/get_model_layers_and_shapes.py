import argparse
from pprint import pprint

from parsers import ModelParser
from utils.utils import get_all_safetensors_model_files, load_json


def parse_all_args():
    # Create the argument parser
    arg_parser = argparse.ArgumentParser(
        description="Parse quantization type argument."
    )
    arg_parser.add_argument(
        "--base_model_dir",
        type=str,
        default="Meta-Llama-3-8B",
        help="Specify the path of base model directory containing LLAMA3. Defaults to 'Meta-Llama-3-8B' if not specified.",
    )

    # Parse the arguments
    args = arg_parser.parse_args()
    return args


args = parse_all_args()
print(args)

base_model_dir = args.base_model_dir

model_files = get_all_safetensors_model_files(base_model_dir)
model_parser = ModelParser(model_files)
config = load_json(f"./{base_model_dir}/config.json")

pprint(model_parser.tensor_shapes)
