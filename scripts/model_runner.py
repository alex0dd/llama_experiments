import argparse
import time
import torch

from pathlib import Path

from transformers import AutoTokenizer
from utils.utils import load_json, save_json


from ops.transformer_ops import Transformer
from ops.generation import generate_text, generate_text_stream

MAGENTA = '\033[35m'
RESET = '\033[0m' # called to return to standard terminal text color

def parse_all_args():
    # Create the argument parser
    arg_parser = argparse.ArgumentParser(description="Model runner script arguments.")
    arg_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="mps",
        help="Specify the device: cpu, cuda or mps. Defaults to 'mps' if not specified.",
    )
    arg_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Specify the path of model directory containing the model to run, after the conversion.",
    )
    arg_parser.add_argument(
        "--disable-streaming",
        action="store_true",
        help="If specified, will disable streaming mode for responses.",
    )
    arg_parser.add_argument(
        "--interaction-type",
        type=str,
        choices=["chat", "completion"],
        default="chat",
        help="Specify the type of interaction: chat or completion. Defaults to 'chat' if not specified.",
    )
    arg_parser.add_argument(
        "--max-gen-len",
        type=int,
        default=256,
        help="Specify the maximum length of generated text. Defaults to 256 if not specified.",
    )

    # Parse the arguments
    args = arg_parser.parse_args()
    return args

args = parse_all_args()

streaming=not args.disable_streaming
device=args.device
model_dir = args.model_dir
interaction_type = args.interaction_type
max_gen_len = args.max_gen_len
config = load_json(f"{model_dir}/config.json")

def text_to_ids(text, tokenizer):
    if type(text) != list: text = [text]
    input_ids = tokenizer(
        text,
    )["input_ids"]
    return input_ids

model = Transformer(model_dir, config, device=device)

tokenizer = AutoTokenizer.from_pretrained(model_dir, clean_up_tokenization_spaces=False)
tokenizer.pad_token = tokenizer.eos_token

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

#TODO: make all status prints logging debug/info
print("[STATUS] Model and tokenizer loaded successfully.")


cur_pos = 0
is_chat = interaction_type == "chat"
if is_chat:
    chat_history = []
user_input_text = input("User: ").strip()
while user_input_text != "/exit":
    if is_chat:
        # TODO: package this conditional into functions
        skip_generation = True
        if user_input_text == "/drop_history":
            chat_history = []
            print("[STATUS] Chat history dropped.")
        elif user_input_text.startswith("/save_history"):
            # Expected format "/save_history path_to_history.json"
            command_tokens = user_input_text.split()
            assert len(command_tokens) == 2, "/save_history command needs only one path argument"
            history_path = command_tokens[-1]
            # Make dir if doesn't exist
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            save_json(history_path, chat_history)
            print(f"[STATUS] Chat history saved to {history_path}.")
        elif user_input_text.startswith("/load_history"):
            # Expected format "/load_history path_to_history.json"
            command_tokens = user_input_text.split()
            assert len(command_tokens) == 2, "/load_history command needs only one path argument"
            history_path = command_tokens[-1]
            chat_history = load_json(history_path)
            cur_pos = 0
            print(f"[STATUS] Chat history loaded from {history_path}.")
        else:
            chat_history.append({"role": "user", "content": user_input_text})
            input_ids = tokenizer.apply_chat_template(chat_history, tokenize=True, add_generation_prompt=True)
            input_ids = [input_ids]
            skip_generation = False

        if skip_generation:
            user_input_text = input("User: ").strip()
            continue
    else:
        input_ids = text_to_ids(user_input_text, tokenizer)
    output_text = []
    total_tokens_count = 0
    start_time = time.time()
    if is_chat:
        print("Assistant: ", end='', flush=True)
    for word, n_tokens, gen_cur_pos in generate_text_stream(model, tokenizer, input_ids, max_gen_len=max_gen_len, stop_tokens_ids=terminators, prev_pos=cur_pos):
        print(MAGENTA+f"{word}"+RESET, end='', flush=True)
        output_text.append(word)
        total_tokens_count += n_tokens
        cur_pos = gen_cur_pos
    delta_time = time.time() - start_time
    output_text = "".join(output_text)
    if is_chat:
        chat_history.append({"role": "assistant", "content": output_text})
    else:
        # in completion mode we reset the current position only
        cur_pos = 0
    print()
    print(f"[STATUS] Generation took {delta_time} seconds, {total_tokens_count/delta_time} tokens/s.")
    user_input_text = input("User: ").strip()
