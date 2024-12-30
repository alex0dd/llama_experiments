import argparse
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

from utils.utils import load_json, save_json, get_all_eos_token_ids
from ops.generation import GenerationConfig, TextGenerator
from ops.transformer_ops import Transformer

MAGENTA = '\033[35m'
RESET = '\033[0m'

@dataclass
class AppConfig:
    """Application configuration class."""
    device: str
    model_dir: str
    streaming: bool
    interaction_type: str
    max_gen_len: int
    temperature: float = 0.6
    top_p: float = 0.9
    stream_interval: int = 4

class TextGenerationApp:
    """Main application class for text generation."""
    
    def __init__(self, config: AppConfig):
        """Initialize the text generation application."""
        self.config = config
        self.model = self._initialize_model()
        self.tokenizer = self._initialize_tokenizer()
        self.generator = TextGenerator(self.model, self.tokenizer)
        self.terminators = self._get_terminators()
        self.chat_history: List[Dict[str, str]] = []
        self.cur_pos = 0
        
    def _initialize_model(self) -> Transformer:
        """Initialize the transformer model."""
        model_config = load_json(f"{self.config.model_dir}/config.json")
        return Transformer(self.config.model_dir, model_config, device=self.config.device)
    
    def _initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_dir, 
            clean_up_tokenization_spaces=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_terminators(self) -> List[int]:
        """Get terminator token IDs."""
        terminators = [self.tokenizer.eos_token_id]
        terminators.extend(get_all_eos_token_ids(self.config.model_dir))
        return terminators

    def _handle_chat_command(self, command: str) -> bool:
        """Handle chat commands and return whether to skip generation."""
        if command == "/drop_history":
            self.chat_history = []
            print("[STATUS] Chat history dropped.")
            return True
            
        if command.startswith("/save_history"):
            tokens = command.split()
            assert len(tokens) == 2, "/save_history command needs only one path argument"
            history_path = tokens[-1]
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            save_json(history_path, self.chat_history)
            print(f"[STATUS] Chat history saved to {history_path}.")
            return True
            
        if command.startswith("/load_history"):
            tokens = command.split()
            assert len(tokens) == 2, "/load_history command needs only one path argument"
            history_path = tokens[-1]
            self.chat_history = load_json(history_path)
            self.cur_pos = 0
            print(f"[STATUS] Chat history loaded from {history_path}.")
            return True
            
        return False

    def _prepare_input_ids(self, user_input: str) -> List[List[int]]:
        """Prepare input IDs based on interaction type."""
        if self.config.interaction_type == "chat":
            self.chat_history.append({"role": "user", "content": user_input})
            return [self.tokenizer.apply_chat_template(
                self.chat_history, 
                tokenize=True, 
                add_generation_prompt=True
            )]
        return self.tokenizer(
            [user_input] if isinstance(user_input, str) else user_input,
        )["input_ids"]

    def generate_response(self, user_input: str) -> Optional[str]:
        """Generate response for user input."""
        if self.config.interaction_type == "chat":
            if self._handle_chat_command(user_input):
                return None
        
        input_ids = self._prepare_input_ids(user_input)
        gen_config = GenerationConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_gen_len=self.config.max_gen_len,
            stream_interval=self.config.stream_interval,
            stop_tokens_ids=self.terminators
        )
        
        output_text = []
        total_tokens_count = 0
        
        if self.config.interaction_type == "chat":
            print("Assistant: ", end='', flush=True)
            
        for word, n_tokens, gen_cur_pos, metrics in self.generator.generate_stream(input_ids, gen_config):
            print(MAGENTA + f"{word}" + RESET, end='', flush=True)
            output_text.append(word)
            total_tokens_count += n_tokens
            self.cur_pos = gen_cur_pos
            
        full_response = "".join(output_text)
        
        if self.config.interaction_type == "chat":
            self.chat_history.append({"role": "assistant", "content": full_response})
        else:
            self.cur_pos = 0
            
        print(f"\n[PERFORMANCE METRICS]\n{metrics}")
        return full_response

def parse_args() -> AppConfig:
    """Parse command line arguments and return AppConfig."""
    parser = argparse.ArgumentParser(description="Model runner script arguments.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="mps",
        help="Specify the device: cpu, cuda or mps. Defaults to 'mps'."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path of model directory containing the model to run."
    )
    parser.add_argument(
        "--disable-streaming",
        action="store_true",
        help="Disable streaming mode for responses."
    )
    parser.add_argument(
        "--interaction-type",
        type=str,
        choices=["chat", "completion"],
        default="chat",
        help="Type of interaction: chat or completion. Defaults to 'chat'."
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=256,
        help="Maximum length of generated text. Defaults to 256."
    )
    
    args = parser.parse_args()
    return AppConfig(
        device=args.device,
        model_dir=args.model_dir,
        streaming=not args.disable_streaming,
        interaction_type=args.interaction_type,
        max_gen_len=args.max_gen_len
    )

def main():
    """Main function to run the text generation application."""
    config = parse_args()
    app = TextGenerationApp(config)
    print("[STATUS] Model and tokenizer loaded successfully.")
    
    user_input = input("User: ").strip()
    while user_input != "/exit":
        app.generate_response(user_input)
        user_input = input("User: ").strip()

if __name__ == "__main__":
    main()