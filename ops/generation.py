import torch
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass

from utils.benchmarking import measure_performance

@dataclass
class GenerationConfig:
    """Configuration class for text generation parameters."""
    temperature: float = 0.6
    top_p: float = 0.9
    max_gen_len: int = 100
    stream_interval: int = 4
    stop_tokens_ids: Optional[List[int]] = None
    pad_id: Optional[int] = None
    echo: bool = False

class TextGenerator:

    def __init__(self, model, tokenizer):
        """
        Initialize the text generator.
        
        Args:
            model: The language model to use for generation
            tokenizer: The tokenizer for encoding/decoding text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.max_seq_len = model.max_seq_len
        self.device = self.model.device
        
    def _prepare_inputs(self, input_ids: List[List[int]], total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for generation.
        """
        pad_id = self.tokenizer.eos_id if hasattr(self.tokenizer, 'eos_id') else self.tokenizer.eos_token_id
        batch_size = len(input_ids)
        
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(input_ids):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
            
        input_text_mask = tokens != pad_id
        return tokens, input_text_mask

    def _get_stop_tokens(self, stop_tokens_ids: Optional[List[int]]) -> torch.Tensor:
        """
        Get stop tokens tensor.
        """
        device = self.device
        if stop_tokens_ids is None:
            stop_tokens_tensor = torch.tensor([13], device=device)  # Default stop token
        else:
            stop_tokens_tensor = torch.tensor(stop_tokens_ids, device=device)
        return stop_tokens_tensor

    @staticmethod
    def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Perform top-p (nucleus) sampling on a probability distribution.
        
        Args:
            probs: Probability distribution tensor
            p: Probability threshold for top-p sampling
            
        Returns:
            Sampled token indices
        """
        if p >= 1.0:
            # In this case, we take all tokens, so no sorting and masking is needed
            return torch.multinomial(probs, num_samples=1)
        else:
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            return torch.gather(probs_idx, -1, next_token)

    def _sample_next_token(self, logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """
        Sample the next token based on the logits and generation config.
        """
        if config.temperature > 0:
            probs = torch.softmax(logits[:, -1] / config.temperature, dim=-1)
            next_token = self.sample_top_p(probs, config.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        return next_token.reshape(-1)

    @measure_performance(name="generation")
    def generate(self, input_ids: List[List[int]], config: GenerationConfig) -> Tuple[List[str], int]:
        """
        Generate text given input token ids.
        
        Args:
            input_ids: List of input token sequences
            config: Generation configuration
            
        Returns:
            Tuple of (generated texts, total token count)
        """
        max_prompt_len = max(len(t) for t in input_ids)
        min_prompt_len = min(len(t) for t in input_ids)
        total_len = min(self.max_seq_len, config.max_gen_len + max_prompt_len)
        
        assert max_prompt_len <= self.max_seq_len, "Prompt length exceeds model's maximum sequence length"
        
        tokens, input_text_mask = self._prepare_inputs(input_ids, total_len)
        stop_tokens = self._get_stop_tokens(config.stop_tokens_ids)
        
        eos_reached = torch.tensor([False] * len(input_ids), device=self.device)
        prev_pos = 0
        
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = self._sample_next_token(logits, config)
            
            tokens[:, cur_pos] = next_token
            is_in = torch.isin(next_token, stop_tokens)
            eos_reached |= (~input_text_mask[:, cur_pos]) & is_in
            prev_pos = cur_pos
            
            if all(eos_reached):
                break
                
        return self._process_output(tokens, input_ids, config)

    def _process_output(self, tokens: torch.Tensor, input_ids: List[List[int]], 
                       config: GenerationConfig) -> Tuple[List[str], int]:
        """
        Process the generated tokens into final output text.
        """
        tokens_output = []
        total_tokens_count = 0
        
        for idx, generated_tokens in enumerate(tokens.tolist()):
            current_prompt_len = len(input_ids[idx])
            start_pos = 0 if config.echo else current_prompt_len
            generated_tokens = generated_tokens[start_pos:current_prompt_len + config.max_gen_len]
            
            if config.stop_tokens_ids:
                for stop_token in config.stop_tokens_ids:
                    try:
                        idx_of_stop_token = generated_tokens.index(stop_token)
                        generated_tokens = generated_tokens[:idx_of_stop_token]
                    except ValueError:
                        continue
                        
            total_tokens_count += len(generated_tokens)
            tokens_output.append(generated_tokens)
            
        return [self.tokenizer.decode(tokens) for tokens in tokens_output], total_tokens_count

    @measure_performance(name="generate_stream")
    def generate_stream(self, input_ids: List[List[int]], config: GenerationConfig) -> Iterator[Tuple[str, int, int]]:
        """
        Stream generated text token by token.
        
        Args:
            input_ids: List of input token sequences
            config: Generation configuration
            
        Yields:
            Tuples of (generated_text, token_count, current_position)
        """
        max_prompt_len = max(len(t) for t in input_ids)
        min_prompt_len = min(len(t) for t in input_ids)
        total_len = min(self.max_seq_len, config.max_gen_len + max_prompt_len)
        
        tokens, input_text_mask = self._prepare_inputs(input_ids, total_len)
        stop_tokens = self._get_stop_tokens(config.stop_tokens_ids)
        
        eos_reached = torch.tensor([False] * len(input_ids), device=self.device)
        prev_pos = 0
        
        out_tokens = []
        inp_len = 0
        to_send_n_tokens = 0
        
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(
                tokens[:, prev_pos:cur_pos], 
                prev_pos, 
                max_seq_len=total_len, 
                min_seq_len=min_prompt_len
            )
            
            next_token = self._sample_next_token(logits, config)
            tokens[:, cur_pos] = next_token
            
            is_in = torch.isin(next_token, stop_tokens)
            eos_reached |= (~input_text_mask[:, cur_pos]) & is_in
            prev_pos = cur_pos
            
            if all(eos_reached):
                break
                
            if to_send_n_tokens == 0:
                to_send_n_tokens = 0
                inp_len = len(out_tokens)
                
            out_tokens.append(next_token.item())
            to_send_n_tokens += 1
            
            if to_send_n_tokens == config.stream_interval:
                yield self.tokenizer.decode(out_tokens[inp_len:]), to_send_n_tokens, cur_pos
                inp_len = len(out_tokens)
                to_send_n_tokens = 0
                
        if to_send_n_tokens > 0:
            yield self.tokenizer.decode(out_tokens[inp_len:]), to_send_n_tokens, cur_pos