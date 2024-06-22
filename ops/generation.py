import torch

def sample_top_p(probs, p):
    """
    Taken from: https://github.com/meta-llama/llama3/blob/main/llama/generation.py
    
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate_text(model, tokenizer, input_ids, max_gen_len, temperature=0.6, top_p=0.9, stop_tokens_ids=None, streaming=False, echo=False):
    """
    If temperature > 0, then top_p is used for sampling.
    """
    device=model.device
    max_seq_len = model.max_seq_len
    min_prompt_len = min(len(t) for t in input_ids)
    max_prompt_len = max(len(t) for t in input_ids)
    assert max_prompt_len <= max_seq_len
    total_len = min(max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.eos_token_id
    batch_size = len(input_ids)
    prev_pos = 0
    
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(input_ids):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    eos_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != pad_id

    if stop_tokens_ids == None:
        stop_tokens = torch.tensor([13], device="cpu") # 13
        #stop_tokens = torch.tensor(list(tokenizer.stop_tokens))
    else:
        stop_tokens = torch.tensor(stop_tokens_ids, device="cpu")

    tokens_output = []

    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
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
    tokens_output = []
    total_tokens_count = 0
    for idx, generated_tokens in enumerate(tokens.tolist()):
        current_prompt_len = len(input_ids[idx])
        start_pos = 0 if echo else current_prompt_len
        generated_tokens = generated_tokens[start_pos: current_prompt_len + max_gen_len]
        for stop_token in stop_tokens_ids:
            try:
                idx_of_stop_token = generated_tokens.index(stop_token)
                generated_tokens = generated_tokens[:idx_of_stop_token]
            except ValueError:
                pass
        total_tokens_count += len(generated_tokens)
        tokens_output.append(generated_tokens)
    decoded_tokens = [tokenizer.decode(generated_tokens) for generated_tokens in tokens_output]
    return decoded_tokens, total_tokens_count


def generate_text_stream(model, tokenizer, input_ids, max_gen_len, temperature=0.6, top_p=0.9, stop_tokens_ids=None, stream_interval=4):
    """
    If temperature > 0, then top_p is used for sampling.
    """
    device=model.device
    max_seq_len = model.max_seq_len
    min_prompt_len = min(len(t) for t in input_ids)
    max_prompt_len = max(len(t) for t in input_ids)
    assert max_prompt_len <= max_seq_len
    total_len = min(max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.eos_token_id
    batch_size = len(input_ids)
    prev_pos = 0
    
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(input_ids):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    eos_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != pad_id

    if stop_tokens_ids == None:
        stop_tokens = torch.tensor([13], device="cpu") # 13
        #stop_tokens = torch.tensor(list(tokenizer.stop_tokens))
    else:
        stop_tokens = torch.tensor(stop_tokens_ids, device="cpu")

    out_resp = ""
    out_tokens = []
    inp_len = 0
    to_send_n_tokens = 0

    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
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

        # Workaround for https://github.com/huggingface/transformers/issues/22710
        if to_send_n_tokens == 0:
            to_send_n_tokens = 0
            inp_len = len(out_resp)
        out_tokens.append(next_token.item())
        to_send_n_tokens += 1
        out_resp = tokenizer.decode(out_tokens)
        if to_send_n_tokens == stream_interval:
            yield out_resp[inp_len:], to_send_n_tokens
            inp_len = 0
            to_send_n_tokens = 0
    # if we finished the generation, but some tokens are still not flushed
    if inp_len != 0:
        yield out_resp[inp_len:], to_send_n_tokens