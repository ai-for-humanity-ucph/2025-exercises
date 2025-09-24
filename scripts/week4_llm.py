"""
Usage:
    python scripts/week4_llm.py
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


def greedy_decoding(logit: torch.Tensor):
    """Performs greedy decoding"""
    return int(logit.argmax(dim=-1, keepdim=True).item())


def topk_sampling(logit: torch.Tensor, k: int = 10):
    """Performs top-k sampling"""
    values, indices = logit.topk(k)
    idx = int(Categorical(logits=values).sample().item())
    return int(indices[idx].item())


def generate_text(
    model: GPT2LMHeadModel,
    tokens: torch.Tensor,
    max_length: int,
    sample_fn: Callable[[torch.Tensor], int] = greedy_decoding,
) -> torch.Tensor:
    """Generates text from GPT2(-small).

    Args:
        model: GPT-2 model
        tokens: tensor of tokens to append to
        max_length: max length of generated text
        sample_fn: a function to sample token given logits

    Returns:
        tokens: tensor of tokens
    """
    while tokens.shape[1] < max_length:
        output = model(tokens)
        logits = output.logits.squeeze()
        next_tok = sample_fn(logits[-1])
        tokens = torch.cat([tokens, torch.tensor([[next_tok]])], dim=1)
    return tokens


def main():
    set_seed(42)

    generator = pipeline("text-generation", model="gpt2")
    extra_kwargs = dict(max_new_tokens=None, pad_token_id=50265)

    print(" Generated sequences ".center(80, "-"))
    for output in generator(
        "Hello, I'm a language model,",
        max_length=50,
        num_return_sequences=2,
        **extra_kwargs,
    ):
        print("-" * 45)
        print(output["generated_text"])
    print("\n")

    print(" Tokenizer and model ".center(80, "-"))
    print(generator.tokenizer, generator.model, sep="\n" + "-" * 45 + "\n")

    # Init model and tokenizer directly
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    print(model, model.config, sep="\n" + "-" * 45 + "\n")

    text = "Hello, I'm a language model,"
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input, labels=encoded_input["input_ids"])

    id_to_token = {i: t for t, i in tokenizer.get_vocab().items()}
    print(" Decoded tokens ".center(80, "-"))
    print([id_to_token[id_] for id_ in encoded_input["input_ids"].squeeze().tolist()])
    print(tokenizer.decode(encoded_input["input_ids"][0]))

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    logits = output.logits.squeeze()
    yhat = logits[:-1]
    y = encoded_input["input_ids"].squeeze()[1:]
    loss = loss_fn(yhat, y).item()

    # assert loss is computed correctly
    assert np.allclose(loss, output.loss.item())

    # Greedy decoding
    print(" Greedy decoding ".center(80, "-"))
    tokens = generate_text(
        model,
        tokenizer("Hello world,", return_tensors="pt")["input_ids"],
        max_length=27,
        sample_fn=greedy_decoding,
    )
    print(tokenizer.decode(tokens[0]))

    # top k sampling
    print(" topk sampling ".center(80, "-"))
    torch.manual_seed(2025)
    tokens = generate_text(
        model,
        tokenizer("Hello world,", return_tensors="pt")["input_ids"],
        max_length=65,
        sample_fn=partial(topk_sampling, k=10),
    )
    print(tokenizer.decode(tokens[0]))

    # Print number of parameters and approximate size in MB
    print(" Parameters ".center(80, "-"))
    print(f"Number of parameters: {model.num_parameters() / 10**6}")
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"Total size of weights: {total / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
