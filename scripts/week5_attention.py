"""
Usage:
    python scripts/week5_attention.py
"""

import math

import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn.functional as F
from circuitsvis.attention import attention_heads
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()


def chicken_example():
    """chicken example"""

    sentence = """The chicken did not cross the road because it was too tired."""
    text = tokenizer.bos_token + sentence
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    str_tokens = tokenizer.batch_decode(
        # note squeeze: expects inputs of shape (n,) where n sequence length
        inputs["input_ids"].squeeze(),
        clean_up_tokenization_spaces=False,
    )
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    print(str_tokens)

    html = attention_heads(
        tokens=str_tokens,
        # note: important that we squeeze s.t. shape of input tensor:
        # torch.Size([1, 12, 14, 14]) -> torch.Size([12, 14, 14])
        attention=outputs.attentions[0].squeeze(),
    )
    with open("output/l1.html", "w") as f:
        f.write(str(html))

    html = attention_heads(
        tokens=str_tokens,
        attention=outputs.attentions[-1].squeeze(),
    )
    with open("output/l12.html", "w") as f:
        f.write(str(html))


def random_example():
    """randomly generated input sequence"""

    torch.manual_seed(2025)
    seq_len = 20
    random_tokens = torch.randint(low=0, high=100, size=(1, seq_len))
    tokenizer.decode(random_tokens.squeeze())
    tokens = torch.cat(
        (torch.Tensor([[tokenizer.bos_token_id]]), random_tokens, random_tokens), dim=-1
    ).long()
    with torch.no_grad():
        outputs = model(tokens, output_attentions=True)
    logits = outputs.logits
    str_tokens = tokenizer.batch_decode(
        tokens.squeeze(), clean_up_tokenization_spaces=False
    )
    print(f"Random tokens:\n{str_tokens}")

    i = 5  # layer 6
    html = attention_heads(
        tokens=str_tokens,
        attention=outputs.attentions[i].squeeze(),
    )
    with open(f"output/l{i + 1}-random.html", "w") as f:
        f.write(str(html))

    """
    Explanation for the loss calculation below:
    - Remove last element of logits away and first element of y (true labels)
    - Then pairing the logits and labels yields pair of logit and next token's
        true label
    - The indexing on `lob_probs_mat` below picks out column `y_i` for 
        each i ∈ {0, 1, ..., log_probs_mat.shape[0] - 1}.
      I.e. the log predicted probability of the correct next token.
      The negative of this is the loss for element i.
      See e.g. https://ai-for-humanity-ucph.github.io/2025/slides/lecture-2/#/33.
    """
    log_probs_mat = logits.squeeze().log_softmax(dim=-1)[:-1]
    y = tokens.squeeze()[1:]
    log_probs = log_probs_mat[torch.arange(log_probs_mat.shape[0]), y].detach().numpy()

    print(" Loss random repeated sequence ".center(80, "-"))
    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(log_probs) + 1), log_probs, marker="o")
    ax.set_xticks(range(1, len(log_probs) + 1))
    ax.set_xlabel("Sequence position, $i$")
    ax.set_ylabel(r"Log probability, $\log \hat{y}_{i}$")
    ax.axvline(x=seq_len, label="Seq. length $n$", color="black", linestyle="--")
    ax.set_title(r"Log probabilities across repeated $\mathit{random}$ sequence")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    fig.savefig("figs/loss_random.png")


def multi_head_attention_example():
    """Example of Multi Head Attention using weights from GPT-2.


    Goal is to compute `A` from:
        https://ai-for-humanity-ucph.github.io/2025/exercises/class-5/#eq-attention-matrix
    for the first transformer block of GPT-2.
    This corresponds to the output of `attn` below for `model.transformer.h[0]`
    (the first transformer block of the model) in "Model architecture" below.
    We start out with some tokens denoted `inputs` below.
    The way up to the output of `attn` is:
        1. Embed tokens with `wte` (token embeddings)
        2. Embed tokens with `wtp` (position embeddings)
        3. Add above two embeddings together and apply layer norm.
           At this point we are at `ln_1` in the model.
           Call the tensor at this point for `hidden`.
        4. Applying `attn` to `hidden` yields the attention matrix A and output O.
           We want to compute it as in Equation (1) of the exercise sheet
           using the weight matrices of the model.

    Model architecture:
        ```python
        print(model)
        ```
        yields:
        ```
        GPT2LMHeadModel(
          (transformer): GPT2Model(
            (wte): Embedding(50257, 768)
            (wpe): Embedding(1024, 768)
            (drop): Dropout(p=0.1, inplace=False)
            (h): ModuleList(
              (0-11): 12 x GPT2Block(
                (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (attn): GPT2Attention(
                  (c_attn): Conv1D(nf=2304, nx=768)
                  (c_proj): Conv1D(nf=768, nx=768)
                  (attn_dropout): Dropout(p=0.1, inplace=False)
                  (resid_dropout): Dropout(p=0.1, inplace=False)
                )
                (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): GPT2MLP(
                  (c_fc): Conv1D(nf=3072, nx=768)
                  (c_proj): Conv1D(nf=768, nx=3072)
                  (act): NewGELUActivation()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
            (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (lm_head): Linear(in_features=768, out_features=50257, bias=False)
        )
        ```
    """

    cfg = model.config
    d_emb = cfg.n_embd
    n_head = cfg.n_head
    d_head = d_emb // n_head

    sentence = """The chicken did not cross the road because it was too tired."""
    text = tokenizer.bos_token + sentence
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        att_l0 = outputs.attentions[0].squeeze()

    # Embed tokens
    tokens = inputs["input_ids"]
    E = model.transformer.wte(tokens)
    P = model.transformer.wpe(t.arange(tokens.shape[1]))
    X = E + P
    # Apply layer norm using layer norm of transformer block 1
    h0 = model.transformer.h[0]
    hidden = h0.ln_1(X)
    _, seq, d_emb = hidden.shape
    # Compute attention using model's implementation
    # We will compare out implementation with this
    O, A = h0.attn(hidden)

    """
    Notes on extracting the weight matrices:
    - W is of shape torch.Size([768, 2304])
    - b is of shape torch.Size([2304])
    - Splitting W along dim=1 yields 3 matrices of shape torch.Size([768, 768]).
    - Likewise for bias term b.
    - We further split those into 12 parts, one for each head, each of dimension
      torch.Size([768, 64]).
    - The authors from OpenAI have coded it this way to compute the linear
      transformations for all heads at once.
    - This is more efficient but doesn't match the equations from the book.
    """
    W = h0.attn.c_attn.weight
    b = h0.attn.c_attn.bias
    W_Q, W_K, W_V = W.split(d_emb, dim=1)
    b_Q, b_K, b_V = b.split(d_emb, dim=0)
    W_O = h0.attn.c_proj.weight
    b_O = h0.attn.c_proj.bias

    # Split weight matrices into tuples of 12 tensors; one for each of 12 heads
    W_Qs = W_Q.split(d_head, dim=1)
    b_Qs = b_Q.split(d_head, dim=0)
    W_Ks = W_K.split(d_head, dim=1)
    b_Ks = b_K.split(d_head, dim=0)
    W_Vs = W_V.split(d_head, dim=1)
    b_Vs = b_V.split(d_head, dim=0)
    W_Os = W_O.split(d_head, dim=0)

    heads = dict()
    # Squeeze out batch dimension so we can do simple 2-dimensional matrix
    # multiplication
    h = hidden.squeeze()
    for i in range(12):
        Q = h @ W_Qs[i] + b_Qs[i]
        K = h @ W_Ks[i] + b_Ks[i]
        V = h @ W_Vs[i] + b_Vs[i]
        # Rest is just applying the equations:
        # https://ai-for-humanity-ucph.github.io/2025/exercises/class-5/#eq-attention-output
        QKT = torch.matmul(Q, K.t()) / math.sqrt(d_head)
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=Q.device), diagonal=1
        )
        QKT = QKT.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(QKT, dim=-1)
        # Assert attention matrices match the output from `model()` and
        # `h0.attn` (they compute the same).
        assert torch.allclose(attn_weights, att_l0[i])
        assert torch.allclose(attn_weights, A[0][i])
        head = attn_weights @ V
        heads[i] = head

    # Each output head is of dimension (n, 64); stack along dim 1 to get
    # `aheads` of dimension (14, 768); then we can apply `W_O` and `b_O`
    # directly on all heads at once to get the output, `O`.
    # Note that `O` here is the sum of the `O`s over all 12 heads with bias
    # `b_O` added.
    aheads = torch.cat([v[:, :] for v in heads.values()], dim=1)
    assert t.allclose(aheads @ W_O + b_O, O, atol=1e-6)

    # Sum of 12 heads with bias term added once only
    # WARNING: Spaghetti computation, don't ever do this in real life.
    O_other = (
        torch.cat(
            [(head @ W_Os[i]).view(14, 768, 1) for i, head in heads.items()], dim=2
        ).sum(dim=-1)
        + b_O
    )
    assert t.allclose(O_other, O.squeeze(), atol=1e-5)

    print("Attention checks passed")


def main():
    chicken_example()
    random_example()
    multi_head_attention_example()


if __name__ == "__main__":
    main()
