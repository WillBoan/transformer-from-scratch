import torch
from torch import Tensor
from model import Head, MultiHeadAttention, FeedForward, Block, Transformer


torch.manual_seed(0)  # type: ignore


def test_head_forward():
    head = Head(head_size=4, n_embd=8, block_size=8)
    x = torch.randn(2, 4, 8)
    out: Tensor = head(x)
    assert out.shape == (2, 4, 4)


def test_mha_and_ffwd_and_block_backward():
    mha = MultiHeadAttention(num_heads=2, n_embd=8, block_size=8, dropout=0.0)
    ff = FeedForward(n_embd=8, dropout=0.0)
    block = Block(num_heads=2, n_embd=8, block_size=8, dropout=0.0)

    x = torch.randn(2, 4, 8, requires_grad=True)
    out_mha: Tensor = mha(x)  # type: ignore # noqa 841
    out_ff: Tensor = ff(x)  # type: ignore # noqa 841
    out_blk: Tensor = block(x)

    loss = out_blk.pow(2).mean()
    loss.backward()  # type: ignore

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_transformer_forward_backward_and_generate():
    vocab_size = 32
    model = Transformer(
        vocab_size=vocab_size,
        n_layer=1,
        num_heads=2,
        n_embd=8,
        block_size=8,
        dropout=0.0,
    )
    B, T = 2, 8
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    logits: Tensor
    loss: Tensor
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, vocab_size)
    assert loss is not None and loss.dim() == 0

    loss.backward()  # type: ignore
    assert model.token_embedding_table.weight.grad is not None

    # generate (no grads)
    start = torch.zeros((1, 1), dtype=torch.long)
    gen = model.generate(start, max_new_tokens=5)
    assert gen.shape == (1, 6)
