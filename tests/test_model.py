import torch
from torch import Tensor
import torch.nn as nn
from model import Head, MultiHeadAttention, FeedForward, Block, Transformer


torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


def test_head_forward():
    B, T, C = 2, 8, 8
    head_size = 4

    head = Head(head_size=head_size, block_size=T, n_embd=C)
    x = torch.randn(B, T, C)
    out: Tensor = head(x)

    assert out.shape == (B, T, head_size)


def test_head_causal_mask():
    """
    Causality check:
    Changing the LAST token should NOT affect outputs at earlier timesteps.
    """

    B, T, C = 2, 8, 8
    head_size = 4

    head = Head(head_size=head_size, block_size=T, n_embd=C)
    x = torch.randn(B, T, C)
    x2 = x.clone()

    x2[:, -1, :] += 999.0  # big perturbation at the final position

    out1: Tensor = head(x)
    out2: Tensor = head(x2)

    # earlier timesteps unchanged (tolerance)
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
    # last timestep changed
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :], atol=1e-5)


def test_head_gradients():
    B, T, C = 2, 8, 8
    head_size = 4

    head = Head(head_size=head_size, block_size=T, n_embd=C)
    x = torch.randn(B, T, C, requires_grad=True)

    out: Tensor = head(x)
    loss: Tensor = out.pow(2).mean()  # simple loss
    loss.backward()  # type: ignore

    # input grads
    assert x.grad is not None, "Input gradients not computed"
    assert x.grad.shape == x.shape, "Input gradients have incorrect shape"
    assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    # weight grads
    assert head.query.weight.grad is not None
    assert head.key.weight.grad is not None
    assert head.value.weight.grad is not None
    assert torch.isfinite(head.query.weight.grad).all()
    assert torch.isfinite(head.key.weight.grad).all()
    assert torch.isfinite(head.value.weight.grad).all()

    # check causal mask buffer exists
    assert hasattr(head, "tril")
    assert head.tril is not None
    assert head.tril.shape[0] >= T


def test_multi_head_attention_forward():
    B, T, C = 2, 8, 8
    n_head = 2

    mha = MultiHeadAttention(n_head=n_head, block_size=T, n_embd=C)
    x = torch.randn(B, T, C)
    out: Tensor = mha(x)

    assert out.shape == (B, T, C)


def test_multi_head_attention_gradients():
    B, T, C = 2, 8, 8
    n_head = 2

    mha = MultiHeadAttention(n_head=n_head, block_size=T, n_embd=C)
    x = torch.randn(B, T, C, requires_grad=True)

    out: Tensor = mha(x)
    loss: Tensor = out.pow(2).mean()  # simple loss
    loss.backward()  # type: ignore

    # input grads
    assert x.grad is not None, "Input gradients not computed"
    assert x.grad.shape == x.shape, "Input gradients have incorrect shape"
    assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    # check each head's weights have gradients
    head: Head
    for head in mha.heads:  # type: ignore
        assert head.query.weight.grad is not None
        assert head.key.weight.grad is not None
        assert head.value.weight.grad is not None
        assert torch.isfinite(head.query.weight.grad).all()
        assert torch.isfinite(head.key.weight.grad).all()
        assert torch.isfinite(head.value.weight.grad).all()

    # check output projection has gradients
    assert mha.proj.weight.grad is not None
    assert torch.isfinite(mha.proj.weight.grad).all()


def test_feed_forward_forward():
    B, T, C = 2, 8, 8

    ff = FeedForward(n_embd=C)
    x = torch.randn(B, T, C)
    out: Tensor = ff(x)

    assert out.shape == (B, T, C)


def test_feed_forward_gradients():
    B, T, C = 2, 8, 8

    ff = FeedForward(n_embd=C)
    x = torch.randn(B, T, C, requires_grad=True)

    out: Tensor = ff(x)
    loss: Tensor = out.pow(2).mean()  # simple loss
    loss.backward()  # type: ignore

    # Accessing layers inside nn.Sequential
    linear1: nn.Linear = ff.net[0]  # type: ignore
    linear2: nn.Linear = ff.net[2]  # type: ignore

    # check layer types
    assert isinstance(linear1, nn.Linear)
    assert isinstance(linear2, nn.Linear)
    assert isinstance(ff.net[1], nn.ReLU)
    assert isinstance(ff.net[3], nn.Dropout)

    # input grads
    assert x.grad is not None, "Input gradients not computed"
    assert x.grad.shape == x.shape, "Input gradients have incorrect shape"
    assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    # check weights have gradients
    assert linear1.weight.grad is not None
    assert linear1.bias.grad is not None
    assert linear2.weight.grad is not None
    assert linear2.bias.grad is not None
    assert torch.isfinite(linear1.weight.grad).all()
    assert torch.isfinite(linear1.bias.grad).all()
    assert torch.isfinite(linear2.weight.grad).all()
    assert torch.isfinite(linear2.bias.grad).all()


def test_block_forward():
    B, T, C = 2, 8, 8
    n_head = 2

    block = Block(n_head=n_head, block_size=T, n_embd=C)
    x = torch.randn(B, T, C)
    out: Tensor = block(x)

    assert out.shape == (B, T, C)


def test_block_gradients():
    B, T, C = 2, 8, 8
    n_head = 2

    block = Block(n_head=n_head, block_size=T, n_embd=C)
    x = torch.randn(B, T, C, requires_grad=True)

    out: Tensor = block(x)
    loss: Tensor = out.pow(2).mean()  # simple loss
    loss.backward()  # type: ignore

    # input grads
    assert x.grad is not None, "Input gradients not computed"
    assert x.grad.shape == x.shape, "Input gradients have incorrect shape"
    assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    assert isinstance(block.mha, MultiHeadAttention)
    assert isinstance(block.ffwd, FeedForward)
    assert isinstance(block.ln1, nn.LayerNorm)
    assert isinstance(block.ln2, nn.LayerNorm)


def test_transformer_forward():
    B, T, C = 2, 8, 8
    vocab_size = 100
    n_layer = 2
    n_head = 2

    model = Transformer(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        block_size=T,
        n_embd=C,
    )

    # Create dummy input and target tensors
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    logits: Tensor
    loss: Tensor
    logits, loss = model(idx, targets)

    loss_value = loss.item()

    assert logits.shape == (B, T, vocab_size), "Logits have incorrect shape"
    assert loss is not None, "Loss should not be None when targets are provided"
    assert loss.ndim == 0, "Loss should be a scalar tensor"
    assert loss.shape == (), "Loss should be a scalar tensor"
    assert torch.isfinite(loss).all(), "Loss is not finite"
    assert isinstance(loss_value, float), "Loss should be a Python float"
    assert loss_value > 0, "Loss should be positive"


def test_transformer_gradients():
    B, T, C = 2, 8, 8
    vocab_size = 100
    n_layer = 2
    n_head = 2

    model = Transformer(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        block_size=T,
        n_embd=C,
    )

    # Create dummy input and target tensors
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    _logits: Tensor
    loss: Tensor
    _logits, loss = model(idx, targets)

    loss.backward()  # type: ignore

    block1: Block = model.blocks[0]  # type: ignore

    # Check gradients in key parts of the model
    assert model.token_embedding_table.weight.grad is not None
    assert block1.mha.proj.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.token_embedding_table.weight.grad).all()
    assert torch.isfinite(block1.mha.proj.weight.grad).all()
    assert torch.isfinite(model.lm_head.weight.grad).all()
