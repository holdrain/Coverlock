import torch

from coverlock.losses import contrastive_loss, hash_loss
from coverlock.model import CoverLockHead


def test_head_shape_and_deterministic_projection():
    first = CoverLockHead(32, message_bits=16, hidden=24, embedding_dim=32, hash_seed=7)
    second = CoverLockHead(32, message_bits=16, hidden=24, embedding_dim=32, hash_seed=7)
    assert torch.equal(first.hash_projection, second.hash_projection)
    logits, embedding = first(torch.randn(3, 32, 4, 4), return_embeddings=True)
    assert logits.shape == (3, 16)
    assert embedding.shape == (3, 32)


def test_losses_are_finite():
    clean = torch.randn(4, 16)
    attacked = torch.randn(4, 16)
    weights = dict(stability=2, independence=1, balance=5,
                   variance=5, decorrelation=.05, binary=.05)
    value, _ = hash_loss(clean, attacked, weights)
    assert torch.isfinite(value)
    assert torch.isfinite(contrastive_loss(torch.randn(4, 8), torch.randn(4, 8)))
