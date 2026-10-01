import torch
import torch.nn.functional as F


def _off_diagonal(matrix):
    n, m = matrix.shape
    if n != m:
        raise ValueError("expected a square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def contrastive_loss(clean, attacked, temperature=0.1):
    batch = clean.shape[0]
    embeddings = torch.cat([clean, attacked])
    similarities = embeddings @ embeddings.T / temperature
    similarities.fill_diagonal_(-torch.inf)
    targets = (torch.arange(2 * batch, device=embeddings.device) + batch) % (2 * batch)
    return F.cross_entropy(similarities, targets)


def hash_loss(clean_logits, attacked_logits, weights, binary_margin=1.0):
    clean, attacked = clean_logits.tanh(), attacked_logits.tanh()
    stability = F.mse_loss(clean, attacked)
    independence = _off_diagonal(clean @ clean.T / clean.shape[1]).square().mean()
    balance = clean.mean(0).square().mean()
    variance = 0.5 * (F.relu(0.5 - clean.std(0, unbiased=False)).mean()
                      + F.relu(0.5 - attacked.std(0, unbiased=False)).mean())
    centered = clean - clean.mean(0, keepdim=True)
    normalized = centered / centered.square().mean(0, keepdim=True).add(1e-4).sqrt()
    decorrelation = _off_diagonal(normalized.T @ normalized / normalized.shape[0]).square().mean()
    binary = 0.5 * (F.softplus(binary_margin - clean_logits.abs()).mean()
                    + F.softplus(binary_margin - attacked_logits.abs()).mean())
    parts = dict(stability=stability, independence=independence, balance=balance,
                 variance=variance, decorrelation=decorrelation, binary=binary)
    return sum(weights[name] * value for name, value in parts.items()), parts

