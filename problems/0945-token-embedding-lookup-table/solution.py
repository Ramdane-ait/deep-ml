import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list.
    """
    E = np.random.default_rng(seed).standard_normal((vocab_size, embed_dim))
	arr = E[token_ids, :].copy().tolist()
    return arr