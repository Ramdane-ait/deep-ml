import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    # Your code here
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(data) - block_size, size=batch_size)
    x = np.stack([data[i: i + block_size] for i in ix])
    y = np.stack([data[i + 1: i + block_size + 1] for i in ix])

    return np.array([x, y])