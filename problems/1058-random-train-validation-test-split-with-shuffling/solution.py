import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]
    shuffle_idx = np.random.default_rng(seed).permutation(n)
    train = int(n * train_frac)
    val = train + int(n * validation_frac)
    return [
        data[shuffle_idx[:train]],
        data[shuffle_idx[train: val]],
        data[shuffle_idx[val:]]
    ]
