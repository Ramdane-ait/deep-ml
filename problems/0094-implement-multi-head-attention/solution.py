import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v),

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[-1])

    w_e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = w_e / np.sum(w_e, axis=-1, keepdims=True)
    return np.dot(weights, V)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = K.shape
    d_k = d_model // n_heads
    Q_m = Q.reshape((seq_len, n_heads, d_k)).transpose(1, 0, 2)
    K_m = K.reshape((seq_len, n_heads, d_k)).transpose(1, 0, 2)
    V_m = V.reshape((seq_len, n_heads, d_k)).transpose(1, 0, 2)

    attention_heads = []
    for n in range(n_heads):
        head = self_attention(Q_m[n,:,:], K_m[n,:,:], V_m[n,:,:])
        attention_heads.append(head)
    
    return np.concatenate(attention_heads, axis=1)