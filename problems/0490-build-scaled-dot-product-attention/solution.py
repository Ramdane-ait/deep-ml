import numpy as np


def softmax(x):
	e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
	return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	d_k = Q.shape[-1]
	scores = np.dot(Q, K.T) / np.sqrt(d_k)
	if mask is not None:
		scores = np.where(mask == 0, -np.inf, scores)
	attention_weights = softmax(scores)
	output = np.dot(attention_weights, V)
	return output, attention_weights