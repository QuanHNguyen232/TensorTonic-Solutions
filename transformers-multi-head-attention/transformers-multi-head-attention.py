import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def self_attention(Q, K, V):
    (B, seq_len, d_k) = K.shape
    scores = np.matmul(Q, np.transpose(K, axes=(0, 2, 1))) / np.sqrt(d_k)
    weights = softmax(scores)
    out = np.matmul(weights, V)
    return out
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    """
    Q.shape = (B, seq_len, hid_size) --> (B, seq_len, num_heads*d_q)
    """
    (B, seq_len, hid_size) = Q.shape
    assert (Q.shape == K.shape) and (Q.shape == V.shape)

    K_heads = np.reshape(K, (B, seq_len, num_heads, -1))
    Q_heads = np.reshape(Q, (B, seq_len, num_heads, -1))
    V_heads = np.reshape(V, (B, seq_len, num_heads, -1))

    K_heads = np.transpose(K_heads, axes=(2, 0, 1, 3))
    Q_heads = np.transpose(Q_heads, axes=(2, 0, 1, 3))
    V_heads = np.transpose(V_heads, axes=(2, 0, 1, 3))

    heads_outs = []
    for i in range(num_heads):
        heads_outs.append(
            self_attention(Q_heads[i], K_heads[i], V_heads[i])
        )

    out = np.concatenate(heads_outs, axis=-1)
    return out
    