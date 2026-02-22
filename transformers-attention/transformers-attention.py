import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    (B, seq_len, d_k) = K.shape
    attn_scores = torch.matmul(Q, torch.transpose(K, 1, 2)) / math.sqrt(d_k) # (B, seq_len, d_k) x (B, d_k, seq_len) = (B, seq_len, seq_len)
    attn_weights = F.softmax(attn_scores, dim=-1) # (B, seq_len, seq_len)
    out = torch.matmul(attn_weights, V)# (B, seq_len, seq_len) x (B, seq_len, d_v) = (B, seq_len, d_v)
    return out