"""
comm_layer.py
Spatial Multi-Head Graph Attention (GAT) Communication Layer for V2V Networks.
Computes attentive message passing among neighboring vehicles subject to
dynamic graph topology and cyber-physical channel disturbances (packet drop/loss).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class V2VAttentionCommLayer(nn.Module):
    """
    Multi-Head Graph Attention layer modeling V2V message aggregation
    across dynamic wireless communication graphs.
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 2, dropout: float = 0.0):
        super(V2VAttentionCommLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"

        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(in_features, out_features, bias=False)
        self.w_k = nn.Linear(in_features, out_features, bias=False)
        self.w_v = nn.Linear(in_features, out_features, bias=False)
        
        # Output projection and normalization
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for V2V communication.
        h: Node features of shape (B, N, in_features) or (N, in_features)
        adj: Adjacency matrix of shape (B, N, N) or (N, N), with entries in {0, 1}
        Returns: Aggregated message representations of shape (B, N, out_features)
        """
        is_batched = h.dim() == 3
        if not is_batched:
            h = h.unsqueeze(0)
            adj = adj.unsqueeze(0)

        B, N, _ = h.shape

        # Compute Q, K, V
        q = self.w_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = self.w_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        v = self.w_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, N, N)

        # Expand adjacency mask for attention heads
        # adj is (B, N, N) -> (B, 1, N, N)
        adj_mask = adj.unsqueeze(1)
        
        # Mask out non-connected pairs (where adj == 0) with a large negative value
        mask_value = -1e9
        scores = scores.masked_fill(adj_mask == 0, mask_value)

        # Softmax over neighbors
        attn_weights = F.softmax(scores, dim=-1)
        # If an agent has no neighbors and disconnected self, softmax might yield NaNs, so fillna with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Compute aggregated context
        context = torch.matmul(attn_weights, v)  # (B, heads, N, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, N, self.out_features)  # (B, N, out_features)

        # Residual connection and layer normalization
        out = self.norm(self.out_proj(context))

        if not is_batched:
            out = out.squeeze(0)
        return out
