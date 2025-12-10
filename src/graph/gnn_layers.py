"""Graph Neural Network layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix.
    A_norm = D^{-1/2} A D^{-1/2}
    """
    # Add self-loops
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    
    # Compute degree
    degree = adj.sum(dim=1)
    degree[degree == 0] = 1  # Avoid division by zero
    
    # D^{-1/2}
    deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree))
    
    # Normalize
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    
    return adj_norm


class GraphConvolution(nn.Module):
    """Graph Convolution Layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output [num_nodes, out_features]
        """
        out = torch.matmul(x, self.weight)
        out = torch.matmul(adj, out)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GCN(nn.Module):
    """Graph Convolutional Network."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: list,
        out_features: int,
        dropout: float = 0.3
    ):
        super(GCN, self).__init__()
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            self.layers.append(GraphConvolution(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.layers.append(GraphConvolution(prev_dim, out_features))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Node embeddings [num_nodes, out_features]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x, adj)
        return x


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.3):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention weights
        self.a = nn.Parameter(torch.FloatTensor(1, num_heads, 2 * self.head_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output [num_nodes, out_features]
        """
        # Linear transformation
        h = self.W(x)  # [num_nodes, out_features]
        num_nodes = h.size(0)
        
        # Reshape for multi-head attention
        h = h.view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention
        a_input = torch.cat([h[adj._indices()[0]], h[adj._indices()[1]]], dim=-1)
        
        # Attention scores
        e = torch.matmul(a_input, self.a.squeeze(0).t())
        e = F.leaky_relu(e, negative_slope=0.2)
        
        # Create sparse attention matrix
        attention = torch.sparse_coo_tensor(
            adj._indices(),
            e.squeeze(-1) if e.dim() > 1 else e,
            adj.size()
        )
        attention = attention.coalesce()
        
        # Softmax
        indices = attention._indices()
        values = attention._values()
        
        # Compute row-wise softmax
        row_max = torch.zeros(num_nodes, device=values.device)
        row_max[indices[0]] = torch.max(row_max[indices[0]], values)
        values = values - row_max[indices[0]]
        values = torch.exp(values)
        
        row_sum = torch.zeros(num_nodes, device=values.device)
        row_sum[indices[0]] += values
        row_sum[row_sum == 0] = 1
        
        values = values / row_sum[indices[0]]
        values = F.dropout(values, p=self.dropout, training=self.training)
        
        attention = torch.sparse_coo_tensor(indices, values, adj.size()).coalesce()
        
        # Apply attention
        out = torch.sparse.mm(attention.float(), h.view(num_nodes, -1))
        
        return out


class GAT(nn.Module):
    """Graph Attention Network."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: list,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            self.layers.append(GraphAttentionLayer(prev_dim, hidden_dim, num_heads, dropout))
            prev_dim = hidden_dim
        
        self.layers.append(GraphAttentionLayer(prev_dim, out_features, num_heads, dropout))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Node embeddings [num_nodes, out_features]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x, adj)
        return x
