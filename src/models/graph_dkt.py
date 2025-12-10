"""
Graph-enhanced Knowledge Tracing models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..graph.gnn_layers import GCN, GAT, normalize_adjacency


class GraphDKT(nn.Module):
    """
    Graph-enhanced Deep Knowledge Tracing.
    Combines temporal modeling (LSTM) with graph structure (GNN).
    """
    
    def __init__(
        self,
        n_skills: int,
        embedding_dim: int = 100,
        hidden_dim: int = 200,
        graph_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        gnn_type: str = 'gcn',
        fusion_method: str = 'concat'
    ):
        """
        Initialize Graph-enhanced DKT.
        
        Args:
            n_skills: Number of unique skills
            embedding_dim: Skill embedding dimension
            hidden_dim: LSTM hidden dimension
            graph_hidden_dim: GNN hidden dimension
            num_lstm_layers: Number of LSTM layers
            num_gnn_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN ('gcn' or 'gat')
            fusion_method: How to fuse temporal and graph features ('concat', 'add', 'attention')
        """
        super(GraphDKT, self).__init__()
        
        self.n_skills = n_skills
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.fusion_method = fusion_method
        
        # Skill embeddings for temporal part
        self.skill_embeddings = nn.Embedding(n_skills, embedding_dim)
        
        # LSTM for temporal modeling
        lstm_input_dim = embedding_dim + 1  # embedding + correctness
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # GNN for graph structure
        hidden_dims = [graph_hidden_dim] * max(0, num_gnn_layers - 1)
        if gnn_type == 'gcn':
            self.gnn = GCN(
                in_features=embedding_dim,
                hidden_dims=hidden_dims,
                out_features=graph_hidden_dim,
                dropout=dropout
            )
        elif gnn_type == 'gat':
            self.gnn = GAT(
                in_features=embedding_dim,
                hidden_dims=hidden_dims,
                out_features=graph_hidden_dim,
                num_heads=4,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = hidden_dim + graph_hidden_dim
        elif fusion_method == 'add':
            # Need to match dimensions
            self.graph_proj = nn.Linear(graph_hidden_dim, hidden_dim)
            fusion_dim = hidden_dim
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout
            )
            self.graph_proj = nn.Linear(graph_hidden_dim, hidden_dim)
            fusion_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(fusion_dim, n_skills)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.skill_embeddings.weight)
    
    def get_graph_embeddings(
        self,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Get skill embeddings enhanced by graph structure.
        
        Args:
            adjacency: Adjacency matrix [n_skills, n_skills]
            
        Returns:
            Graph-enhanced embeddings [n_skills, graph_hidden_dim]
        """
        # Use skill embeddings as initial node features
        node_features = self.skill_embeddings.weight  # [n_skills, embedding_dim]
        
        # Apply GNN - GNN layers handle normalization internally
        graph_embeddings = self.gnn(node_features, adjacency)
        
        return graph_embeddings
    
    def forward(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            skills: Skill indices [batch_size, seq_len]
            corrects: Correctness [batch_size, seq_len]
            adjacency: Adjacency matrix [n_skills, n_skills]
            mask: Mask for valid positions [batch_size, seq_len]
            
        Returns:
            Predicted probabilities [batch_size, seq_len, n_skills]
        """
        batch_size, seq_len = skills.shape
        
        # Temporal part: LSTM
        skill_emb = self.skill_embeddings(skills)  # [batch_size, seq_len, embedding_dim]
        corrects_expanded = corrects.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        lstm_input = torch.cat([skill_emb, corrects_expanded], dim=-1)
        
        lstm_out, _ = self.lstm(lstm_input)  # [batch_size, seq_len, hidden_dim]
        
        # Graph part: GNN
        graph_emb = self.get_graph_embeddings(adjacency)  # [n_skills, graph_hidden_dim]
        
        # Get graph embeddings for current skills
        skill_graph_emb = graph_emb[skills]  # [batch_size, seq_len, graph_hidden_dim]
        
        # Fusion
        fused = lstm_out  # Default initialization
        if self.fusion_method == 'concat':
            fused = torch.cat([lstm_out, skill_graph_emb], dim=-1)
        elif self.fusion_method == 'add':
            graph_proj = self.graph_proj(skill_graph_emb)
            fused = lstm_out + graph_proj
        elif self.fusion_method == 'attention':
            # Use attention to fuse
            graph_proj = self.graph_proj(skill_graph_emb)
            # Reshape for attention: [seq_len, batch_size, hidden_dim]
            lstm_out_t = lstm_out.transpose(0, 1)
            graph_proj_t = graph_proj.transpose(0, 1)
            
            attn_out, _ = self.attention(lstm_out_t, graph_proj_t, graph_proj_t)
            fused = attn_out.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_dim]
        
        # Apply dropout
        fused = self.dropout(fused)
        
        # Output layer
        logits = self.fc(fused)  # [batch_size, seq_len, n_skills]
        probs = torch.sigmoid(logits)
        
        return probs


class TemporalGraphAttention(nn.Module):
    """
    Temporal Graph Attention model.
    Uses graph attention to model both temporal and structural dependencies.
    """
    
    def __init__(
        self,
        n_skills: int,
        embedding_dim: int = 100,
        hidden_dim: int = 200,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize Temporal Graph Attention model.
        
        Args:
            n_skills: Number of unique skills
            embedding_dim: Skill embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super(TemporalGraphAttention, self).__init__()
        
        self.n_skills = n_skills
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Skill embeddings
        self.skill_embeddings = nn.Embedding(n_skills, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim + 1, dropout)
        
        # Transformer encoder for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim + 1,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Graph attention for structural dependencies
        tg_hidden_dims = [hidden_dim] * max(0, num_layers - 1)
        self.graph_attention = GAT(
            in_features=embedding_dim,
            hidden_dims=tg_hidden_dims,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion and output
        self.fusion = nn.Linear(embedding_dim + 1 + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_skills)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.skill_embeddings.weight)
    
    def forward(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            skills: Skill indices [batch_size, seq_len]
            corrects: Correctness [batch_size, seq_len]
            adjacency: Adjacency matrix [n_skills, n_skills]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Predicted probabilities [batch_size, seq_len, n_skills]
        """
        # Get embeddings
        skill_emb = self.skill_embeddings(skills)  # [batch_size, seq_len, embedding_dim]
        corrects_expanded = corrects.unsqueeze(-1).float()
        
        # Temporal modeling with transformer
        temporal_input = torch.cat([skill_emb, corrects_expanded], dim=-1)
        temporal_input = self.pos_encoder(temporal_input)
        temporal_out = self.transformer(temporal_input, src_key_padding_mask=mask)
        
        # Graph modeling
        graph_emb = self.graph_attention(
            self.skill_embeddings.weight,
            adjacency
        )  # [n_skills, hidden_dim]
        
        skill_graph_emb = graph_emb[skills]  # [batch_size, seq_len, hidden_dim]
        
        # Fusion
        combined = torch.cat([temporal_out, skill_graph_emb], dim=-1)
        fused = F.relu(self.fusion(combined))
        fused = self.dropout(fused)
        
        # Output
        logits = self.fc(fused)
        probs = torch.sigmoid(logits)
        
        return probs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe.data[:, :x.size(1), :]  # type: ignore
        return self.dropout(x)
