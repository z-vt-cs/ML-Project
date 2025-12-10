"""
Deep Knowledge Tracing (DKT) model implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DKTModel(nn.Module):
    """Deep Knowledge Tracing using LSTM/GRU."""
    
    def __init__(
        self,
        n_skills: int,
        hidden_dim: int = 200,
        num_layers: int = 2,
        dropout: float = 0.3,
        model_type: str = 'lstm'
    ):
        """
        Initialize DKT model.
        
        Args:
            n_skills: Number of unique skills
            hidden_dim: Hidden dimension size
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            model_type: 'lstm' or 'gru'
        """
        super(DKTModel, self).__init__()
        
        self.n_skills = n_skills
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Input size is 2 * n_skills (skill one-hot + correctness)
        self.input_dim = 2 * n_skills
        
        # Recurrent layer
        if model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, n_skills)
        
    def _encode_input(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode skills and correctness as one-hot vectors.
        
        Args:
            skills: Skill indices [batch_size, seq_len]
            corrects: Correctness [batch_size, seq_len]
            
        Returns:
            Encoded input [batch_size, seq_len, 2 * n_skills]
        """
        batch_size, seq_len = skills.shape
        
        # Create one-hot encoding for skills
        skills_one_hot = torch.zeros(
            batch_size, seq_len, self.n_skills,
            device=skills.device
        )
        skills_one_hot.scatter_(2, skills.unsqueeze(-1), 1)
        
        # Combine with correctness
        # If correct=1, use first n_skills dimensions
        # If correct=0, use second n_skills dimensions
        input_encoding = torch.zeros(
            batch_size, seq_len, 2 * self.n_skills,
            device=skills.device
        )
        
        for b in range(batch_size):
            for t in range(seq_len):
                skill_idx = int(skills[b, t].item())
                if skill_idx >= 0:
                    if corrects[b, t].item() == 1:
                        input_encoding[b, t, skill_idx] = 1
                    else:
                        input_encoding[b, t, self.n_skills + skill_idx] = 1
        
        return input_encoding
    
    def forward(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            skills: Skill indices [batch_size, seq_len]
            corrects: Correctness [batch_size, seq_len]
            mask: Mask for valid positions [batch_size, seq_len]
            
        Returns:
            Predicted probabilities [batch_size, seq_len, n_skills]
        """
        # Encode input
        x = self._encode_input(skills, corrects)  # [batch_size, seq_len, 2*n_skills]
        
        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Pass through output layer
        logits = self.fc(rnn_out)  # [batch_size, seq_len, n_skills]
        
        # Apply sigmoid for probabilities
        probs = torch.sigmoid(logits)
        
        return probs
    
    def predict_next(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor,
        next_skills: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict correctness for next skills.
        
        Args:
            skills: Previous skill indices [batch_size, seq_len]
            corrects: Previous correctness [batch_size, seq_len]
            next_skills: Next skill to predict [batch_size, seq_len]
            
        Returns:
            Predicted probabilities for next skills [batch_size, seq_len]
        """
        # Get predictions for all skills
        all_probs = self.forward(skills, corrects)  # [batch_size, seq_len, n_skills]
        
        # Gather probabilities for next skills
        batch_size, seq_len = next_skills.shape
        next_probs = torch.zeros(batch_size, seq_len, device=skills.device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                next_skill_idx = int(next_skills[b, t].item())
                if next_skill_idx >= 0 and next_skill_idx < self.n_skills:
                    next_probs[b, t] = all_probs[b, t, next_skill_idx]
        
        return next_probs


class DKTPlusModel(nn.Module):
    """
    Enhanced DKT with skill embeddings instead of one-hot encoding.
    More efficient and can capture skill similarities.
    """
    
    def __init__(
        self,
        n_skills: int,
        embedding_dim: int = 100,
        hidden_dim: int = 200,
        num_layers: int = 2,
        dropout: float = 0.3,
        model_type: str = 'lstm'
    ):
        """
        Initialize DKT+ model.
        
        Args:
            n_skills: Number of unique skills
            embedding_dim: Skill embedding dimension
            hidden_dim: Hidden dimension size
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            model_type: 'lstm' or 'gru'
        """
        super(DKTPlusModel, self).__init__()
        
        self.n_skills = n_skills
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Skill embeddings
        self.skill_embeddings = nn.Embedding(n_skills, embedding_dim)
        
        # Input is skill embedding + correctness (1 dim)
        input_dim = embedding_dim + 1
        
        # Recurrent layer
        if model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, n_skills)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.skill_embeddings.weight)
        
    def forward(
        self,
        skills: torch.Tensor,
        corrects: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            skills: Skill indices [batch_size, seq_len]
            corrects: Correctness [batch_size, seq_len]
            mask: Mask for valid positions [batch_size, seq_len]
            
        Returns:
            Predicted probabilities [batch_size, seq_len, n_skills]
        """
        # Get skill embeddings
        skill_emb = self.skill_embeddings(skills)  # [batch_size, seq_len, embedding_dim]
        
        # Concatenate with correctness
        corrects_expanded = corrects.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x = torch.cat([skill_emb, corrects_expanded], dim=-1)  # [batch_size, seq_len, embedding_dim+1]
        
        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Pass through output layer
        logits = self.fc(rnn_out)  # [batch_size, seq_len, n_skills]
        
        # Apply sigmoid
        probs = torch.sigmoid(logits)
        
        return probs
