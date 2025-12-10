"""
PyTorch dataset classes for knowledge tracing.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class KTDataset(Dataset):
    """Knowledge Tracing dataset for sequence-based models."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 200,
        stride: Optional[int] = None
    ):
        """
        Initialize KT dataset.
        
        Args:
            df: Dataframe with columns: student_idx, skill_idx, correct, position
            sequence_length: Maximum sequence length
            stride: Stride for creating sequences (default: sequence_length, no overlap)
        """
        self.df = df
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        
        # Group by student
        self.student_sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict]:
        """
        Create sequences from student interaction data.
        
        Returns:
            List of sequence dictionaries
        """
        sequences = []
        
        for student_idx, group in self.df.groupby('student_idx'):
            # Sort by position
            group = group.sort_values('position')
            
            skills = group['skill_idx'].values
            corrects = group['correct'].values
            
            # Create overlapping or non-overlapping sequences
            n_interactions = len(skills)
            
            for start_idx in range(0, n_interactions, self.stride):
                end_idx = min(start_idx + self.sequence_length, n_interactions)
                
                if end_idx - start_idx < 2:  # Need at least 2 interactions
                    continue
                
                seq_skills = skills[start_idx:end_idx]
                seq_corrects = corrects[start_idx:end_idx]
                
                sequences.append({
                    'student_idx': student_idx,
                    'skills': seq_skills,
                    'corrects': seq_corrects,
                    'length': len(seq_skills)
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.student_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary with tensors
        """
        seq = self.student_sequences[idx]
        
        # Pad sequences to max length
        skills = np.zeros(self.sequence_length, dtype=np.int64)
        corrects = np.zeros(self.sequence_length, dtype=np.int64)
        mask = np.zeros(self.sequence_length, dtype=np.float32)
        
        length = seq['length']
        skills[:length] = seq['skills']
        corrects[:length] = seq['corrects']
        mask[:length] = 1.0
        
        # For knowledge tracing, we predict the next correctness
        # Input: skills[:-1] and corrects[:-1]
        # Target: corrects[1:]
        
        return {
            'skills': torch.LongTensor(skills),
            'corrects': torch.LongTensor(corrects),
            'mask': torch.FloatTensor(mask),
            'student_idx': seq['student_idx'],
            'length': length
        }


class StaticKTDataset(Dataset):
    """Static dataset for baseline models (non-sequential)."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize static KT dataset.
        
        Args:
            df: Dataframe with student_idx, skill_idx, correct
        """
        self.df = df.reset_index(drop=True)
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single interaction.
        
        Args:
            idx: Interaction index
            
        Returns:
            Dictionary with tensors
        """
        row = self.df.iloc[idx]
        
        return {
            'student_idx': torch.LongTensor([row['student_idx']])[0],
            'skill_idx': torch.LongTensor([row['skill_idx']])[0],
            'correct': torch.FloatTensor([row['correct']])[0]
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary
    """
    skills = torch.stack([item['skills'] for item in batch])
    corrects = torch.stack([item['corrects'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    student_idx = torch.LongTensor([item['student_idx'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    
    return {
        'skills': skills,
        'corrects': corrects,
        'mask': mask,
        'student_idx': student_idx,
        'lengths': lengths
    }
