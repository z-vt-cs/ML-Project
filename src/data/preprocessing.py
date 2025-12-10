"""
Data loading and preprocessing utilities for adaptive quizzing system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle


class DataProcessor:
    """Base class for data processing."""
    
    def __init__(self, data_path: str, min_interactions: int = 3):
        """
        Initialize data processor.
        
        Args:
            data_path: Path to raw data file
            min_interactions: Minimum number of interactions per student
        """
        self.data_path = Path(data_path)
        self.min_interactions = min_interactions
        self.data = None
        self.skill_map = {}
        self.student_map = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from file."""
        raise NotImplementedError
        
    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data."""
        raise NotImplementedError
        
    def create_mappings(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Create student and skill ID mappings.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (student_map, skill_map)
        """
        unique_students = df['student_id'].unique()
        unique_skills = df['skill_id'].unique()
        
        student_map = {sid: idx for idx, sid in enumerate(unique_students)}
        skill_map = {sid: idx for idx, sid in enumerate(unique_skills)}
        
        return student_map, skill_map
    
    def filter_sparse_students(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out students with fewer than min_interactions.
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe
        """
        interaction_counts = df.groupby('student_id').size()
        valid_students = interaction_counts[interaction_counts >= self.min_interactions].index
        
        return df[df['student_id'].isin(valid_students)].copy()
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file."""
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
    def save_mappings(self, output_dir: str):
        """Save ID mappings."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'student_map.pkl', 'wb') as f:
            pickle.dump(self.student_map, f)
            
        with open(output_path / 'skill_map.pkl', 'wb') as f:
            pickle.dump(self.skill_map, f)
            
        print(f"Saved mappings to {output_path}")


def load_mappings(mapping_dir: str) -> Tuple[Dict, Dict]:
    """
    Load student and skill mappings.
    
    Args:
        mapping_dir: Directory containing mapping files
        
    Returns:
        Tuple of (student_map, skill_map)
    """
    mapping_path = Path(mapping_dir)
    
    with open(mapping_path / 'student_map.pkl', 'rb') as f:
        student_map = pickle.load(f)
        
    with open(mapping_path / 'skill_map.pkl', 'rb') as f:
        skill_map = pickle.load(f)
        
    return student_map, skill_map


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets by students.
    
    Args:
        df: Input dataframe
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(random_seed)
    
    # Split by students to avoid data leakage
    unique_students = df['student_id'].unique()
    np.random.shuffle(unique_students)
    
    n_students = len(unique_students)
    n_train = int(n_students * train_ratio)
    n_val = int(n_students * val_ratio)
    
    train_students = unique_students[:n_train]
    val_students = unique_students[n_train:n_train + n_val]
    test_students = unique_students[n_train + n_val:]
    
    train_df = df[df['student_id'].isin(train_students)].copy()
    val_df = df[df['student_id'].isin(val_students)].copy()
    test_df = df[df['student_id'].isin(test_students)].copy()
    
    print(f"Train: {len(train_df)} interactions, {len(train_students)} students")
    print(f"Val: {len(val_df)} interactions, {len(val_students)} students")
    print(f"Test: {len(test_df)} interactions, {len(test_students)} students")
    
    return train_df, val_df, test_df
