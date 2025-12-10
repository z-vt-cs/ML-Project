"""
Dataset loader for ASSISTments dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .preprocessing import DataProcessor


class ASSISTmentsProcessor(DataProcessor):
    """Data processor for ASSISTments dataset."""
    
    def load_data(self) -> pd.DataFrame:
        """
        Load ASSISTments data from CSV.
        
        Expected columns:
        - order_id or row_id: Temporal ordering
        - user_id: Student identifier
        - problem_id or skill_id: Question/skill identifier
        - correct: Binary correctness (0 or 1)
        - ms_first_response (optional): Response time
        """
        df = pd.read_csv(self.data_path)
        
        # Standardize column names
        column_mapping = {
            'user_id': 'student_id',
            'problem_id': 'skill_id',
            'sequence_id': 'skill_id',  # ASSISTments 2015 uses sequence_id as skill
            'order_id': 'timestamp',
            'row_id': 'timestamp',
            'log_id': 'timestamp',  # ASSISTments 2015 uses log_id for ordering
            'ms_first_response': 'response_time'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['student_id', 'skill_id', 'correct']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.data = df
        return df
    
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess ASSISTments data.
        
        Returns:
            Preprocessed dataframe
        """
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # Sort by student and timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values(['student_id', 'timestamp'])
        else:
            # If no timestamp, group by student and assume order
            df = df.sort_values('student_id')
            df['timestamp'] = df.groupby('student_id').cumcount()
        
        # Filter sparse students
        df = self.filter_sparse_students(df)
        
        # Create mappings
        self.student_map, self.skill_map = self.create_mappings(df)
        
        # Apply mappings
        df['student_idx'] = df['student_id'].map(self.student_map)
        df['skill_idx'] = df['skill_id'].map(self.skill_map)
        
        # Ensure correct is binary
        df['correct'] = df['correct'].astype(int)
        
        # Add sequence position within each student
        df['position'] = df.groupby('student_id').cumcount()
        
        print(f"Processed data: {len(df)} interactions")
        print(f"Students: {len(self.student_map)}")
        print(f"Skills: {len(self.skill_map)}")
        print(f"Average accuracy: {df['correct'].mean():.3f}")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get dataset statistics.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_interactions': len(df),
            'n_students': df['student_id'].nunique(),
            'n_skills': df['skill_id'].nunique(),
            'avg_accuracy': df['correct'].mean(),
            'avg_interactions_per_student': df.groupby('student_id').size().mean(),
            'median_interactions_per_student': df.groupby('student_id').size().median(),
            'avg_attempts_per_skill': df.groupby('skill_id').size().mean(),
        }
        
        return stats
