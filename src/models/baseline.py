"""
Baseline models for knowledge tracing.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from typing import Dict, Tuple


class LogisticRegression:
    """Logistic Regression baseline for knowledge tracing."""
    
    def __init__(
        self,
        n_students: int,
        n_skills: int,
        max_iter: int = 1000,
        C: float = 1.0
    ):
        """
        Initialize logistic regression model.
        
        Args:
            n_students: Number of unique students
            n_skills: Number of unique skills
            max_iter: Maximum iterations for optimization
            C: Regularization strength
        """
        self.n_students = n_students
        self.n_skills = n_skills
        self.max_iter = max_iter
        self.C = C
        self.model = None
        
    def prepare_features(
        self,
        student_idx: np.ndarray,
        skill_idx: np.ndarray
    ) -> np.ndarray:
        """
        Prepare features using one-hot encoding.
        
        Args:
            student_idx: Student indices
            skill_idx: Skill indices
            
        Returns:
            Feature matrix
        """
        n_samples = len(student_idx)
        
        # One-hot encode students and skills
        features = np.zeros((n_samples, self.n_students + self.n_skills))
        
        for i in range(n_samples):
            features[i, student_idx[i]] = 1  # Student one-hot
            features[i, self.n_students + skill_idx[i]] = 1  # Skill one-hot
        
        return features
    
    def fit(self, student_idx: np.ndarray, skill_idx: np.ndarray, correct: np.ndarray):
        """
        Fit the model.
        
        Args:
            student_idx: Student indices
            skill_idx: Skill indices
            correct: Binary correctness labels
        """
        X = self.prepare_features(student_idx, skill_idx)
        y = correct
        
        self.model = SKLogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X, y)
        
    def predict_proba(
        self,
        student_idx: np.ndarray,
        skill_idx: np.ndarray
    ) -> np.ndarray:
        """
        Predict probability of correctness.
        
        Args:
            student_idx: Student indices
            skill_idx: Skill indices
            
        Returns:
            Predicted probabilities
        """
        X = self.prepare_features(student_idx, skill_idx)
        return self.model.predict_proba(X)[:, 1]  # type: ignore
    
    def predict(self, student_idx: np.ndarray, skill_idx: np.ndarray) -> np.ndarray:
        """
        Predict binary correctness.
        
        Args:
            student_idx: Student indices
            skill_idx: Skill indices
            
        Returns:
            Binary predictions
        """
        X = self.prepare_features(student_idx, skill_idx)
        return self.model.predict(X)  # type: ignore

