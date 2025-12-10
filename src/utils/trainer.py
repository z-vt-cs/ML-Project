"""
Training utilities for knowledge tracing models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from typing import Dict, Optional, Callable, Union
import numpy as np
from tqdm import tqdm
from pathlib import Path


class Trainer:
    """Trainer for knowledge tracing models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        grad_clip: Optional[float] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            grad_clip: Gradient clipping value
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        adjacency: Optional[torch.Tensor] = None
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            adjacency: Adjacency matrix for graph models
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch in pbar:
            # Move to device
            skills = batch['skills'].to(self.device)
            corrects = batch['corrects'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if adjacency is not None:
                # Graph-enhanced model
                adj = adjacency.to(self.device)
                predictions = self.model(skills, corrects, adj, mask)
            else:
                # Standard sequential model
                predictions = self.model(skills, corrects, mask)
            
            # Compute loss
            # Predict next correctness
            # Input: skills[:-1], corrects[:-1]
            # Target: corrects[1:]
            if len(predictions.shape) == 3:
                # Shape: [batch_size, seq_len, n_skills]
                # Need to get predictions for next skills
                batch_size, seq_len = skills.shape
                targets = corrects[:, 1:].float()
                pred_next = predictions[:, :-1, :]
                
                # Get predictions for actual next skills
                loss = 0
                for b in range(batch_size):
                    for t in range(seq_len - 1):
                        if mask[b, t+1] > 0:
                            next_skill = skills[b, t+1]
                            pred_prob = pred_next[b, t, next_skill]
                            target = targets[b, t]
                            loss += self.criterion(pred_prob.unsqueeze(0), target.unsqueeze(0))
                
                loss = loss / (mask[:, 1:].sum() + 1e-8)
            else:
                # Shape: [batch_size, seq_len]
                targets = corrects[:, 1:].float()
                pred_next = predictions[:, :-1]
                mask_next = mask[:, 1:]
                
                loss = self.criterion(pred_next * mask_next, targets * mask_next)
                loss = loss.sum() / (mask_next.sum() + 1e-8)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': total_loss / n_batches})
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader,
        adjacency: Optional[torch.Tensor] = None
    ) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            adjacency: Adjacency matrix for graph models
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                skills = batch['skills'].to(self.device)
                corrects = batch['corrects'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                if adjacency is not None:
                    adj = adjacency.to(self.device)
                    predictions = self.model(skills, corrects, adj, mask)
                else:
                    predictions = self.model(skills, corrects, mask)
                
                # Compute loss (same as training)
                if len(predictions.shape) == 3:
                    batch_size, seq_len = skills.shape
                    targets = corrects[:, 1:].float()
                    pred_next = predictions[:, :-1, :]
                    
                    loss = 0
                    for b in range(batch_size):
                        for t in range(seq_len - 1):
                            if mask[b, t+1] > 0:
                                next_skill = skills[b, t+1]
                                pred_prob = pred_next[b, t, next_skill]
                                target = targets[b, t]
                                loss += self.criterion(pred_prob.unsqueeze(0), target.unsqueeze(0))
                    
                    loss = loss / (mask[:, 1:].sum() + 1e-8)
                else:
                    targets = corrects[:, 1:].float()
                    pred_next = predictions[:, :-1]
                    mask_next = mask[:, 1:]
                    
                    loss = self.criterion(pred_next * mask_next, targets * mask_next)
                    loss = loss.sum() / (mask_next.sum() + 1e-8)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str,
        adjacency: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 10
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            adjacency: Adjacency matrix for graph models
            early_stopping_patience: Patience for early stopping
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, adjacency)
            
            # Validate
            val_loss = self.validate(val_loader, adjacency)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, save_path / 'best_model.pth')
                print(f"Saved best model with val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")


class BaselineTrainer:
    """Trainer for baseline models (sklearn-based or simple PyTorch)."""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize baseline trainer.
        
        Args:
            model: Model to train
            device: Device to use
        """
        self.model = model
        self.device = device
        
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.01
    ):
        """
        Train baseline model.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
        """
        # Check if PyTorch model
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()
            
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                
                for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                    student_idx = batch['student_idx'].to(self.device)
                    skill_idx = batch['skill_idx'].to(self.device)
                    correct = batch['correct'].to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(student_idx, skill_idx)
                    loss = criterion(predictions, correct)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        else:
            # Sklearn model
            # Collect all data
            student_indices = []
            skill_indices = []
            corrects = []
            
            for batch in train_loader:
                student_indices.append(batch['student_idx'].numpy())
                skill_indices.append(batch['skill_idx'].numpy())
                corrects.append(batch['correct'].numpy())
            
            student_indices = np.concatenate(student_indices)
            skill_indices = np.concatenate(skill_indices)
            corrects = np.concatenate(corrects)
            
            # Train
            self.model.fit(student_indices, skill_indices, corrects)
