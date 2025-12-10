"""
Evaluation metrics for knowledge tracing.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error
)
from typing import Dict, List, Tuple, Optional


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets, pred_binary),
        'auc': roc_auc_score(targets, predictions),
        'precision': precision_score(targets, pred_binary, zero_division=0),
        'recall': recall_score(targets, pred_binary, zero_division=0),
        'f1': f1_score(targets, pred_binary, zero_division=0),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions)
    }
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    adjacency: Optional[torch.Tensor] = None
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        adjacency: Adjacency matrix for graph models
        
    Returns:
        Tuple of (metrics dict, predictions array, targets array)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            skills = batch['skills'].to(device)
            corrects = batch['corrects'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            if adjacency is not None:
                adj = adjacency.to(device)
                predictions = model(skills, corrects, adj, mask)
            else:
                predictions = model(skills, corrects, mask)
            
            # Extract predictions for next steps
            if len(predictions.shape) == 3:
                # [batch_size, seq_len, n_skills]
                batch_size, seq_len = skills.shape
                
                for b in range(batch_size):
                    for t in range(seq_len - 1):
                        if mask[b, t+1] > 0:
                            next_skill = skills[b, t+1]
                            pred_prob = predictions[b, t, next_skill].item()
                            target = corrects[b, t+1].item()
                            
                            all_predictions.append(pred_prob)
                            all_targets.append(target)
            else:
                # [batch_size, seq_len]
                pred_next = predictions[:, :-1]
                targets = corrects[:, 1:]
                mask_next = mask[:, 1:]
                
                # Flatten and filter by mask
                pred_flat = pred_next.cpu().numpy().flatten()
                target_flat = targets.cpu().numpy().flatten()
                mask_flat = mask_next.cpu().numpy().flatten()
                
                valid_indices = mask_flat > 0
                all_predictions.extend(pred_flat[valid_indices].tolist())
                all_targets.extend(target_flat[valid_indices].tolist())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    return metrics, predictions, targets


def evaluate_by_skill(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    adjacency: Optional[torch.Tensor] = None
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model performance per skill.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        adjacency: Adjacency matrix for graph models
        
    Returns:
        Dictionary mapping skill_id to metrics
    """
    model.eval()
    
    skill_predictions = {}  # skill_id -> list of (prediction, target)
    
    with torch.no_grad():
        for batch in data_loader:
            skills = batch['skills'].to(device)
            corrects = batch['corrects'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            if adjacency is not None:
                adj = adjacency.to(device)
                predictions = model(skills, corrects, adj, mask)
            else:
                predictions = model(skills, corrects, mask)
            
            # Extract per-skill predictions
            if len(predictions.shape) == 3:
                batch_size, seq_len = skills.shape
                
                for b in range(batch_size):
                    for t in range(seq_len - 1):
                        if mask[b, t+1] > 0:
                            next_skill = skills[b, t+1].item()
                            pred_prob = predictions[b, t, next_skill].item()
                            target = corrects[b, t+1].item()
                            
                            if next_skill not in skill_predictions:
                                skill_predictions[next_skill] = {'preds': [], 'targets': []}
                            
                            skill_predictions[next_skill]['preds'].append(pred_prob)
                            skill_predictions[next_skill]['targets'].append(target)
    
    # Compute metrics per skill
    skill_metrics = {}
    
    for skill_id, data in skill_predictions.items():
        preds = np.array(data['preds'])
        targets = np.array(data['targets'])
        
        if len(preds) >= 10:  # Only compute if sufficient samples
            skill_metrics[skill_id] = compute_metrics(preds, targets)
            skill_metrics[skill_id]['n_samples'] = len(preds)
    
    return skill_metrics


def evaluate_by_student(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    adjacency: Optional[torch.Tensor] = None
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model performance per student.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        adjacency: Adjacency matrix for graph models
        
    Returns:
        Dictionary mapping student_id to metrics
    """
    model.eval()
    
    student_predictions = {}  # student_id -> list of (prediction, target)
    
    with torch.no_grad():
        for batch in data_loader:
            skills = batch['skills'].to(device)
            corrects = batch['corrects'].to(device)
            mask = batch['mask'].to(device)
            student_idx = batch['student_idx'].cpu().numpy()
            
            # Forward pass
            if adjacency is not None:
                adj = adjacency.to(device)
                predictions = model(skills, corrects, adj, mask)
            else:
                predictions = model(skills, corrects, mask)
            
            # Extract per-student predictions
            if len(predictions.shape) == 3:
                batch_size, seq_len = skills.shape
                
                for b in range(batch_size):
                    student_id = student_idx[b]
                    
                    if student_id not in student_predictions:
                        student_predictions[student_id] = {'preds': [], 'targets': []}
                    
                    for t in range(seq_len - 1):
                        if mask[b, t+1] > 0:
                            next_skill = skills[b, t+1]
                            pred_prob = predictions[b, t, next_skill].item()
                            target = corrects[b, t+1].item()
                            
                            student_predictions[student_id]['preds'].append(pred_prob)
                            student_predictions[student_id]['targets'].append(target)
    
    # Compute metrics per student
    student_metrics = {}
    
    for student_id, data in student_predictions.items():
        preds = np.array(data['preds'])
        targets = np.array(data['targets'])
        
        if len(preds) >= 5:  # Only compute if sufficient samples
            student_metrics[student_id] = compute_metrics(preds, targets)
            student_metrics[student_id]['n_samples'] = len(preds)
    
    return student_metrics


def retention_prediction_error(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    time_gaps: List[int],
    device: str = 'cuda'
) -> Dict[int, float]:
    """
    Compute retention prediction error at different time gaps.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        time_gaps: List of time gaps to evaluate (in number of interactions)
        device: Device to use
        
    Returns:
        Dictionary mapping time_gap to RMSE
    """
    # This is a simplified version
    # In practice, you'd need temporal information in the data
    
    errors = {gap: [] for gap in time_gaps}
    
    # Implementation depends on dataset structure with temporal information
    # Placeholder for now
    
    return {gap: float(np.mean(errs)) if errs else 0.0 for gap, errs in errors.items()}
