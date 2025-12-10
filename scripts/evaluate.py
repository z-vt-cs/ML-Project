"""
Evaluation script for trained models.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import joblib

sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    ASSISTmentsProcessor,
    KTDataset,
    train_val_test_split,
    collate_fn,
    load_mappings
)
from src.models import DKTPlusModel, GraphDKT
from src.utils import evaluate_model, evaluate_by_skill, evaluate_by_student, compute_metrics
from src.graph import KnowledgeGraphBuilder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_logistic_regression(model_path: str, config: dict, split: str = 'test'):
    """Evaluate logistic regression baseline model."""
    # Load model and encoder
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    encoder = checkpoint['encoder']
    
    # Load data
    processor = ASSISTmentsProcessor(config['data']['data_path'])
    df = processor.preprocess()
    train_df, val_df, test_df = train_val_test_split(
        df,
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split'],
        config['data']['random_seed']
    )
    
    # Select split
    if split == 'train':
        eval_df = train_df
    elif split == 'val':
        eval_df = val_df
    else:
        eval_df = test_df
    
    # Prepare features
    from sklearn.preprocessing import OneHotEncoder
    X_eval = encoder.transform(eval_df[['student_idx', 'skill_idx']])
    y_eval = eval_df['correct'].values
    
    # Get predictions
    predictions = model.predict_proba(X_eval)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(predictions, np.array(y_eval))
    
    return metrics, predictions, y_eval


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--model_type', type=str, default='dkt',
                        choices=['logistic', 'dkt', 'graph_dkt'],
                        help='Type of model to evaluate')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if logistic regression
    if args.model_type == 'logistic' or args.model_path.endswith('.joblib'):
        print(f"Evaluating Logistic Regression on {args.split} set...")
        metrics, predictions, targets = evaluate_logistic_regression(
            args.model_path, config, args.split
        )
        
        print("\n=== Overall Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save predictions if requested
        if config['evaluation'].get('save_predictions', False):
            pred_df = pd.DataFrame({
                'prediction': predictions,
                'target': targets
            })
            pred_path = config['evaluation']['prediction_path']
            pred_df.to_csv(pred_path, index=False)
            print(f"\nSaved predictions to {pred_path}")
        
        return
    
    # Load data
    processor = ASSISTmentsProcessor(config['data']['data_path'])
    
    df = processor.preprocess()
    train_df, val_df, test_df = train_val_test_split(
        df,
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split'],
        config['data']['random_seed']
    )
    
    # Select split
    if args.split == 'train':
        eval_df = train_df
    elif args.split == 'val':
        eval_df = val_df
    else:
        eval_df = test_df
    
    # Create dataset
    eval_dataset = KTDataset(eval_df, config['data']['sequence_length'])
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    n_skills = len(processor.skill_map)
    device = config['training']['device']
    
    # Determine model type from config
    if 'gnn' in config:
        # Graph-enhanced model
        model = GraphDKT(
            n_skills=n_skills,
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            graph_hidden_dim=config['gnn']['hidden_dim'],
            num_lstm_layers=config['model']['num_layers'],
            num_gnn_layers=config['gnn']['num_layers'],
            dropout=config['model']['dropout'],
            gnn_type=config['gnn']['type'],
            fusion_method=config['model']['fusion_method']
        )
        
        # Load graph
        graph_builder = KnowledgeGraphBuilder.load_graph(config['graph']['graph_path'])
        adjacency = torch.FloatTensor(graph_builder.adjacency_matrix)
    else:
        # Standard DKT
        model = DKTPlusModel(
            n_skills=n_skills,
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            model_type=config['model']['model_type']
        )
        adjacency = None
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Evaluate
    print(f"Evaluating on {args.split} set...")
    metrics, predictions, targets = evaluate_model(
        model, eval_loader, device, adjacency if adjacency is not None else None  # type: ignore
    )
    
    print("\n=== Overall Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Per-skill analysis
    if config['evaluation'].get('analyze_by_concept', False):
        print("\n=== Per-Skill Analysis ===")
        skill_metrics = evaluate_by_skill(model, eval_loader, device, adjacency if adjacency is not None else None)  # type: ignore
        
        # Sort by performance
        sorted_skills = sorted(
            skill_metrics.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"\nTop 5 skills (by accuracy):")
        for skill_id, metrics in sorted_skills[:5]:
            print(f"Skill {skill_id}: Acc={metrics['accuracy']:.4f}, "
                  f"AUC={metrics['auc']:.4f}, N={metrics['n_samples']}")
        
        print(f"\nBottom 5 skills (by accuracy):")
        for skill_id, metrics in sorted_skills[-5:]:
            print(f"Skill {skill_id}: Acc={metrics['accuracy']:.4f}, "
                  f"AUC={metrics['auc']:.4f}, N={metrics['n_samples']}")
    
    # Save predictions
    if config['evaluation'].get('save_predictions', False):
        pred_df = pd.DataFrame({
            'prediction': predictions,
            'target': targets
        })
        pred_path = config['evaluation']['prediction_path']
        pred_df.to_csv(pred_path, index=False)
        print(f"\nSaved predictions to {pred_path}")


if __name__ == '__main__':
    main()
