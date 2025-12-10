"""
Training script for knowledge tracing models.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    ASSISTmentsProcessor,
    KTDataset,
    StaticKTDataset,
    train_val_test_split,
    collate_fn
)
from src.models import (
    DKTModel,
    DKTPlusModel
)
from src.utils import Trainer, BaselineTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_dkt(config: dict):
    """Train DKT model."""
    print("Training DKT model...")
    
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
    
    # Create datasets
    train_dataset = KTDataset(train_df, config['data']['sequence_length'])
    val_dataset = KTDataset(val_df, config['data']['sequence_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    n_skills = len(processor.skill_map)
    
    model = DKTPlusModel(
        n_skills=n_skills,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        model_type=config['model']['model_type']
    )
    
    # Optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.BCELoss(reduction='none')
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience']
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config['training']['device'],
        scheduler=scheduler,
        grad_clip=config['training']['grad_clip']
    )
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=config['training']['save_dir'],
        early_stopping_patience=config['early_stopping']['patience']
    )


def main():
    parser = argparse.ArgumentParser(description='Train knowledge tracing model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['logistic_regression', 'dkt'],
                        help='Model to train')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train model
    if args.model == 'logistic_regression':
        print("Error: Use scripts/train_logistic_regression.py for logistic regression training")
    elif args.model == 'dkt':
        train_dkt(config)


if __name__ == '__main__':
    main()
