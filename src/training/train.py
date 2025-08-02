#!/usr/bin/env python3
"""
Main training script for Virtual Try-On System.
Implements proprietary training pipeline with patentable innovations.
"""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.try_on_model import VirtualTryOnModel
from data.dataset import VirtualTryOnDataset
from training.trainer import VirtualTryOnTrainer


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config: dict):
    """Setup training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Set device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config['hardware']['device'] = 'cpu'
    
    print(f"Using device: {config['hardware']['device']}")


def create_datasets(config: dict):
    """Create training and validation datasets."""
    print("Creating datasets...")
    
    # Training dataset
    train_dataset = VirtualTryOnDataset(
        data_root=config['data']['data_root'],
        pairs_file=config['data']['train_pairs_file'],
        image_size=tuple(config['data']['image_size']),
        is_training=True,
        use_augmentation=config['data']['use_augmentation'],
        pose_refinement_stages=config['data']['pose_refinement_stages']
    )
    
    # Validation dataset
    val_dataset = VirtualTryOnDataset(
        data_root=config['data']['data_root'],
        pairs_file=config['data']['test_pairs_file'],
        image_size=tuple(config['data']['image_size']),
        is_training=False,
        use_augmentation=False,  # No augmentation for validation
        pose_refinement_stages=config['data']['pose_refinement_stages']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_model(config: dict):
    """Create the virtual try-on model."""
    print("Creating model...")
    
    model = VirtualTryOnModel(
        image_size=tuple(config['data']['image_size']),
        feature_dim=config['model']['feature_dim'],
        use_occlusion_awareness=config['model']['use_occlusion_awareness'],
        use_texture_synthesis=config['model']['use_texture_synthesis'],
        use_quality_assessment=config['model']['use_quality_assessment']
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_trainer(model, train_dataset, val_dataset, config: dict):
    """Create the trainer with proprietary configurations."""
    print("Creating trainer...")
    
    # Prepare trainer config
    trainer_config = {
        'device': config['hardware']['device'],
        'batch_size': config['training']['batch_size'],
        'num_workers': config['hardware']['num_workers'],
        'pin_memory': config['hardware']['pin_memory'],
        'lr': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'max_grad_norm': config['training']['max_grad_norm'],
        'scheduler_t0': config['training']['scheduler_t0'],
        'scheduler_t_mult': config['training']['scheduler_t_mult'],
        'log_dir': config['logging']['log_dir'],
        'checkpoint_dir': config['logging']['checkpoint_dir'],
        'save_interval': config['logging']['save_interval'],
        'log_interval': config['logging']['log_interval'],
        'use_wandb': config['logging']['use_wandb'],
        'project_name': config['logging']['project_name']
    }
    
    trainer = VirtualTryOnTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Virtual Try-On Model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    # Setup environment
    setup_environment(config)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = create_trainer(model, train_dataset, val_dataset, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    trainer.train(config['training']['num_epochs'])
    
    print("Training completed!")


if __name__ == '__main__':
    main() 