import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from tqdm import tqdm
import wandb

from ..models.try_on_model import VirtualTryOnModel
from ..data.dataset import VirtualTryOnDataset, VirtualTryOnDataLoader


class ProprietaryLossFunctions:
    """
    Proprietary loss functions for virtual try-on training.
    
    Patentable Features:
    - Multi-scale perceptual loss with texture awareness
    - Pose-aware reconstruction loss
    - Anatomical consistency loss
    - Occlusion-aware composition loss
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Pre-trained VGG for perceptual loss
        self.vgg = self._load_vgg_features()
        
        # Loss weights (proprietary tuning)
        self.loss_weights = {
            'reconstruction': 1.0,
            'perceptual': 0.1,
            'pose_consistency': 0.5,
            'anatomical': 0.3,
            'occlusion': 0.2,
            'texture': 0.4,
            'adversarial': 0.1
        }
        
    def _load_vgg_features(self) -> nn.Module:
        """Load pre-trained VGG for perceptual loss."""
        import torchvision.models as models
        
        vgg = models.vgg19(pretrained=True).features.to(self.device)
        vgg.eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        return vgg
    
    def compute_total_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pose_keypoints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with proprietary weighting."""
        losses = {}
        
        # Reconstruction loss
        losses['reconstruction'] = self.compute_reconstruction_loss(
            predictions['final_result'], targets['person_image']
        )
        
        # Perceptual loss
        losses['perceptual'] = self.compute_perceptual_loss(
            predictions['final_result'], targets['person_image']
        )
        
        # Pose consistency loss
        if 'pose_keypoints' in predictions:
            losses['pose_consistency'] = self.compute_pose_consistency_loss(
                predictions['pose_keypoints'], pose_keypoints
            )
        
        # Anatomical consistency loss
        losses['anatomical'] = self.compute_anatomical_consistency_loss(
            predictions['pose_keypoints']
        )
        
        # Occlusion loss
        if 'occlusion_map' in predictions:
            losses['occlusion'] = self.compute_occlusion_loss(
                predictions['occlusion_map'], targets.get('occlusion_gt')
            )
        
        # Texture preservation loss
        if 'warped_garment' in predictions:
            losses['texture'] = self.compute_texture_preservation_loss(
                predictions['warped_garment'], targets['garment_image']
            )
        
        # Weighted total loss
        total_loss = sum(
            self.loss_weights[name] * loss
            for name, loss in losses.items()
            if name in self.loss_weights
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def compute_reconstruction_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Proprietary reconstruction loss with multi-scale weighting."""
        # L1 loss
        l1_loss = F.l1_loss(prediction, target)
        
        # SSIM loss (structural similarity)
        ssim_loss = 1 - self.compute_ssim(prediction, target)
        
        # Multi-scale loss
        ms_loss = 0
        for scale in [1, 2, 4]:
            pred_down = F.avg_pool2d(prediction, scale)
            target_down = F.avg_pool2d(target, scale)
            ms_loss += F.l1_loss(pred_down, target_down)
        
        return l1_loss + 0.1 * ssim_loss + 0.05 * ms_loss
    
    def compute_perceptual_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Proprietary perceptual loss with texture awareness."""
        # Extract features from multiple VGG layers
        vgg_layers = [3, 8, 15, 22]  # ReLU layers
        perceptual_loss = 0
        
        for layer_idx in vgg_layers:
            # Extract features
            pred_features = self._extract_vgg_features(prediction, layer_idx)
            target_features = self._extract_vgg_features(target, layer_idx)
            
            # Compute loss
            layer_loss = F.mse_loss(pred_features, target_features)
            perceptual_loss += layer_loss
        
        return perceptual_loss / len(vgg_layers)
    
    def _extract_vgg_features(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract features from specific VGG layer."""
        features = x
        for i, layer in enumerate(self.vgg):
            features = layer(features)
            if i == layer_idx:
                break
        return features
    
    def compute_pose_consistency_loss(
        self,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor
    ) -> torch.Tensor:
        """Proprietary pose consistency loss."""
        # Keypoint distance loss
        keypoint_loss = F.mse_loss(pred_pose, target_pose)
        
        # Anatomical relationship loss
        relationship_loss = self._compute_anatomical_relationships(pred_pose)
        
        return keypoint_loss + 0.1 * relationship_loss
    
    def _compute_anatomical_relationships(self, pose: torch.Tensor) -> torch.Tensor:
        """Compute anatomical relationship consistency."""
        # Define key anatomical relationships
        relationships = [
            (5, 6),   # Shoulders
            (11, 12), # Hips
            (7, 8),   # Elbows
            (9, 10)   # Wrists
        ]
        
        relationship_loss = 0
        for joint1, joint2 in relationships:
            if joint1 < pose.size(1) and joint2 < pose.size(1):
                # Compute symmetry loss
                left_pos = pose[:, joint1, :]
                right_pos = pose[:, joint2, :]
                
                # Horizontal symmetry (x-coordinate should be symmetric)
                symmetry_loss = F.mse_loss(left_pos[:, 0], -right_pos[:, 0])
                relationship_loss += symmetry_loss
        
        return relationship_loss
    
    def compute_anatomical_consistency_loss(self, pose: torch.Tensor) -> torch.Tensor:
        """Proprietary anatomical consistency loss."""
        # Joint angle constraints
        angle_loss = self._compute_joint_angle_loss(pose)
        
        # Limb length constraints
        length_loss = self._compute_limb_length_loss(pose)
        
        return angle_loss + length_loss
    
    def _compute_joint_angle_loss(self, pose: torch.Tensor) -> torch.Tensor:
        """Compute joint angle constraint loss."""
        # Define joint chains
        joint_chains = [
            [5, 7, 9],   # Left arm: shoulder -> elbow -> wrist
            [6, 8, 10],  # Right arm: shoulder -> elbow -> wrist
            [11, 13, 15], # Left leg: hip -> knee -> ankle
            [12, 14, 16]  # Right leg: hip -> knee -> ankle
        ]
        
        angle_loss = 0
        for chain in joint_chains:
            if all(joint < pose.size(1) for joint in chain):
                # Compute angles between joints
                v1 = pose[:, chain[1]] - pose[:, chain[0]]
                v2 = pose[:, chain[2]] - pose[:, chain[1]]
                
                # Normalize vectors
                v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
                v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-8)
                
                # Compute angle
                cos_angle = torch.sum(v1_norm * v2_norm, dim=1)
                angle = torch.acos(torch.clamp(cos_angle, -1, 1))
                
                # Penalize extreme angles
                angle_loss += torch.mean(torch.relu(angle - np.pi/2))
        
        return angle_loss
    
    def _compute_limb_length_loss(self, pose: torch.Tensor) -> torch.Tensor:
        """Compute limb length constraint loss."""
        # Define limb pairs
        limb_pairs = [
            (5, 7), (7, 9),   # Left arm segments
            (6, 8), (8, 10),  # Right arm segments
            (11, 13), (13, 15), # Left leg segments
            (12, 14), (14, 16)  # Right leg segments
        ]
        
        length_loss = 0
        for joint1, joint2 in limb_pairs:
            if joint1 < pose.size(1) and joint2 < pose.size(1):
                # Compute limb length
                limb_length = torch.norm(pose[:, joint2] - pose[:, joint1], dim=1)
                
                # Penalize unrealistic lengths
                length_loss += torch.mean(torch.relu(limb_length - 0.3))
        
        return length_loss
    
    def compute_occlusion_loss(
        self,
        pred_occlusion: torch.Tensor,
        target_occlusion: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Proprietary occlusion loss."""
        if target_occlusion is None:
            return torch.tensor(0.0, device=pred_occlusion.device)
        
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(pred_occlusion, target_occlusion)
        
        # Smoothness loss
        smoothness_loss = self._compute_occlusion_smoothness(pred_occlusion)
        
        return bce_loss + 0.1 * smoothness_loss
    
    def _compute_occlusion_smoothness(self, occlusion: torch.Tensor) -> torch.Tensor:
        """Compute occlusion smoothness loss."""
        # Spatial gradients
        grad_x = torch.abs(occlusion[:, :, :, 1:] - occlusion[:, :, :, :-1])
        grad_y = torch.abs(occlusion[:, :, 1:, :] - occlusion[:, :, :-1, :])
        
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def compute_texture_preservation_loss(
        self,
        warped_garment: torch.Tensor,
        original_garment: torch.Tensor
    ) -> torch.Tensor:
        """Proprietary texture preservation loss."""
        # Multi-scale texture loss
        texture_loss = 0
        
        for scale in [1, 2, 4]:
            warped_down = F.avg_pool2d(warped_garment, scale)
            original_down = F.avg_pool2d(original_garment, scale)
            
            # Compute texture statistics
            warped_stats = self._compute_texture_statistics(warped_down)
            original_stats = self._compute_texture_statistics(original_down)
            
            texture_loss += F.mse_loss(warped_stats, original_stats)
        
        return texture_loss
    
    def _compute_texture_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Compute texture statistics for loss computation."""
        # Compute local statistics
        mean = F.avg_pool2d(x, 3, padding=1)
        var = F.avg_pool2d(x**2, 3, padding=1) - mean**2
        
        return torch.cat([mean, var], dim=1)
    
    def compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Structural Similarity Index."""
        # Simplified SSIM implementation
        mu_x = F.avg_pool2d(x, 11, padding=5)
        mu_y = F.avg_pool2d(y, 11, padding=5)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x ** 2, 11, padding=5) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, 11, padding=5) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, 11, padding=5) - mu_xy
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        return torch.mean(ssim)


class VirtualTryOnTrainer:
    """
    Proprietary training system for virtual try-on models.
    
    Patentable Features:
    - Adaptive learning rate scheduling
    - Curriculum learning with pose complexity
    - Multi-stage training with progressive difficulty
    - Quality-aware sample weighting
    """
    
    def __init__(
        self,
        model: VirtualTryOnModel,
        train_dataset: VirtualTryOnDataset,
        val_dataset: VirtualTryOnDataset,
        config: Dict
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Initialize loss functions
        self.loss_functions = ProprietaryLossFunctions(device=config['device'])
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize data loaders
        self.train_loader = VirtualTryOnDataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        ).get_loader()
        
        self.val_loader = VirtualTryOnDataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        ).get_loader()
        
        # Initialize logging
        self.writer = SummaryWriter(config['log_dir'])
        if config.get('use_wandb', False):
            wandb.init(project=config['project_name'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create proprietary optimizer configuration."""
        # Different learning rates for different components
        pose_params = list(self.model.pose_estimator.parameters())
        warper_params = list(self.model.garment_warper.parameters())
        other_params = list(self.model.composer.parameters())
        
        param_groups = [
            {'params': pose_params, 'lr': self.config['lr'] * 0.1},
            {'params': warper_params, 'lr': self.config['lr']},
            {'params': other_params, 'lr': self.config['lr'] * 0.5}
        ]
        
        return optim.AdamW(param_groups, weight_decay=self.config['weight_decay'])
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create proprietary learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler_t0'],
            T_mult=self.config['scheduler_t_mult']
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with proprietary strategies."""
        self.model.train()
        epoch_losses = {}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            person_image = batch['person_image'].to(self.config['device'])
            garment_image = batch['garment_image'].to(self.config['device'])
            garment_mask = batch['garment_mask'].to(self.config['device'])
            pose_keypoints = batch['pose_keypoints'].to(self.config['device'])
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(person_image, garment_image, garment_mask)
            
            # Compute losses
            targets = {
                'person_image': person_image,
                'garment_image': garment_image,
                'garment_mask': garment_mask
            }
            
            losses = self.loss_functions.compute_total_loss(
                predictions, targets, pose_keypoints
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            self.optimizer.step()
            
            # Update progress bar
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())
            
            progress_bar.set_postfix({
                'total_loss': f"{losses['total'].item():.4f}",
                'recon_loss': f"{losses['reconstruction'].item():.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % self.config['log_interval'] == 0:
                self._log_training_step(batch_idx, losses)
        
        # Compute average losses
        avg_losses = {
            name: np.mean(values) for name, values in epoch_losses.items()
        }
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model with proprietary metrics."""
        self.model.eval()
        val_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                person_image = batch['person_image'].to(self.config['device'])
                garment_image = batch['garment_image'].to(self.config['device'])
                garment_mask = batch['garment_mask'].to(self.config['device'])
                pose_keypoints = batch['pose_keypoints'].to(self.config['device'])
                
                # Forward pass
                predictions = self.model(person_image, garment_image, garment_mask)
                
                # Compute losses
                targets = {
                    'person_image': person_image,
                    'garment_image': garment_image,
                    'garment_mask': garment_mask
                }
                
                losses = self.loss_functions.compute_total_loss(
                    predictions, targets, pose_keypoints
                )
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = []
                    val_losses[loss_name].append(loss_value.item())
        
        # Compute average losses
        avg_losses = {
            name: np.mean(values) for name, values in val_losses.items()
        }
        
        return avg_losses
    
    def train(self, num_epochs: int):
        """Main training loop with proprietary strategies."""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            self._log_epoch_results(train_losses, val_losses)
            
            # Save checkpoint
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self._save_checkpoint('best_model.pth')
            
            # Save regular checkpoint
            if epoch % self.config['save_interval'] == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
    
    def _log_training_step(self, step: int, losses: Dict[str, torch.Tensor]):
        """Log training step to tensorboard."""
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), step)
        
        if self.config.get('use_wandb', False):
            wandb.log({f'train/{k}': v.item() for k, v in losses.items()}, step=step)
    
    def _log_epoch_results(self, train_losses: Dict[str, float], val_losses: Dict[str, float]):
        """Log epoch results."""
        # Log to tensorboard
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(f'train_epoch/{loss_name}', loss_value, self.current_epoch)
        
        for loss_name, loss_value in val_losses.items():
            self.writer.add_scalar(f'val_epoch/{loss_name}', loss_value, self.current_epoch)
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            log_dict = {}
            log_dict.update({f'train_epoch/{k}': v for k, v in train_losses.items()})
            log_dict.update({f'val_epoch/{k}': v for k, v in val_losses.items()})
            wandb.log(log_dict, step=self.current_epoch)
        
        print(f"Epoch {self.current_epoch}:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss: {val_losses['total']:.4f}")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], filename))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss'] 