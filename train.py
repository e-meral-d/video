"""
Modified train.py for video anomaly detection training.
Implements the dual-mode prompting strategy and temporal modeling.
"""

import torch
import torch.nn.functional as F
import argparse
import yaml
import os
import random
import numpy as np
from tqdm import tqdm

from dataset import create_dataset
from utils import get_transform
from logger import get_logger
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble import VideoAnomalyCLIP_PromptLearner
from AnomalyCLIP_lib.models import build_video_anomaly_clip
import AnomalyCLIP_lib


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def temporal_consistency_loss(features, lambda_weight=0.01):
    """
    Compute temporal consistency loss to encourage smooth temporal features.
    
    Args:
        features: Temporal features of shape (B, T, D)
        lambda_weight: Weight for temporal consistency loss
    
    Returns:
        Temporal consistency loss
    """
    if features.size(1) <= 1:
        return torch.tensor(0.0, device=features.device)
    
    # Compute differences between consecutive frames
    diff = features[:, 1:] - features[:, :-1]  # (B, T-1, D)
    
    # L2 norm of differences
    temporal_loss = torch.mean(torch.norm(diff, dim=-1))
    
    return lambda_weight * temporal_loss


def train_epoch(model, prompt_learner, dataloader, optimizer, device, config, logger):
    """Train for one epoch."""
    model.eval()
    prompt_learner.train()
    
    # Loss functions
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    total_loss = 0.0
    image_loss_total = 0.0
    temporal_loss_total = 0.0
    num_batches = 0
    
    for batch_idx, items in enumerate(tqdm(dataloader, desc="Training")):
        # Move data to device
        video = items['video'].to(device)  # Shape: (B, T, C, H, W)
        labels = items['anomaly'].to(device)  # Shape: (B,)
        gt_masks = items['gt_mask'].to(device)  # Shape: (B, T, H, W)
        
        B, T, C, H, W = video.shape
        
        # Flatten video for CLIP processing: (B*T, C, H, W)
        video_flat = video.view(B * T, C, H, W)
        
        with torch.no_grad():
            # Extract visual features using CLIP
            image_features, patch_features = model.encode_image(
                video_flat, 
                config['features_list'], 
                DPAM_layer=20
            )
            
            # Clear cache after feature extraction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Reshape back to video format: (B, T, D)
            feature_dim = image_features.shape[1]
            image_features = image_features.view(B, T, feature_dim)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Apply temporal transformer
            temporal_features = model.temporal_transformer(image_features)
        
        # Generate prompts
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        
        # Encode text with learnable prompts
        text_features = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        
        # Reshape text features: (2,) -> (1, 2, D) for normal/abnormal
        # text_features has shape (2, D) where 2 represents normal/abnormal classes
        text_features = text_features.unsqueeze(0)  # (1, 2, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute image-level loss using temporal features
        # Use mean pooling across time for video-level classification
        video_features = temporal_features.mean(dim=1)  # (B, T, D) -> (B, D)
        text_probs = video_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
        text_probs = text_probs[:, 0, ...] / 0.07  # Temperature scaling
        image_loss = F.cross_entropy(text_probs.squeeze(), labels.long())
        
        # Compute pixel-level loss using patch features
        pixel_loss = 0.0
        if patch_features:
            # Process each layer's features
            for idx, patch_feature in enumerate(patch_features):
                if idx >= config['feature_map_layers'][0]:
                    # Normalize patch features
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    
                    # Reshape patch features to video format
                    BT, N_patches, D = patch_feature.shape
                    patch_feature_video = patch_feature.view(B, T, N_patches, D)
                    
                    # Apply temporal modeling to patches
                    patch_feature_flat = patch_feature_video.view(B * N_patches, T, D)
                    temporal_patches = model.temporal_transformer(patch_feature_flat)
                    temporal_patches = temporal_patches.view(B, T, N_patches, D)
                    
                    # Compute similarity for each frame
                    for t in range(T):
                        frame_patches = temporal_patches[:, t]  # (B, N_patches, D)
                        
                        # Compute similarity with text features
                        # text_features has shape (1, 2, D), so text_features[0] has shape (2, D)
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(
                            frame_patches, text_features[0]
                        )
                        
                        # Get similarity map
                        # similarity has shape (B, N_patches, N_classes) where N_classes=2
                        # We want to use the abnormal class (index 1) for anomaly detection
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(
                            similarity, config['image_size']
                        ).permute(0, 3, 1, 2)  # (B, N_classes, H, W)
                        
                        # Resize gt_mask to match similarity map
                        gt_frame = gt_masks[:, t].unsqueeze(1)  # (B, 1, H, W)
                        if gt_frame.shape[-2:] != similarity_map.shape[-2:]:
                            gt_frame = F.interpolate(
                                gt_frame.float(), 
                                size=similarity_map.shape[-2:], 
                                mode='nearest'
                            )
                        gt_frame = gt_frame.squeeze(1)  # (B, H, W)
                        
                        # Compute focal and dice losses
                        # similarity_map: (B, N_classes, H, W) where N_classes=2
                        # gt_frame: (B, H, W)
                        # For focal loss, we need to reshape to (B*H*W, N_classes)
                        similarity_map_flat = similarity_map.permute(0, 2, 3, 1).reshape(-1, similarity_map.shape[1])

                        gt_frame_flat = gt_frame.reshape(-1)
                        
                        # Ensure gt_frame_flat has the right shape for focal loss
                        # focal loss expects target to be (N,) where N = B*H*W
                        pixel_loss += loss_focal(similarity_map_flat, gt_frame_flat)
                        pixel_loss += loss_dice(similarity_map[:, 1], gt_frame)  # abnormal class
                        pixel_loss += loss_dice(similarity_map[:, 0], 1 - gt_frame)  # normal class
        
        # Temporal consistency loss
        temporal_loss = temporal_consistency_loss(
            temporal_features, 
            config.get('temporal_smooth_weight', 0.01)
        )
        
        # Total loss
        loss = (config.get('focal_weight', 4.0) * pixel_loss + 
                config.get('image_loss_weight', 0.1) * image_loss + 
                temporal_loss)
        
        # Check for invalid loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss value detected: {loss}")
            continue
        
        # Check if loss requires grad
        if not loss.requires_grad:
            print(f"Warning: Loss does not require grad: {loss}")
            continue
        
        # Backward pass
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Warning: Backward pass failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # Update statistics
        total_loss += loss.item()
        image_loss_total += image_loss.item()
        temporal_loss_total += temporal_loss.item()
        num_batches += 1
        
        # Print progress
        if batch_idx % config.get('print_freq', 50) == 0:
            logger.info(
                f'Batch {batch_idx}/{len(dataloader)}: '
                f'Loss: {loss.item():.4f}, '
                f'Image Loss: {image_loss.item():.4f}, '
                f'Temporal Loss: {temporal_loss.item():.4f}'
            )
    
    return {
        'total_loss': total_loss / num_batches,
        'image_loss': image_loss_total / num_batches, 
        'temporal_loss': temporal_loss_total / num_batches
    }


def train(config_path, args):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create logger
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    logger = get_logger(config['checkpoint_dir'])
    logger.info(f"Starting training with config: {config}")
    
    # Model parameters
    AnomalyCLIP_parameters = {
        "Prompt_length": config['n_ctx'],
        "learnabel_text_embedding_depth": config['depth'], 
        "learnabel_text_embedding_length": config['t_n_ctx']
    }
    
    # Load pre-trained CLIP and build video model
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", 
        device=device, 
        design_details=AnomalyCLIP_parameters
    )
    
    # Convert to VideoAnomalyCLIP
    video_model = build_video_anomaly_clip(
        model.state_dict(),
        design_details=AnomalyCLIP_parameters,
        temporal_layers=config['temporal_transformer_layers'],
        temporal_heads=config['temporal_heads']
    ).to(device)
    
    # Setup prompt learner
    prompt_learner = VideoAnomalyCLIP_PromptLearner(
        video_model.to("cpu"), AnomalyCLIP_parameters
    )
    prompt_learner.to(device)
    video_model.to(device)
    video_model.visual.DAPM_replace(DPAM_layer=20)
    
    # Setup data
    preprocess, target_transform = get_transform(args)
    train_dataset = create_dataset(
        root=config['dataset_path'],
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=config['dataset_name'],
        mode='train',
        clip_length=config['clip_length'],
        stride=config['stride']
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2  # 减少worker数量
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(prompt_learner.parameters()) + list(video_model.temporal_transformer.parameters()),
        lr=config['learning_rate'],
        betas=(0.5, 0.999)
    )
    
    # Training loop
    logger.info(f"Starting training for {config['epochs']} epochs")
    
    for epoch in range(config['epochs']):
        logger.info(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        
        # Train one epoch
        epoch_stats = train_epoch(
            video_model, prompt_learner, train_dataloader, 
            optimizer, device, config, logger
        )
        
        # Log epoch statistics
        logger.info(
            f'Epoch {epoch + 1} Summary: '
            f'Avg Loss: {epoch_stats["total_loss"]:.4f}, '
            f'Avg Image Loss: {epoch_stats["image_loss"]:.4f}, '
            f'Avg Temporal Loss: {epoch_stats["temporal_loss"]:.4f}'
        )
        
        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'], 
                f'epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'prompt_learner': prompt_learner.state_dict(),
                'temporal_transformer': video_model.temporal_transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch_stats': epoch_stats
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VideoAnomalyCLIP Training")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration YAML file")
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    
    # Image processing parameters (for compatibility with get_transform)
    parser.add_argument("--image_size", type=int, default=336, help="Image size")
    parser.add_argument("--features_list", type=int, nargs="+", 
                       default=[6, 12, 18, 24], help="Features used")
    
    args = parser.parse_args()
    
    train(args.config, args)