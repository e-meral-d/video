"""
Modified test.py for video anomaly detection testing.
Implements sliding window inference and semantic prompt evaluation.
"""

import torch
import torch.nn.functional as F
import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import random
from dataset import create_dataset
from utils import get_transform
from logger import get_logger
from prompt_ensemble import VideoAnomalyCLIP_PromptLearner
from AnomalyCLIP_lib.models import build_video_anomaly_clip
from metrics import image_level_metrics, pixel_level_metrics
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


def calculate_anomaly_score(visual_features, text_features):
    """
    Calculate anomaly scores using the new scoring function.
    
    Args:
        visual_features: Visual features of shape (B, T, D) or (B*T, D)
        text_features: Text features of shape (N, D) where N is number of semantic prompts
    
    Returns:
        Anomaly scores
    """
    # Normalize features
    visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix: (B*T, N)
    similarity = visual_features @ text_features.T
    
    # Take maximum similarity across all semantic prompts (max over prompts dimension)
    anomaly_scores, _ = torch.max(similarity, dim=-1)
    
    return anomaly_scores


def aggregate_frame_scores(frame_scores, method='mean'):
    """
    Aggregate frame-level scores for video-level prediction.
    
    Args:
        frame_scores: Frame scores of shape (T,)
        method: Aggregation method ('mean', 'max', 'top_k')
    
    Returns:
        Video-level anomaly score
    """
    if method == 'mean':
        return torch.mean(frame_scores)
    elif method == 'max':
        return torch.max(frame_scores)
    elif method == 'top_k':
        k = max(1, len(frame_scores) // 3)  # Top 33%
        top_k_scores, _ = torch.topk(frame_scores, k)
        return torch.mean(top_k_scores)
    else:
        return torch.mean(frame_scores)


def process_video_sliding_window(model, video_path, config, device):
    """
    Process a single video using sliding window approach.
    
    Args:
        model: VideoAnomalyCLIP model
        video_path: Path to video file
        config: Configuration dictionary
        device: Device to run inference on
    
    Returns:
        Dictionary with frame-level and video-level anomaly scores
    """
    from video_utils import get_video_info, create_sliding_windows, sample_video_clips
    
    # Get video information
    video_info = get_video_info(video_path)
    total_frames = video_info['total_frames']
    
    # Create sliding windows
    clip_length = config['clip_length']
    stride = config['stride']
    windows = create_sliding_windows(total_frames, clip_length, stride)
    
    # Initialize frame scores
    frame_scores = torch.zeros(total_frames)
    frame_counts = torch.zeros(total_frames)
    
    # Process each window
    for start_frame, end_frame in tqdm(windows, desc="Processing windows"):
        # Sample video clip
        video_clip = sample_video_clips(video_path, start_frame, clip_length)
        
        # Apply transforms
        from video_utils import apply_frame_transforms
        preprocess, _ = get_transform(argparse.Namespace(image_size=config['image_size']))
        video_tensor = apply_frame_transforms(video_clip, preprocess)
        
        # Add batch dimension and move to device
        video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
        
        with torch.no_grad():
            # Extract features
            B, T, C, H, W = video_tensor.shape
            video_flat = video_tensor.view(B * T, C, H, W)
            
            # Get visual features
            image_features, patch_features = model.encode_image(
                video_flat, config['features_list'], DPAM_layer=20
            )
            
            # Reshape and apply temporal transformer
            feature_dim = image_features.shape[1]
            image_features = image_features.view(B, T, feature_dim)
            temporal_features = model.temporal_transformer(image_features)
            
            # Calculate frame-level anomaly scores
            temporal_features_flat = temporal_features.view(T, feature_dim)
            
            # Load semantic prompts and encode
            semantic_prompts = config['semantic_prompts']
            from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
            
            tokenized_prompts = []
            for prompt in semantic_prompts:
                tokens = tokenizer.encode(prompt)
                tokenized_prompts.append(tokens)
            
            # Get text features for semantic prompts
            text_tokens = torch.zeros(len(semantic_prompts), 77, dtype=torch.long).to(device)
            for i, tokens in enumerate(tokenized_prompts):
                text_tokens[i, :len(tokens)] = torch.tensor(tokens[:77])
            
            text_features = model.encode_text(text_tokens)
            
            # Calculate anomaly scores for this window
            window_scores = calculate_anomaly_score(temporal_features_flat, text_features)
            
            # Accumulate scores for overlapping regions
            actual_length = min(T, total_frames - start_frame)
            for i in range(actual_length):
                frame_idx = start_frame + i
                frame_scores[frame_idx] += window_scores[i].cpu()
                frame_counts[frame_idx] += 1
    
    # Average scores for overlapping regions
    frame_counts[frame_counts == 0] = 1  # Avoid division by zero
    frame_scores = frame_scores / frame_counts
    
    # Calculate video-level score
    video_score = aggregate_frame_scores(frame_scores, method=config.get('score_aggregation', 'max'))
    
    return {
        'frame_scores': frame_scores.numpy(),
        'video_score': video_score.item()
    }


def test(config_path, args):
    """Main testing function."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create logger
    os.makedirs(config['results_dir'], exist_ok=True)
    logger = get_logger(config['results_dir'])
    logger.info(f"Starting testing with config: {config}")
    
    # Model parameters
    AnomalyCLIP_parameters = {
        "Prompt_length": config['n_ctx'],
        "learnabel_text_embedding_depth": config['depth'], 
        "learnabel_text_embedding_length": config['t_n_ctx']
    }
    
    # Load pre-trained model
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
    
    # Load trained weights
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    
    # Load prompt learner
    prompt_learner = VideoAnomalyCLIP_PromptLearner(
        video_model.to("cpu"), AnomalyCLIP_parameters
    )
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    
    # Load temporal transformer weights
    if 'temporal_transformer' in checkpoint:
        video_model.temporal_transformer.load_state_dict(checkpoint['temporal_transformer'])
    
    video_model.to(device)
    video_model.visual.DAPM_replace(DPAM_layer=20)
    video_model.eval()
    prompt_learner.eval()
    
    # Setup test dataset
    preprocess, target_transform = get_transform(args)
    test_dataset = create_dataset(
        root=config['dataset_path'],
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=config['dataset_name'],
        mode='test',
        clip_length=config['clip_length'],
        stride=config['stride']
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one video at a time
        shuffle=False,
        num_workers=1
    )
    
    # Initialize results storage
    results = {}
    obj_list = test_dataset.obj_list
    
    for obj in obj_list:
        results[obj] = {
            'gt_sp': [],        # Video-level ground truth
            'pr_sp': [],        # Video-level predictions
            'imgs_masks': [],   # Frame-level ground truth masks
            'anomaly_maps': []  # Frame-level anomaly maps
        }
    
    logger.info("Starting inference...")
    
    # Process each video
    processed_videos = set()
    
    for batch_idx, items in enumerate(tqdm(test_dataloader, desc="Testing")):
        video_path = items['video_path'][0]
        cls_name = items['cls_name'][0]
        anomaly = items['anomaly'][0].item()
        
        # Skip if we've already processed this video
        if video_path in processed_videos:
            continue
        processed_videos.add(video_path)
        
        logger.info(f"Processing video: {video_path}")
        
        try:
            # Process entire video with sliding windows
            video_results = process_video_sliding_window(
                video_model, video_path, config, device
            )
            
            # Store results
            results[cls_name]['gt_sp'].append(anomaly)
            results[cls_name]['pr_sp'].append(video_results['video_score'])
            
            # For frame-level evaluation, we need to handle ground truth masks
            gt_masks = items['gt_mask'].squeeze().numpy()  # (T, H, W)
            frame_scores = video_results['frame_scores']
            
            # Resize frame scores to match ground truth resolution if needed
            if len(frame_scores) != gt_masks.shape[0]:
                # Interpolate scores to match ground truth length
                import scipy.interpolate as interp
                x_old = np.linspace(0, 1, len(frame_scores))
                x_new = np.linspace(0, 1, gt_masks.shape[0])
                frame_scores = interp.interp1d(x_old, frame_scores, kind='linear')(x_new)
            
            # Apply Gaussian smoothing if specified
            if config.get('gaussian_sigma', 0) > 0:
                frame_scores = gaussian_filter(frame_scores, sigma=config['gaussian_sigma'])
            
            # Store frame-level results
            results[cls_name]['imgs_masks'].append(torch.from_numpy(gt_masks))
            
            # Create anomaly maps for each frame
            anomaly_maps = []
            for score in frame_scores:
                # Create a dummy anomaly map (in practice, you'd want pixel-level maps)
                H, W = gt_masks.shape[1], gt_masks.shape[2]
                anomaly_map = np.full((H, W), score)
                anomaly_maps.append(anomaly_map)
            
            results[cls_name]['anomaly_maps'].append(np.stack(anomaly_maps))
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            # Add dummy results to maintain consistency
            results[cls_name]['gt_sp'].append(anomaly)
            results[cls_name]['pr_sp'].append(0.0)
            continue
    
    # Evaluate results
    logger.info("Evaluating results...")
    
    from tabulate import tabulate
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    
    for obj in obj_list:
        if not results[obj]['gt_sp']:  # Skip if no data
            continue
            
        table = [obj]
        
        # Convert lists to arrays for evaluation
        if results[obj]['imgs_masks']:
            results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            results[obj]['anomaly_maps'] = np.concatenate(results[obj]['anomaly_maps'], axis=0)
        
        # Video-level metrics
        if len(results[obj]['gt_sp']) > 0:
            image_auroc = image_level_metrics(results, obj, "image-auroc") 
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.extend([
                f"{image_auroc * 100:.1f}",
                f"{image_ap * 100:.1f}"
            ])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        
        # Frame-level metrics (if available)
        if results[obj]['imgs_masks'] is not None and len(results[obj]['anomaly_maps']) > 0:
            try:
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro") 
                table.extend([
                    f"{pixel_auroc * 100:.1f}",
                    f"{pixel_aupro * 100:.1f}"
                ])
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            except Exception as e:
                logger.warning(f"Could not compute pixel-level metrics for {obj}: {e}")
                table.extend(["N/A", "N/A"])
        else:
            table.extend(["N/A", "N/A"])
        
        table_ls.append(table)
    
    # Add mean row
    mean_row = ['mean']
    if image_auroc_list:
        mean_row.extend([
            f"{np.mean(image_auroc_list) * 100:.1f}",
            f"{np.mean(image_ap_list) * 100:.1f}"
        ])
    if pixel_auroc_list:
        mean_row.extend([
            f"{np.mean(pixel_auroc_list) * 100:.1f}",
            f"{np.mean(pixel_aupro_list) * 100:.1f}"
        ])
    else:
        mean_row.extend(["N/A", "N/A"])
    
    table_ls.append(mean_row)
    
    # Print results table
    headers = ['Object', 'Video-AUROC', 'Video-AP', 'Frame-AUROC', 'Frame-AUPRO']
    results_table = tabulate(table_ls, headers=headers, tablefmt="pipe")
    logger.info(f"\nResults:\n{results_table}")
    
    # Save results
    results_file = os.path.join(config['results_dir'], 'results.txt')
    with open(results_file, 'w') as f:
        f.write(results_table)
    
    logger.info(f"Results saved to {results_file}")
    logger.info("Testing completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VideoAnomalyCLIP Testing")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    
    # Image processing parameters (for compatibility)
    parser.add_argument("--image_size", type=int, default=336, help="Image size")
    parser.add_argument("--features_list", type=int, nargs="+", 
                       default=[6, 12, 18, 24], help="Features used")
    
    args = parser.parse_args()
    
    test(args.config, args)