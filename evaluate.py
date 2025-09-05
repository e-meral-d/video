"""
Independent evaluation script for video anomaly detection.
Computes frame-level and video-level metrics from saved predictions.
"""

import numpy as np
import argparse
import os
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from tabulate import tabulate


def load_ground_truth(gt_path, dataset_name):
    """
    Load ground truth annotations.
    
    Args:
        gt_path: Path to ground truth file/directory
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with ground truth data
    """
    gt_data = {}
    
    if dataset_name.lower() == 'ucsd':
        # UCSD dataset format
        if os.path.isfile(gt_path):
            # Single file with frame labels
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                gt_data['frame_labels'] = [int(line.strip()) for line in lines]
        else:
            # Directory with individual files
            gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith('.txt')])
            gt_data['frame_labels'] = []
            for gt_file in gt_files:
                with open(os.path.join(gt_path, gt_file), 'r') as f:
                    labels = [int(line.strip()) for line in f.readlines()]
                    gt_data['frame_labels'].extend(labels)
    
    elif dataset_name.lower() == 'shanghaitech':
        # ShanghaiTech dataset format  
        gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith('.npy')])
        gt_data['frame_labels'] = []
        for gt_file in gt_files:
            labels = np.load(os.path.join(gt_path, gt_file))
            gt_data['frame_labels'].extend(labels.tolist())
    
    return gt_data


def load_predictions(pred_path):
    """
    Load prediction scores.
    
    Args:
        pred_path: Path to predictions file
        
    Returns:
        Dictionary with prediction data
    """
    if pred_path.endswith('.json'):
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
    elif pred_path.endswith('.npy'):
        scores = np.load(pred_path)
        pred_data = {'frame_scores': scores.tolist()}
    else:
        # Text file
        with open(pred_path, 'r') as f:
            lines = f.readlines()
            scores = [float(line.strip()) for line in lines]
            pred_data = {'frame_scores': scores}
    
    return pred_data


def calculate_frame_level_auc(gt_labels, pred_scores):
    """Calculate frame-level AUC."""
    if len(gt_labels) != len(pred_scores):
        min_len = min(len(gt_labels), len(pred_scores))
        gt_labels = gt_labels[:min_len]
        pred_scores = pred_scores[:min_len]
    
    if len(set(gt_labels)) < 2:
        print("Warning: Ground truth contains only one class")
        return 0.0
    
    return roc_auc_score(gt_labels, pred_scores)


def calculate_frame_level_ap(gt_labels, pred_scores):
    """Calculate frame-level Average Precision."""
    if len(gt_labels) != len(pred_scores):
        min_len = min(len(gt_labels), len(pred_scores))
        gt_labels = gt_labels[:min_len]
        pred_scores = pred_scores[:min_len]
    
    if len(set(gt_labels)) < 2:
        return 0.0
    
    return average_precision_score(gt_labels, pred_scores)


def calculate_video_level_metrics(gt_labels, pred_scores, aggregation='max'):
    """
    Calculate video-level metrics by aggregating frame scores.
    
    Args:
        gt_labels: Frame-level ground truth
        pred_scores: Frame-level prediction scores
        aggregation: How to aggregate frame scores ('max', 'mean', 'top_k')
        
    Returns:
        Dictionary with video-level metrics
    """
    # For simplicity, assume each video has equal number of frames
    # In practice, you'd need video boundaries information
    
    # Simple approach: if any frame is anomalous, video is anomalous
    video_gt = 1 if any(gt_labels) else 0
    
    if aggregation == 'max':
        video_pred = max(pred_scores)
    elif aggregation == 'mean':
        video_pred = np.mean(pred_scores)
    elif aggregation == 'top_k':
        k = max(1, len(pred_scores) // 10)  # Top 10%
        video_pred = np.mean(sorted(pred_scores, reverse=True)[:k])
    else:
        video_pred = max(pred_scores)
    
    return {
        'video_gt': video_gt,
        'video_pred': video_pred
    }


def plot_roc_curve(gt_labels, pred_scores, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(gt_labels, pred_scores)
    auc_score = roc_auc_score(gt_labels, pred_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Frame Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pr_curve(gt_labels, pred_scores, save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(gt_labels, pred_scores)
    ap_score = average_precision_score(gt_labels, pred_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Frame Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_results(gt_path, pred_path, dataset_name, output_dir=None, plot_curves=False):
    """
    Main evaluation function.
    
    Args:
        gt_path: Path to ground truth
        pred_path: Path to predictions
        dataset_name: Dataset name
        output_dir: Output directory for results
        plot_curves: Whether to plot ROC and PR curves
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load data
    print("Loading ground truth...")
    gt_data = load_ground_truth(gt_path, dataset_name)
    
    print("Loading predictions...")
    pred_data = load_predictions(pred_path)
    
    gt_labels = gt_data['frame_labels']
    pred_scores = pred_data['frame_scores']
    
    print(f"Loaded {len(gt_labels)} ground truth labels")
    print(f"Loaded {len(pred_scores)} prediction scores")
    
    # Calculate frame-level metrics
    print("\nCalculating frame-level metrics...")
    frame_auc = calculate_frame_level_auc(gt_labels, pred_scores)
    frame_ap = calculate_frame_level_ap(gt_labels, pred_scores)
    
    # Calculate video-level metrics (simplified)
    print("Calculating video-level metrics...")
    video_metrics_max = calculate_video_level_metrics(gt_labels, pred_scores, 'max')
    video_metrics_mean = calculate_video_level_metrics(gt_labels, pred_scores, 'mean')
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'frame_level': {
            'auc': frame_auc,
            'ap': frame_ap
        },
        'video_level': {
            'max_aggregation': video_metrics_max,
            'mean_aggregation': video_metrics_mean
        },
        'data_stats': {
            'total_frames': len(gt_labels),
            'anomalous_frames': sum(gt_labels),
            'normal_frames': len(gt_labels) - sum(gt_labels),
            'anomaly_ratio': sum(gt_labels) / len(gt_labels) if len(gt_labels) > 0 else 0
        }
    }
    
    # Print results table
    table_data = [
        ['Frame-level AUC', f'{frame_auc:.3f}'],
        ['Frame-level AP', f'{frame_ap:.3f}'],
        ['Total Frames', len(gt_labels)],
        ['Anomalous Frames', sum(gt_labels)],
        ['Normal Frames', len(gt_labels) - sum(gt_labels)],
        ['Anomaly Ratio', f'{sum(gt_labels) / len(gt_labels):.3f}']
    ]
    
    results_table = tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid')
    print(f"\nEvaluation Results for {dataset_name}:")
    print(results_table)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {results_file}")
        
        # Save summary table
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Results for {dataset_name}\n")
            f.write("="*50 + "\n\n")
            f.write(results_table)
            f.write(f"\n\nDetailed Statistics:\n")
            f.write(f"Total frames processed: {len(gt_labels)}\n")
            f.write(f"Anomalous frames: {sum(gt_labels)} ({sum(gt_labels)/len(gt_labels)*100:.1f}%)\n")
            f.write(f"Normal frames: {len(gt_labels)-sum(gt_labels)} ({(len(gt_labels)-sum(gt_labels))/len(gt_labels)*100:.1f}%)\n")
        print(f"Summary saved to {summary_file}")
    
    # Plot curves if requested
    if plot_curves and len(set(gt_labels)) >= 2:
        print("\nGenerating plots...")
        if output_dir:
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            pr_path = os.path.join(output_dir, 'pr_curve.png')
        else:
            roc_path = pr_path = None
            
        plot_roc_curve(gt_labels, pred_scores, roc_path)
        plot_pr_curve(gt_labels, pred_scores, pr_path)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate video anomaly detection results')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth file/directory')
    parser.add_argument('--pred_path', type=str, required=True, 
                       help='Path to predictions file')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ucsd', 'shanghaitech', 'avenue'],
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results and plots')
    parser.add_argument('--plot_curves', action='store_true',
                       help='Generate ROC and PR curves')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_results(
        args.gt_path, 
        args.pred_path, 
        args.dataset,
        args.output_dir,
        args.plot_curves
    )
    
    print("\nEvaluation completed successfully!")