"""
ShanghaiTech Campus dataset processor for video anomaly detection.
Converts ShanghaiTech dataset to unified JSON metadata format.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path


class ShanghaiTechProcessor:
    """Process ShanghaiTech Campus dataset."""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.meta_path = self.dataset_root / "meta.json"
        
        print(f"ShanghaiTech Dataset Root: {self.dataset_root}")
        print(f"Meta file will be saved to: {self.meta_path}")
    
    def process(self):
        """Process ShanghaiTech dataset and create meta.json."""
        print("Processing ShanghaiTech Campus dataset...")
        
        info = {"train": {}, "test": {}}
        
        # Process training data (all normal videos)
        print("\n--- Processing Training Data ---")
        train_dir = self.dataset_root / "training" / "videos"
        
        if not train_dir.exists():
            print(f"Warning: Training directory not found at {train_dir}")
            # Try alternative paths
            train_dir = self.dataset_root / "Train"
            if not train_dir.exists():
                train_dir = self.dataset_root / "training"
        
        if train_dir.exists():
            print(f"Found training directory: {train_dir}")
            train_videos = self._process_training_videos(train_dir)
            info["train"]["campus_scene"] = train_videos
            print(f"Processed {len(train_videos)} training videos")
        else:
            print("No training directory found!")
            info["train"]["campus_scene"] = []
        
        # Process testing data
        print("\n--- Processing Testing Data ---")
        test_dir = self.dataset_root / "testing" / "videos"
        gt_dir = self.dataset_root / "testing" / "test_frame_mask"
        
        # Try alternative paths if not found
        if not test_dir.exists():
            test_dir = self.dataset_root / "Test"
            gt_dir = self.dataset_root / "Test_gt"
            
            if not test_dir.exists():
                test_dir = self.dataset_root / "testing"
                gt_dir = self.dataset_root / "ground_truth"
        
        if test_dir.exists():
            print(f"Found testing directory: {test_dir}")
            print(f"Ground truth directory: {gt_dir}")
            test_videos = self._process_testing_videos(test_dir, gt_dir)
            info["test"]["campus_scene"] = test_videos
            print(f"Processed {len(test_videos)} testing videos")
        else:
            print("No testing directory found!")
            info["test"]["campus_scene"] = []
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nShanghaiTech metadata saved to {self.meta_path}")
        self._print_summary(info)
        return info
    
    def _process_training_videos(self, train_dir):
        """Process training videos."""
        train_videos = []
        
        # Look for video files with common extensions
        video_extensions = ["*.avi", "*.mp4", "*.mov", "*.mkv"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(train_dir.glob(ext))
        
        print(f"Found {len(video_files)} video files in training directory")
        
        for video_file in sorted(video_files):
            print(f"Processing: {video_file.name}")
            
            # Get relative path from dataset root
            rel_path = video_file.relative_to(self.dataset_root)
            
            video_info = {
                "video_path": str(rel_path).replace('\\', '/'),  # Ensure forward slashes
                "anomaly": 0,  # All training videos are normal
                "gt_path": "",
                "total_frames": self._get_video_length(video_file)
            }
            train_videos.append(video_info)
        
        return train_videos
    
    def _process_testing_videos(self, test_dir, gt_dir):
        """Process testing videos."""
        test_videos = []
        
        # Look for video files
        video_extensions = ["*.avi", "*.mp4", "*.mov", "*.mkv"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(test_dir.glob(ext))
        
        print(f"Found {len(video_files)} video files in testing directory")
        
        for video_file in sorted(video_files):
            print(f"Processing: {video_file.name}")
            
            # Get relative path from dataset root
            rel_path = video_file.relative_to(self.dataset_root)
            
            # Check if ground truth exists
            gt_file_patterns = [
                gt_dir / f"{video_file.stem}.npy",
                gt_dir / f"{video_file.stem}.txt", 
                gt_dir / f"{video_file.stem}_gt.npy",
                gt_dir / f"{video_file.stem}_gt.txt"
            ]
            
            is_anomalous = 0
            gt_path = ""
            
            for gt_file in gt_file_patterns:
                if gt_file.exists():
                    print(f"  Found ground truth: {gt_file.name}")
                    
                    try:
                        if gt_file.suffix == '.npy':
                            gt_data = np.load(gt_file)
                        elif gt_file.suffix == '.txt':
                            gt_data = np.loadtxt(gt_file)
                        else:
                            continue
                        
                        is_anomalous = 1 if np.any(gt_data > 0) else 0
                        gt_rel_path = gt_file.relative_to(self.dataset_root)
                        gt_path = str(gt_rel_path).replace('\\', '/')
                        break
                        
                    except Exception as e:
                        print(f"  Error reading ground truth {gt_file}: {e}")
                        continue
            
            if not gt_path:
                print(f"  No ground truth found for {video_file.name}")
            
            video_info = {
                "video_path": str(rel_path).replace('\\', '/'),
                "anomaly": is_anomalous,
                "gt_path": gt_path,
                "total_frames": self._get_video_length(video_file)
            }
            test_videos.append(video_info)
        
        return test_videos
    
    def _get_video_length(self, video_path):
        """Get number of frames in video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if length <= 0:
                print(f"  Warning: Could not get frame count for {video_path}")
                return 1000  # Default fallback
            
            print(f"  Video length: {length} frames")
            return length
            
        except Exception as e:
            print(f"  Error getting video length for {video_path}: {e}")
            return 1000  # Default fallback
    
    def _print_summary(self, info):
        """Print processing summary."""
        print("\n" + "="*50)
        print("SHANGHAITECH PROCESSING SUMMARY")
        print("="*50)
        
        train_count = len(info["train"]["campus_scene"])
        test_count = len(info["test"]["campus_scene"])
        
        print(f"Training videos: {train_count}")
        print(f"Testing videos: {test_count}")
        
        # Count anomalies in test set
        test_anomalies = sum(1 for v in info["test"]["campus_scene"] if v["anomaly"] == 1)
        test_normal = test_count - test_anomalies
        
        print(f"Test anomalous videos: {test_anomalies}")
        print(f"Test normal videos: {test_normal}")
        print("="*50)


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process ShanghaiTech Campus dataset")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of ShanghaiTech dataset")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset root directory '{args.data_root}' does not exist!")
        return
    
    # Process dataset
    processor = ShanghaiTechProcessor(args.data_root)
    processor.process()
    
    print("\nShanghaiTech dataset processing completed!")


if __name__ == "__main__":
    main()