"""
Dataset processing scripts for video anomaly detection datasets.
Converts various dataset formats to unified JSON metadata format.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import scipy.io as sio
from PIL import Image


class ShanghaiTechProcessor:
    """Process ShanghaiTech Campus dataset."""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.meta_path = self.dataset_root / "meta.json"
    
    def process(self):
        """Process ShanghaiTech dataset and create meta.json."""
        print("Processing ShanghaiTech Campus dataset...")
        
        info = {"train": {}, "test": {}}
        
        # Process training data (all normal videos)
        train_dir = self.dataset_root / "training" / "videos"
        if train_dir.exists():
            train_videos = []
            for video_file in sorted(train_dir.glob("*.avi")):
                video_info = {
                    "video_path": f"training/videos/{video_file.name}",
                    "anomaly": 0,  # All training videos are normal
                    "gt_path": "",
                    "total_frames": self._get_video_length(video_file)
                }
                train_videos.append(video_info)
            
            info["train"]["campus_scene"] = train_videos
            print(f"Processed {len(train_videos)} training videos")
        
        # Process testing data
        test_dir = self.dataset_root / "testing" / "videos"
        gt_dir = self.dataset_root / "testing" / "test_frame_mask"
        
        if test_dir.exists():
            test_videos = []
            for video_file in sorted(test_dir.glob("*.avi")):
                # Check if ground truth exists
                gt_file = gt_dir / f"{video_file.stem}.npy"
                is_anomalous = 0
                gt_path = ""
                
                if gt_file.exists():
                    gt_data = np.load(gt_file)
                    is_anomalous = 1 if np.any(gt_data > 0) else 0
                    gt_path = f"testing/test_frame_mask/{gt_file.name}"
                
                video_info = {
                    "video_path": f"testing/videos/{video_file.name}",
                    "anomaly": is_anomalous,
                    "gt_path": gt_path,
                    "total_frames": self._get_video_length(video_file)
                }
                test_videos.append(video_info)
            
            info["test"]["campus_scene"] = test_videos
            print(f"Processed {len(test_videos)} testing videos")
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"ShanghaiTech metadata saved to {self.meta_path}")
        return info
    
    def _get_video_length(self, video_path):
        """Get number of frames in video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return length
        except:
            return 0


class UCSDProcessor:
    """Process UCSD Ped1/Ped2 datasets."""
    
    def __init__(self, dataset_root, dataset_type="ped2"):
        self.dataset_root = Path(dataset_root)
        self.dataset_type = dataset_type.lower()
        self.meta_path = self.dataset_root / "meta.json"
    
    def process(self):
        """Process UCSD dataset and create meta.json."""
        print(f"Processing UCSD {self.dataset_type.upper()} dataset...")
        
        info = {"train": {}, "test": {}}
        
        # UCSD uses image sequences, need to convert to video format
        # or process as image sequences
        
        # Process test data (main focus for evaluation)
        test_videos = []
        test_dir = self.dataset_root / "Test"
        gt_dir = self.dataset_root / "Test_gt"
        
        if test_dir.exists():
            for test_folder in sorted(test_dir.glob("Test*")):
                if not test_folder.is_dir():
                    continue
                
                # Check for ground truth
                gt_file = gt_dir / f"{test_folder.name}_gt.mat"
                is_anomalous = 0
                gt_path = ""
                
                if gt_file.exists():
                    try:
                        gt_data = sio.loadmat(str(gt_file))
                        # Extract ground truth array (varies by dataset version)
                        if 'gt' in gt_data:
                            gt_array = gt_data['gt']
                        elif 'volLabel' in gt_data:
                            gt_array = gt_data['volLabel']
                        else:
                            gt_array = list(gt_data.values())[-1]  # Last item usually contains data
                        
                        is_anomalous = 1 if np.any(gt_array > 0) else 0
                        
                        # Convert .mat to .npy for easier processing
                        npy_gt_path = gt_dir / f"{test_folder.name}_gt.npy"
                        np.save(npy_gt_path, gt_array)
                        gt_path = f"Test_gt/{npy_gt_path.name}"
                        
                    except Exception as e:
                        print(f"Error processing ground truth for {test_folder.name}: {e}")
                
                # Create video from image sequence or use existing video
                video_info = {
                    "video_path": f"Test/{test_folder.name}",  # Folder path for image sequences
                    "anomaly": is_anomalous,
                    "gt_path": gt_path,
                    "total_frames": len(list(test_folder.glob("*.tif")))
                }
                test_videos.append(video_info)
            
            info["test"]["pedestrian"] = test_videos
            print(f"Processed {len(test_videos)} test sequences")
        
        # Process training data (if needed, usually not used for zero-shot)
        train_dir = self.dataset_root / "Train"
        if train_dir.exists():
            train_videos = []
            for train_folder in sorted(train_dir.glob("Train*")):
                if not train_folder.is_dir():
                    continue
                
                video_info = {
                    "video_path": f"Train/{train_folder.name}",
                    "anomaly": 0,  # Training data is normal
                    "gt_path": "",
                    "total_frames": len(list(train_folder.glob("*.tif")))
                }
                train_videos.append(video_info)
            
            info["train"]["pedestrian"] = train_videos
            print(f"Processed {len(train_videos)} training sequences")
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"UCSD {self.dataset_type.upper()} metadata saved to {self.meta_path}")
        return info
    
    def convert_sequences_to_videos(self, output_format="avi"):
        """Convert image sequences to video files (optional)."""
        print("Converting image sequences to videos...")
        
        for split in ["Train", "Test"]:
            split_dir = self.dataset_root / split
            if not split_dir.exists():
                continue
            
            videos_dir = self.dataset_root / f"{split}_videos"
            videos_dir.mkdir(exist_ok=True)
            
            for seq_folder in sorted(split_dir.glob(f"{split}*")):
                if not seq_folder.is_dir():
                    continue
                
                # Get all images
                images = sorted(seq_folder.glob("*.tif"))
                if not images:
                    continue
                
                # Create video
                video_path = videos_dir / f"{seq_folder.name}.{output_format}"
                self._create_video_from_images(images, video_path)
        
        print("Video conversion completed")
    
    def _create_video_from_images(self, image_paths, output_path, fps=10):
        """Create video from image sequence."""
        if not image_paths:
            return
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_paths[0]))
        height, width, channels = first_img.shape
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            out.write(img)
        
        out.release()
        print(f"Created video: {output_path}")


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video anomaly detection datasets")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["shanghaitech", "ucsd_ped1", "ucsd_ped2"],
                       help="Dataset to process")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--convert_to_video", action="store_true",
                       help="Convert image sequences to videos (UCSD only)")
    
    args = parser.parse_args()
    
    if args.dataset == "shanghaitech":
        processor = ShanghaiTechProcessor(args.data_root)
        processor.process()
    
    elif args.dataset in ["ucsd_ped1", "ucsd_ped2"]:
        dataset_type = args.dataset.split("_")[1]  # Extract ped1 or ped2
        processor = UCSDProcessor(args.data_root, dataset_type)
        
        if args.convert_to_video:
            processor.convert_sequences_to_videos()
        
        processor.process()
    
    print("Dataset processing completed!")


if __name__ == "__main__":
    main()