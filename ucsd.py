"""
UCSD Ped1/Ped2 dataset processor for video anomaly detection.
Converts UCSD datasets (image sequences) to unified JSON metadata format.
"""

import os
import json
import numpy as np
import cv2
import glob
from pathlib import Path
import scipy.io as sio


class UCSDProcessor:
    """Process UCSD Ped1/Ped2 datasets."""
    
    def __init__(self, dataset_root, dataset_type="ped2"):
        self.dataset_root = Path(dataset_root)
        self.dataset_type = dataset_type.lower()
        self.meta_path = self.dataset_root / "meta.json"
        
        print(f"UCSD {self.dataset_type.upper()} Dataset Root: {self.dataset_root}")
        print(f"Meta file will be saved to: {self.meta_path}")
    
    def process(self):
        """Process UCSD dataset and create meta.json."""
        print(f"Processing UCSD {self.dataset_type.upper()} dataset...")
        
        info = {"train": {}, "test": {}}
        
        # Process test data (main focus for evaluation)
        print("\n--- Processing Test Data ---")
        test_videos = self._process_test_data()
        info["test"]["pedestrian"] = test_videos
        
        # Process training data (optional, usually not used for zero-shot)
        print("\n--- Processing Train Data ---")
        train_videos = self._process_train_data()
        info["train"]["pedestrian"] = train_videos
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nUCSD {self.dataset_type.upper()} metadata saved to {self.meta_path}")
        self._print_summary(info)
        return info
    
    def _process_test_data(self):
        """Process test data sequences."""
        test_videos = []
        test_dir = self.dataset_root / "Test"
        
        if not test_dir.exists():
            print(f"Warning: Test directory not found at {test_dir}")
            return []
        
        print(f"Found test directory: {test_dir}")
        
        # Load unified ground truth once
        self._unified_gt_data = self._load_unified_ground_truth()
        if self._unified_gt_data is None:
            print("Warning: No unified ground truth found, all sequences will be marked as normal")
        
        # Get all test sequence folders
        test_folders = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("Test")])
        print(f"Found {len(test_folders)} test sequences")
        
        for test_folder in test_folders:
            print(f"Processing: {test_folder.name}")
            
            # Count frames in sequence
            frame_count = self._count_frames_in_sequence(test_folder)
            
            # Process ground truth using unified data
            gt_info = self._process_ground_truth(test_folder.name, None)  # No gt_dir needed
            
            # Get relative path from dataset root
            rel_path = test_folder.relative_to(self.dataset_root)
            
            video_info = {
                "video_path": str(rel_path).replace('\\', '/'),
                "anomaly": gt_info["is_anomalous"],
                "gt_path": gt_info["gt_path"],
                "total_frames": frame_count,
                "sequence_type": "image_sequence"  # Mark as image sequence
            }
            test_videos.append(video_info)
        
        return test_videos
    
    def _process_train_data(self):
        """Process training data sequences."""
        train_videos = []
        train_dir = self.dataset_root / "Train"
        
        if not train_dir.exists():
            print(f"Training directory not found at {train_dir}")
            return []
        
        print(f"Found train directory: {train_dir}")
        
        # Get all train sequence folders
        train_folders = sorted([d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("Train")])
        print(f"Found {len(train_folders)} training sequences")
        
        for train_folder in train_folders:
            print(f"Processing: {train_folder.name}")
            
            # Count frames in sequence
            frame_count = self._count_frames_in_sequence(train_folder)
            
            # Get relative path from dataset root
            rel_path = train_folder.relative_to(self.dataset_root)
            
            video_info = {
                "video_path": str(rel_path).replace('\\', '/'),
                "anomaly": 0,  # Training data is normal
                "gt_path": "",
                "total_frames": frame_count,
                "sequence_type": "image_sequence"
            }
            train_videos.append(video_info)
        
        return train_videos
    
    def _count_frames_in_sequence(self, sequence_dir):
        """Count number of frames in an image sequence."""
        # UCSD datasets typically use .tif files
        image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
        
        frame_count = 0
        for ext in image_extensions:
            frames = list(sequence_dir.glob(ext))
            frame_count = max(frame_count, len(frames))
        
        print(f"  Found {frame_count} frames")
        return frame_count
    
    def _process_ground_truth(self, sequence_name, gt_dir):
        """Process ground truth for a test sequence from unified .m file."""
        gt_info = {"is_anomalous": 0, "gt_path": ""}
        
        if not hasattr(self, '_unified_gt_data'):
            self._unified_gt_data = self._load_unified_ground_truth()
        
        if self._unified_gt_data is None:
            print(f"  No unified ground truth data found")
            return gt_info
        
        # Extract sequence number from name (e.g., "Test001" -> 1)
        try:
            if sequence_name.startswith("Test"):
                seq_num = int(sequence_name.replace("Test", ""))
            else:
                seq_num = int(sequence_name[-3:])  # Last 3 digits
            
            # Get ground truth for this sequence (1-indexed)
            if seq_num <= len(self._unified_gt_data):
                gt_data = self._unified_gt_data[seq_num - 1]  # Convert to 0-indexed
                
                if gt_data is not None and gt_data.size > 0:
                    # Check if sequence contains anomalies
                    is_anomalous = 1 if np.any(gt_data > 0) else 0
                    
                    # Save individual ground truth file
                    os.makedirs(self.dataset_root / "processed_gt", exist_ok=True)
                    npy_gt_path = self.dataset_root / "processed_gt" / f"{sequence_name}_gt.npy"
                    np.save(npy_gt_path, gt_data.astype(np.uint8))
                    
                    # Get relative path
                    gt_rel_path = npy_gt_path.relative_to(self.dataset_root)
                    
                    gt_info = {
                        "is_anomalous": is_anomalous,
                        "gt_path": str(gt_rel_path).replace('\\', '/')
                    }
                    
                    print(f"  Sequence {seq_num}: {'Anomalous' if is_anomalous else 'Normal'}")
                    print(f"  GT shape: {gt_data.shape}")
                else:
                    print(f"  No ground truth data for sequence {seq_num}")
            else:
                print(f"  Sequence number {seq_num} out of range")
                
        except ValueError as e:
            print(f"  Could not extract sequence number from {sequence_name}: {e}")
        except Exception as e:
            print(f"  Error processing ground truth for {sequence_name}: {e}")
        
        return gt_info
    
    def _load_unified_ground_truth(self):
        """Load unified ground truth from UCSDped1.m or UCSDped2.m file."""
        # Look for unified ground truth files
        gt_patterns = [
            self.dataset_root / f"UCSD{self.dataset_type}.m",
            self.dataset_root / f"ucsd{self.dataset_type}.m", 
            self.dataset_root / f"UCSD_{self.dataset_type.upper()}.m",
            self.dataset_root / f"{self.dataset_type}_gt.m",
            self.dataset_root / "ground_truth.m"
        ]
        
        print(f"Looking for unified ground truth files...")
        
        for gt_file in gt_patterns:
            print(f"  Checking: {gt_file}")
            if gt_file.exists():
                print(f"  Found unified ground truth: {gt_file}")
                try:
                    return self._load_matlab_unified_gt(gt_file)
                except Exception as e:
                    print(f"  Error loading {gt_file}: {e}")
                    continue
        
        print("  No unified ground truth file found")
        return None
    
    def _load_matlab_unified_gt(self, gt_file):
        """Load MATLAB unified ground truth file."""
        try:
            mat_data = sio.loadmat(str(gt_file))
            print(f"  MATLAB file keys: {list(mat_data.keys())}")
            
            # Common field names for unified ground truth
            gt_field_names = [
                'gt', 'volLabel', 'l', 'labels', 'groundtruth', 
                'anomaly_labels', 'test_labels', 'vol'
            ]
            
            # Try known field names first
            for field_name in gt_field_names:
                if field_name in mat_data:
                    gt_data = mat_data[field_name]
                    print(f"  Loading GT from field: {field_name}")
                    return self._process_unified_gt_array(gt_data)
            
            # If no known field, try to find the largest cell or numeric array
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    if isinstance(value, np.ndarray):
                        if value.dtype == 'object':  # Cell array
                            print(f"  Loading GT from cell array: {key}")
                            return self._process_unified_gt_array(value)
                        elif value.size > 100:  # Large numeric array
                            print(f"  Loading GT from numeric array: {key}")
                            return self._process_unified_gt_array(value)
            
            print(f"  Warning: No suitable ground truth found in {gt_file}")
            return None
            
        except Exception as e:
            print(f"  Error loading unified ground truth {gt_file}: {e}")
            return None
    
    def _process_unified_gt_array(self, gt_array):
        """Process unified ground truth array (typically a cell array)."""
        try:
            if gt_array.dtype == 'object':
                # Handle MATLAB cell array
                gt_list = []
                print(f"  Processing cell array with {len(gt_array)} sequences")
                
                for i, cell in enumerate(gt_array.flat):
                    if cell is not None and hasattr(cell, 'shape'):
                        # Process each cell (sequence ground truth)
                        seq_gt = self._process_gt_array(cell)
                        gt_list.append(seq_gt)
                        print(f"    Sequence {i+1}: shape {seq_gt.shape if seq_gt is not None else 'None'}")
                    else:
                        print(f"    Sequence {i+1}: empty or invalid")
                        gt_list.append(None)
                
                return gt_list
            
            else:
                # Handle regular numeric array
                print(f"  Processing numeric array with shape {gt_array.shape}")
                return [self._process_gt_array(gt_array)]
                
        except Exception as e:
            print(f"  Error processing unified GT array: {e}")
            return None
    
    def _load_mat_file(self, mat_file):
        """Load MATLAB .mat file and extract ground truth array."""
        try:
            mat_data = sio.loadmat(str(mat_file))
            
            # Common field names in UCSD dataset
            gt_field_names = ['gt', 'volLabel', 'l', 'labels']
            
            for field_name in gt_field_names:
                if field_name in mat_data:
                    gt_array = mat_data[field_name]
                    print(f"  Loaded GT from field: {field_name}")
                    return self._process_gt_array(gt_array)
            
            # If no known field names, try to find the largest numeric array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.size > 100:  # Reasonable size threshold
                        print(f"  Loaded GT from field: {key} (auto-detected)")
                        return self._process_gt_array(value)
            
            print(f"  Warning: No suitable ground truth array found in {mat_file}")
            return None
            
        except Exception as e:
            print(f"  Error loading .mat file {mat_file}: {e}")
            return None
    
    def _process_gt_array(self, gt_array):
        """Process ground truth array to standard format."""
        # Handle different possible shapes and data types
        if len(gt_array.shape) > 2:
            # If 3D array, take the first channel or reshape
            if gt_array.shape[2] == 1:
                gt_array = gt_array[:, :, 0]
            else:
                # Take maximum across channels
                gt_array = np.max(gt_array, axis=2)
        
        # Ensure binary values (0 or 1)
        if gt_array.max() > 1:
            gt_array = (gt_array > 0).astype(np.uint8)
        
        return gt_array
    
    def _print_summary(self, info):
        """Print processing summary."""
        print("\n" + "="*50)
        print(f"UCSD {self.dataset_type.upper()} PROCESSING SUMMARY")
        print("="*50)
        
        train_count = len(info["train"]["pedestrian"])
        test_count = len(info["test"]["pedestrian"])
        
        print(f"Training sequences: {train_count}")
        print(f"Testing sequences: {test_count}")
        
        # Count anomalies in test set
        test_anomalies = sum(1 for v in info["test"]["pedestrian"] if v["anomaly"] == 1)
        test_normal = test_count - test_anomalies
        
        print(f"Test anomalous sequences: {test_anomalies}")
        print(f"Test normal sequences: {test_normal}")
        
        # Calculate total frames
        total_train_frames = sum(v["total_frames"] for v in info["train"]["pedestrian"])
        total_test_frames = sum(v["total_frames"] for v in info["test"]["pedestrian"])
        
        print(f"Total training frames: {total_train_frames}")
        print(f"Total testing frames: {total_test_frames}")
        print("="*50)
    
    def convert_sequences_to_videos(self, output_format="avi", fps=10):
        """Convert image sequences to video files (optional)."""
        print(f"Converting image sequences to {output_format} videos...")
        
        for split in ["Train", "Test"]:
            split_dir = self.dataset_root / split
            if not split_dir.exists():
                continue
            
            # Create output directory for videos
            videos_dir = self.dataset_root / f"{split}_videos"
            videos_dir.mkdir(exist_ok=True)
            print(f"Output directory: {videos_dir}")
            
            # Process each sequence folder
            sequence_folders = sorted([d for d in split_dir.iterdir() 
                                     if d.is_dir() and d.name.startswith(split)])
            
            for seq_folder in sequence_folders:
                print(f"Converting: {seq_folder.name}")
                
                # Get all images in sequence
                image_files = self._get_sequence_images(seq_folder)
                
                if not image_files:
                    print(f"  No images found in {seq_folder}")
                    continue
                
                # Create video file
                video_path = videos_dir / f"{seq_folder.name}.{output_format}"
                success = self._create_video_from_images(image_files, video_path, fps)
                
                if success:
                    print(f"  Created: {video_path.name}")
                else:
                    print(f"  Failed to create: {video_path.name}")
        
        print("Video conversion completed")
    
    def _get_sequence_images(self, sequence_dir):
        """Get sorted list of images in a sequence."""
        image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(sequence_dir.glob(ext))
        
        # Sort by filename (assuming numeric ordering)
        image_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
        return image_files
    
    def _create_video_from_images(self, image_paths, output_path, fps=10):
        """Create video from image sequence."""
        if not image_paths:
            return False
        
        try:
            # Read first image to get dimensions
            first_img = cv2.imread(str(image_paths[0]))
            if first_img is None:
                print(f"    Could not read first image: {image_paths[0]}")
                return False
            
            height, width, channels = first_img.shape
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Write all images to video
            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    out.write(img)
                else:
                    print(f"    Warning: Could not read {img_path}")
            
            out.release()
            return True
            
        except Exception as e:
            print(f"    Error creating video: {e}")
            return False


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process UCSD Ped1/Ped2 dataset")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of UCSD dataset")
    parser.add_argument("--dataset_type", type=str, choices=["ped1", "ped2"], 
                       default="ped2", help="Dataset type: ped1 or ped2")
    parser.add_argument("--convert_to_video", action="store_true",
                       help="Convert image sequences to video files")
    parser.add_argument("--video_fps", type=int, default=10,
                       help="FPS for generated videos")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset root directory '{args.data_root}' does not exist!")
        return
    
    # Process dataset
    processor = UCSDProcessor(args.data_root, args.dataset_type)
    
    # Convert to video if requested
    if args.convert_to_video:
        processor.convert_sequences_to_videos(fps=args.video_fps)
    
    # Process and create JSON metadata
    processor.process()
    
    print(f"\nUCSD {args.dataset_type.upper()} dataset processing completed!")


if __name__ == "__main__":
    main()