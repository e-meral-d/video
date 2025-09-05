"""
Modified dataset.py for video anomaly detection.
Handles video data loading, clip sampling, and preprocessing.
"""

import torch.utils.data as data
import json
import random
import numpy as np
import torch
import os
from video_utils import sample_video_clips, apply_frame_transforms, get_video_info, create_sliding_windows


def generate_class_info(dataset_name):
    """Generate class information for different video datasets."""
    class_name_map_class_id = {}
    
    # Video anomaly detection datasets
    if dataset_name == 'shanghaitech':
        obj_list = ['campus_scene']  # Single class for normal campus scenes
    elif dataset_name == 'ucsd':
        obj_list = ['pedestrian']  # Single class for pedestrian scenes
    elif dataset_name == 'avenue':
        obj_list = ['street_scene']  # Avenue dataset
    elif dataset_name == 'ubi':
        obj_list = ['normal_activity']  # UBnormal dataset
    else:
        # Default single class
        obj_list = ['scene']
    
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index
    
    return obj_list, class_name_map_class_id


class VideoDataset(data.Dataset):
    """
    Video dataset for anomaly detection.
    Samples video clips from training/testing videos.
    """
    
    def __init__(self, root, transform, target_transform, dataset_name, 
                 mode='test', clip_length=16, stride=8):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_name = dataset_name
        self.mode = mode
        self.clip_length = clip_length
        self.stride = stride
        
        # Load metadata
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[mode]
        
        self.cls_names = list(meta_info.keys())
        self.data_all = []
        
        # Process video files and create clip samples
        for cls_name in self.cls_names:
            videos = meta_info[cls_name]
            for video_info in videos:
                video_path = video_info['video_path']
                full_video_path = os.path.join(self.root, video_path)
                
                # Get video information
                try:
                    vid_info = get_video_info(full_video_path)
                    total_frames = vid_info['total_frames']
                    
                    if mode == 'train':
                        # For training, sample clips with stride
                        clip_starts = list(range(0, max(1, total_frames - clip_length), stride))
                        if not clip_starts:
                            clip_starts = [0]
                    else:
                        # For testing, use sliding windows for complete coverage
                        windows = create_sliding_windows(total_frames, clip_length, stride)
                        clip_starts = [w[0] for w in windows]
                    
                    # Create data entries for each clip
                    for start_frame in clip_starts:
                        clip_info = {
                            'video_path': video_path,
                            'start_frame': start_frame,
                            'cls_name': cls_name,
                            'anomaly': video_info.get('anomaly', 0),
                            'gt_path': video_info.get('gt_path', ''),
                            'total_frames': total_frames
                        }
                        self.data_all.append(clip_info)
                        
                except Exception as e:
                    print(f"Error processing video {full_video_path}: {e}")
                    continue
        
        self.length = len(self.data_all)
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        
        print(f"Loaded {self.length} video clips for {mode} mode")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.data_all[index]
        video_path = data['video_path']
        start_frame = data['start_frame']
        cls_name = data['cls_name']
        anomaly = data['anomaly']
        
        # Load video clip
        full_video_path = os.path.join(self.root, video_path)
        try:
            # Sample video clip: shape (T, H, W, C)
            video_clip = sample_video_clips(full_video_path, start_frame, self.clip_length)
            
            # Apply transforms to each frame
            if self.transform is not None:
                video_tensor = apply_frame_transforms(video_clip, self.transform)
            else:
                # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
                video_tensor = torch.from_numpy(video_clip).float().permute(0, 3, 1, 2)
            
            # For ground truth masks (if available)
            if anomaly == 1 and data['gt_path']:
                # Load ground truth frames corresponding to this clip
                gt_path = os.path.join(self.root, data['gt_path'])
                gt_mask = self._load_gt_frames(gt_path, start_frame, self.clip_length)
            else:
                # Create dummy ground truth (all zeros for normal)
                H, W = video_tensor.shape[2], video_tensor.shape[3]
                gt_mask = torch.zeros(self.clip_length, H, W)
            
            return {
                'video': video_tensor,  # Shape: (T, C, H, W)
                'gt_mask': gt_mask,     # Shape: (T, H, W)
                'cls_name': cls_name,
                'anomaly': anomaly,
                'video_path': full_video_path,
                'cls_id': self.class_name_map_class_id[cls_name],
                'start_frame': start_frame
            }
            
        except Exception as e:
            print(f"Error loading video clip {full_video_path}: {e}")
            # Return dummy data
            return self._get_dummy_item(cls_name, anomaly)
    
    def _load_gt_frames(self, gt_path, start_frame, clip_length):
        """Load ground truth mask frames."""
        try:
            if os.path.isfile(gt_path):
                # Single ground truth file (frame-level labels)
                gt_data = np.load(gt_path)  # or load text file
                gt_frames = gt_data[start_frame:start_frame + clip_length]
                return torch.from_numpy(gt_frames).float()
            else:
                # Directory with individual mask files
                gt_frames = []
                for i in range(clip_length):
                    frame_idx = start_frame + i
                    mask_file = os.path.join(gt_path, f"{frame_idx:06d}.png")
                    if os.path.exists(mask_file):
                        from PIL import Image
                        mask = np.array(Image.open(mask_file).convert('L'))
                        mask = (mask > 127).astype(np.float32)  # Binarize
                        gt_frames.append(mask)
                    else:
                        # Use last available mask or zeros
                        if gt_frames:
                            gt_frames.append(gt_frames[-1])
                        else:
                            gt_frames.append(np.zeros((240, 360)))  # Default size
                
                return torch.from_numpy(np.stack(gt_frames))
        except:
            # Return zeros if GT loading fails
            return torch.zeros(clip_length, 240, 360)
    
    def _get_dummy_item(self, cls_name, anomaly):
        """Return dummy data when video loading fails."""
        return {
            'video': torch.zeros(self.clip_length, 3, 224, 224),
            'gt_mask': torch.zeros(self.clip_length, 224, 224),
            'cls_name': cls_name,
            'anomaly': anomaly,
            'video_path': '',
            'cls_id': 0,
            'start_frame': 0
        }


# Keep the original Dataset class for backward compatibility with image datasets
class ImageDataset(data.Dataset):
    """Original image dataset class."""
    
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[mode]
        
        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)
        
        # Use original generate_class_info for image datasets
        if dataset_name == 'mvtec':
            obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                        'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        elif dataset_name == 'visa':
            obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        else:
            obj_list = ['object']
        
        self.obj_list = obj_list
        self.class_name_map_class_id = {k: i for i, k in enumerate(obj_list)}
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = (
            data['img_path'], data['mask_path'], data['cls_name'], 
            data['specie_name'], data['anomaly']
        )
        
        from PIL import Image
        img = Image.open(os.path.join(self.root, img_path))
        
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        
        return {
            'img': img, 
            'img_mask': img_mask, 
            'cls_name': cls_name, 
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path), 
            'cls_id': self.class_name_map_class_id[cls_name]
        }


# Factory function to create appropriate dataset
def create_dataset(root, transform, target_transform, dataset_name, mode='test', **kwargs):
    """
    Factory function to create appropriate dataset based on dataset type.
    """
    video_datasets = ['shanghaitech', 'ucsd', 'avenue', 'ubi']
    
    if dataset_name.lower() in video_datasets:
        return VideoDataset(
            root, transform, target_transform, dataset_name, mode, 
            clip_length=kwargs.get('clip_length', 16),
            stride=kwargs.get('stride', 8)
        )
    else:
        return ImageDataset(root, transform, target_transform, dataset_name, mode)


# Keep backward compatibility
Dataset = ImageDataset