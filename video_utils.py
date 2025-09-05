"""
Video processing utilities for AnomalyCLIP video anomaly detection.
Provides functions for video reading, sampling, and frame preprocessing.
"""

import cv2
import numpy as np
import torch
import glob
import os
from decord import VideoReader, cpu
from PIL import Image
import random


def sample_video_clips(video_path, start_index, clip_length, stride=1):
    """
    Sample a video clip from a video file or image sequence.
    
    Args:
        video_path (str): Path to video file or image sequence directory
        start_index (int): Starting frame index
        clip_length (int): Number of frames to sample
        stride (int): Stride between frames (default: 1 for dense sampling)
    
    Returns:
        numpy.ndarray: Video clip of shape (T, H, W, C)
    """
    # Check if input is directory (image sequence) or video file
    if os.path.isdir(video_path):
        return sample_from_image_sequence(video_path, start_index, clip_length, stride)
    else:
        return sample_from_video_file(video_path, start_index, clip_length, stride)


def sample_from_image_sequence(sequence_dir, start_index, clip_length, stride=1):
    """
    Sample frames from an image sequence directory (like UCSD datasets).
    
    Args:
        sequence_dir (str): Directory containing image sequence
        start_index (int): Starting frame index
        clip_length (int): Number of frames to sample
        stride (int): Stride between frames
        
    Returns:
        numpy.ndarray: Video clip of shape (T, H, W, C)
    """
    # Get all image files and sort them
    image_extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(sequence_dir, ext)))
    
    image_files.sort()  # Ensure proper ordering
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {sequence_dir}")
    
    # Sample frames
    frames = []
    for i in range(clip_length):
        frame_idx = start_index + i * stride
        if frame_idx >= len(image_files):
            # If we exceed available frames, repeat the last frame
            frame_idx = len(image_files) - 1
        
        # Load image
        img_path = image_files[frame_idx]
        frame = cv2.imread(img_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
        else:
            # If image loading fails, create a black frame
            frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
    
    return np.stack(frames)


def sample_from_video_file(video_path, start_index, clip_length, stride=1):
    """
    Sample frames from a video file.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Ensure we don't exceed video length
        end_index = min(start_index + clip_length * stride, total_frames)
        indices = list(range(start_index, end_index, stride))
        
        # If we don't have enough frames, repeat the last frame
        while len(indices) < clip_length:
            indices.append(indices[-1])
        
        # Sample frames
        indices = indices[:clip_length]
        frames = vr.get_batch(indices).asnumpy()
        
        return frames
        
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        # Fallback to OpenCV
        return sample_video_clips_opencv(video_path, start_index, clip_length, stride)
    """
    Sample a video clip from a video file using decord for efficient reading.
    
    Args:
        video_path (str): Path to the video file
        start_index (int): Starting frame index
        clip_length (int): Number of frames to sample
        stride (int): Stride between frames (default: 1 for dense sampling)
    
    Returns:
        numpy.ndarray: Video clip of shape (T, H, W, C)
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Ensure we don't exceed video length
        end_index = min(start_index + clip_length * stride, total_frames)
        indices = list(range(start_index, end_index, stride))
        
        # If we don't have enough frames, repeat the last frame
        while len(indices) < clip_length:
            indices.append(indices[-1])
        
        # Sample frames
        indices = indices[:clip_length]
        frames = vr.get_batch(indices).asnumpy()
        
        return frames
        
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        # Fallback to OpenCV
        return sample_video_clips_opencv(video_path, start_index, clip_length, stride)


def sample_video_clips_opencv(video_path, start_index, clip_length, stride=1):
    """
    Fallback video sampling using OpenCV.
    
    Args:
        video_path (str): Path to the video file
        start_index (int): Starting frame index
        clip_length (int): Number of frames to sample
        stride (int): Stride between frames
    
    Returns:
        numpy.ndarray: Video clip of shape (T, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        
        frame_count = 0
        while len(frames) < clip_length:
            ret, frame = cap.read()
            if not ret:
                # If we reach end of video, repeat last frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create a black frame if no frames read
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                if frame_count % stride == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                frame_count += 1
                
    finally:
        cap.release()
    
    return np.stack(frames)


def apply_frame_transforms(video_clip, transform):
    """
    Apply image transforms to each frame in a video clip.
    
    Args:
        video_clip (numpy.ndarray): Video clip of shape (T, H, W, C)
        transform: torchvision transform to apply
    
    Returns:
        torch.Tensor: Transformed video clip of shape (T, C, H, W)
    """
    transformed_frames = []
    
    for frame in video_clip:
        # Convert numpy array to PIL Image
        pil_frame = Image.fromarray(frame.astype(np.uint8))
        # Apply transform
        transformed_frame = transform(pil_frame)
        transformed_frames.append(transformed_frame)
    
    # Stack frames: (T, C, H, W)
    return torch.stack(transformed_frames)


def get_video_info(video_path):
    """
    Get basic information about a video file.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Video information including total_frames, fps, duration
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        info = {
            'total_frames': len(vr),
            'fps': vr.get_avg_fps(),
            'duration': len(vr) / vr.get_avg_fps()
        }
        return info
    except:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0
        }


def create_sliding_windows(total_frames, clip_length, stride=None):
    """
    Create sliding window indices for video processing.
    
    Args:
        total_frames (int): Total number of frames in video
        clip_length (int): Length of each clip
        stride (int): Stride between windows (default: clip_length//2)
    
    Returns:
        list: List of (start_index, end_index) tuples
    """
    if stride is None:
        stride = clip_length // 2
    
    windows = []
    start = 0
    
    while start + clip_length <= total_frames:
        windows.append((start, start + clip_length))
        start += stride
    
    # Add final window if needed
    if windows and windows[-1][1] < total_frames:
        windows.append((total_frames - clip_length, total_frames))
    elif not windows:
        # Video shorter than clip_length, use entire video
        windows.append((0, total_frames))
    
    return windows