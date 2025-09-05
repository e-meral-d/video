"""
Modified models.py to add TemporalTransformer for video processing.
Integrates temporal modeling capabilities into the AnomalyCLIP architecture.
"""

import torch
import torch.nn as nn
import math
from .AnomalyCLIP import *  # Import existing components


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer module for modeling temporal dependencies in video sequences.
    Uses self-attention to capture long-range temporal relationships between frames.
    """
    
    def __init__(self, embed_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Positional encoding for temporal positions
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_len=64)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass of temporal transformer.
        
        Args:
            x: Input tensor of shape (B, T, Feature_Dim)
            
        Returns:
            Output tensor of shape (B, T, Feature_Dim) with temporal context
        """
        B, T, D = x.shape
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        # Input: (B, T, D), Output: (B, T, D)
        temporal_features = self.transformer_encoder(x)
        
        # Apply layer normalization
        temporal_features = self.layer_norm(temporal_features)
        
        return temporal_features


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, D)
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class VideoAnomalyCLIP(AnomalyCLIP):
    """
    Extended AnomalyCLIP model with temporal modeling capabilities for video anomaly detection.
    Integrates TemporalTransformer after CLIP visual encoder.
    """
    
    def __init__(self, *args, temporal_layers=2, temporal_heads=8, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add temporal transformer module
        # Assuming visual feature dimension is the same as embed_dim
        visual_dim = self.visual.output_dim if hasattr(self.visual, 'output_dim') else self.transformer.width
        
        self.temporal_transformer = TemporalTransformer(
            embed_dim=visual_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads
        )
        
        # Store temporal modeling parameters
        self.temporal_layers = temporal_layers
        self.temporal_heads = temporal_heads
        
    def encode_video(self, video, feature_list=[], ori_patch=False, proj_use=True, DPAM_layer=None, ffn=False):
        """
        Encode video input with temporal modeling.
        
        Args:
            video: Input video tensor of shape (B, T, C, H, W)
            feature_list: List of layer indices to extract features from
            ori_patch: Whether to use original patch features
            proj_use: Whether to use projection
            DPAM_layer: DPAM layer specification
            ffn: Feed-forward network flag
            
        Returns:
            Tuple of (temporal_features, patch_features_list)
        """
        B, T, C, H, W = video.shape
        
        # Reshape for CLIP processing: (B*T, C, H, W)
        video_flat = video.view(B * T, C, H, W)
        
        # Extract features from CLIP visual encoder
        visual_features, patch_features = self.visual(
            video_flat.type(self.dtype), 
            feature_list, 
            ori_patch=ori_patch, 
            proj_use=proj_use, 
            DPAM_layer=DPAM_layer, 
            ffn=ffn
        )
        
        # Reshape visual features back to video format: (B, T, Feature_Dim)
        if len(visual_features.shape) == 2:
            # Global features: (B*T, D) -> (B, T, D)
            feature_dim = visual_features.shape[1]
            visual_features = visual_features.view(B, T, feature_dim)
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(visual_features)
        
        # Process patch features if available
        processed_patch_features = []
        if patch_features:
            for patch_feature in patch_features:
                if len(patch_feature.shape) == 3:
                    # Patch features: (B*T, N_patches, D)
                    BT, N, D = patch_feature.shape
                    # Reshape to (B, T, N_patches, D)
                    patch_feature = patch_feature.view(B, T, N, D)
                    
                    # Apply temporal modeling to each patch
                    patch_feature_flat = patch_feature.view(B * N, T, D)
                    temporal_patch = self.temporal_transformer(patch_feature_flat)
                    temporal_patch = temporal_patch.view(B, T, N, D)
                    
                    # Flatten back for compatibility: (B*T, N_patches, D)
                    temporal_patch = temporal_patch.view(BT, N, D)
                    processed_patch_features.append(temporal_patch)
                else:
                    processed_patch_features.append(patch_feature)
        
        return temporal_features, processed_patch_features
    
    def forward_video(self, video, text):
        """
        Forward pass for video input.
        
        Args:
            video: Input video tensor of shape (B, T, C, H, W)
            text: Text input
            
        Returns:
            Tuple of (logits_per_video, logits_per_text)
        """
        # Encode video with temporal modeling
        video_features, _ = self.encode_video(video)
        
        # Use mean pooling across temporal dimension for classification
        video_features = video_features.mean(dim=1)  # (B, T, D) -> (B, D)
        
        # Encode text
        text_features = self.encode_text(text)
        
        # Normalize features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()
        
        return logits_per_video, logits_per_text


def build_video_anomaly_clip(state_dict, design_details=None, temporal_layers=2, temporal_heads=8):
    """
    Build VideoAnomalyCLIP model with temporal modeling capabilities.
    
    Args:
        state_dict: Pre-trained CLIP state dict
        design_details: Design configuration
        temporal_layers: Number of temporal transformer layers
        temporal_heads: Number of attention heads in temporal transformer
        
    Returns:
        VideoAnomalyCLIP model instance
    """
    # Extract model configuration from state_dict (same as original build_model)
    vit = "visual.proj" in state_dict
    
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    # Create VideoAnomalyCLIP model
    model = VideoAnomalyCLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        design_details=design_details,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads
    )
    
    # Remove keys not in the original model
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # Load pre-trained weights (temporal transformer will be randomly initialized)
    incompatible = model.load_state_dict(state_dict, strict=False)
    # print(f"Loaded pre-trained CLIP weights. Missing keys: {incompatible.missing_keys}")  # Commented out to reduce output noise
    
    return model.eval()