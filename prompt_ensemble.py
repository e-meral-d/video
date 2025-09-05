"""
Modified prompt_ensemble.py for dual-mode prompting strategy.
Implements train-generic, test-specific prompting for video anomaly detection.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def tokenize(texts, context_length=77, truncate=False):
    """Tokenize text inputs."""
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class VideoAnomalyCLIP_PromptLearner(nn.Module):
    """
    Prompt learner for video anomaly detection with dual-mode prompting strategy.
    Supports both generic prompts for training and semantic prompts for testing.
    """
    
    def __init__(self, clip_model, design_details):
        super().__init__()
        
        # Basic setup
        classnames = ["scene"]  # Generic class for video scenes
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        
        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.classnames = classnames
        
        # Generic prompts for training (object-agnostic)
        self.generic_normal_prompts = [
            "a normal scene",
            "normal activity", 
            "regular behavior",
            "typical situation"
        ]
        
        self.generic_abnormal_prompts = [
            "an abnormal event",
            "unusual activity",
            "irregular behavior", 
            "atypical situation"
        ]
        
        self.normal_num = len(self.generic_normal_prompts)
        self.abnormal_num = len(self.generic_abnormal_prompts)
        
        # Initialize learnable context vectors
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        
        # Random initialization of context vectors
        ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.n_cls, self.abnormal_num, n_ctx_neg, ctx_dim, dtype=dtype)
        
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # Learnable positive contexts
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # Learnable negative contexts
        
        # Deep compound prompts for enhanced text modeling
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        # Projection layers for compound prompts
        single_layer = nn.Linear(ctx_dim, ctx_dim)  # Keep same dimension
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        
        # Create tokenized prompts and embeddings
        self._initialize_prompt_embeddings(clip_model, ctx_dim, dtype, n_ctx_pos, n_ctx_neg)
    
    def _initialize_prompt_embeddings(self, clip_model, ctx_dim, dtype, n_ctx_pos, n_ctx_neg):
        """Initialize prompt embeddings and tokens."""
        
        # Create prompt templates
        prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
        prompt_prefix_neg = " ".join(["X"] * n_ctx_neg) 
        
        # Generate full prompts
        prompts_pos = [
            prompt_prefix_pos + " " + template + "."
            for template in self.generic_normal_prompts
            for name in self.classnames
        ]
        prompts_neg = [
            prompt_prefix_neg + " " + template + "."
            for template in self.generic_abnormal_prompts  
            for name in self.classnames
        ]
        
        # Tokenize prompts
        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        
        # Get embeddings
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            
            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.abnormal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
        
        # Register buffers for prompt components
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])
        
        # Reshape tokenized prompts
        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)
        
        n, d = tokenized_prompts_neg.shape  
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.abnormal_num, self.n_cls, d).permute(1, 0, 2)
        
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
    
    def forward(self, cls_id=None):
        """
        Forward pass to generate learnable prompts.
        
        Args:
            cls_id: Class ID (for compatibility, not used in video case)
            
        Returns:
            Tuple of (prompts, tokenized_prompts, compound_prompts_text)
        """
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg
        
        # Construct positive prompts
        prompts_pos = torch.cat([
            prefix_pos,  # (n_cls, n_normal, 1, dim)
            ctx_pos,     # (n_cls, n_normal, n_ctx, dim)  
            suffix_pos   # (n_cls, n_normal, *, dim)
        ], dim=2)
        
        # Construct negative prompts
        prompts_neg = torch.cat([
            prefix_neg,  # (n_cls, n_abnormal, 1, dim)
            ctx_neg,     # (n_cls, n_abnormal, n_ctx, dim)
            suffix_neg   # (n_cls, n_abnormal, *, dim)
        ], dim=2)
        
        # Reshape for output
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        
        # Concatenate positive and negative prompts
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)
        
        # Handle tokenized prompts
        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)
        
        return prompts, tokenized_prompts, self.compound_prompts_text
    
    def forward_semantic(self, semantic_prompts, clip_model):
        """
        Forward pass for semantic prompts during testing.
        
        Args:
            semantic_prompts: List of semantic prompt strings
            clip_model: CLIP model for encoding
            
        Returns:
            Encoded semantic prompt features
        """
        # Tokenize semantic prompts
        tokenized_semantic = tokenize(semantic_prompts)
        
        # Encode using CLIP text encoder
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_semantic.to(next(self.parameters()).device))
        
        return text_features


def encode_semantic_prompts(clip_model, semantic_prompts, device):
    """
    Utility function to encode semantic prompts for testing.
    
    Args:
        clip_model: CLIP model
        semantic_prompts: List of semantic prompt strings
        device: Device to run on
        
    Returns:
        Encoded semantic prompt features
    """
    # Add context to prompts for better performance
    contextualized_prompts = []
    for prompt in semantic_prompts:
        if not prompt.startswith("a "):
            prompt = "a " + prompt
        if not prompt.endswith("."):
            prompt = prompt + "."
        contextualized_prompts.append(prompt)
    
    # Tokenize
    tokenized = tokenize(contextualized_prompts).to(device)
    
    # Encode
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features


def create_generic_prompts():
    """
    Create generic prompts for training phase.
    These are object-agnostic and focus on normality/abnormality patterns.
    """
    normal_prompts = [
        "a normal scene",
        "normal activity in the area", 
        "regular pedestrian behavior",
        "typical daily activity",
        "ordinary scene situation",
        "usual environment activity"
    ]
    
    abnormal_prompts = [
        "an abnormal event", 
        "unusual activity in the area",
        "irregular pedestrian behavior", 
        "atypical daily activity",
        "extraordinary scene situation",
        "unusual environment activity"
    ]
    
    return normal_prompts, abnormal_prompts


def create_ucsd_semantic_prompts():
    """
    Create semantic prompts specific to UCSD Ped2 dataset anomalies.
    These are used during testing phase.
    """
    semantic_prompts = [
        "a person riding a bicycle",
        "a person on a skateboard",
        "a small cart on a walkway", 
        "a person in a wheelchair",
        "a vehicle on the pedestrian area",
        "people running in the scene",
        "people fighting or struggling",
        "unusual objects in the walkway",
        "emergency vehicles or situations"
    ]
    
    return semantic_prompts


def create_shanghaitech_semantic_prompts():
    """
    Create semantic prompts for ShanghaiTech dataset.
    """
    semantic_prompts = [
        "people running in panic",
        "people fighting or violence", 
        "vehicles in pedestrian areas",
        "people jumping or falling",
        "emergency situations",
        "unusual crowd behavior",
        "objects thrown or dropped",
        "people climbing structures",
        "accidents or collisions"
    ]
    
    return semantic_prompts