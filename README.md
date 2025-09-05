# VideoAnomalyCLIP: Zero-shot Semantic Video Anomaly Detection

This repository contains the implementation of VideoAnomalyCLIP, an extension of AnomalyCLIP for zero-shot semantic video anomaly detection. The key innovation is the integration of temporal modeling capabilities and a dual-mode prompting strategy.

## Key Features

- **Temporal Modeling**: Integrates a Temporal Transformer module to capture temporal dependencies between video frames
- **Dual-Mode Prompting**: "Train-Generic, Test-Specific" strategy that learns generic normality patterns during training and uses semantic prompts during testing
- **Zero-Shot Transfer**: Train once on one dataset (e.g., ShanghaiTech) and test on another (e.g., UCSD) without retraining
- **Semantic Anomaly Detection**: Can detect specific types of anomalies using natural language descriptions

## Architecture Overview

The system extends the original AnomalyCLIP with:

1. **Temporal Transformer Module**: Added after CLIP visual encoder to model temporal relationships
2. **Video Processing Pipeline**: Handles video input with sliding window inference
3. **Dual Prompting System**: Generic prompts for training, semantic prompts for testing
4. **New Scoring Function**: Maximum similarity across semantic prompts

## Installation

```bash
git clone <repository-url>
cd VideoAnomalyCLIP
pip install -r requirements.txt
```

## Project Structure

```
VideoAnomalyCLIP/
├── AnomalyCLIP_lib/           # Original CLIP components
│   ├── models.py              # Extended with TemporalTransformer
│   └── ...
├── configs/                   # Configuration files
│   ├── shanghaitech_train.yaml
│   └── ucsd_test.yaml
├── video_utils.py            # Video processing utilities
├── dataset.py                # Modified for video data
├── prompt_ensemble.py        # Dual-mode prompting
├── train.py                  # Training script
├── test.py                   # Testing script
├── evaluate.py               # Independent evaluation
└── README.md
```

## Dataset Preparation

### ShanghaiTech Campus Dataset (Training)
```json
{
  "train": {
    "campus_scene": [
      {
        "video_path": "training/video_001.avi",
        "anomaly": 0,
        "gt_path": ""
      }
    ]
  }
}
```

### UCSD Ped2 Dataset (Testing) 
```json
{
  "test": {
    "pedestrian": [
      {
        "video_path": "Test/Test001.avi", 
        "anomaly": 1,
        "gt_path": "Test/Test001_gt.txt"
      }
    ]
  }
}
```

## Usage

### Training

Train on ShanghaiTech dataset to learn generic temporal patterns:

```bash
# Configure training in configs/shanghaitech_train.yaml
python train.py --config configs/shanghaitech_train.yaml
```

### Testing

Test on UCSD dataset using semantic prompts:

```bash  
# Configure testing in configs/ucsd_test.yaml
python test.py --config configs/ucsd_test.yaml
```

### Evaluation

Independent evaluation of results:

```bash
python evaluate.py \
    --gt_path /path/to/ucsd/ground_truth \
    --pred_path results/ucsd/frame_scores.npy \
    --dataset ucsd \
    --output_dir results/evaluation \
    --plot_curves
```

## Configuration

### Training Configuration (shanghaitech_train.yaml)
```yaml
# Model parameters
temporal_transformer_layers: 2
temporal_heads: 8
clip_length: 16
stride: 8

# Generic prompts for training
generic_prompts:
  normal: "a normal scene"
  abnormal: "an abnormal event"
```

### Testing Configuration (ucsd_test.yaml)
```yaml
# Semantic prompts for UCSD anomalies
semantic_prompts:
  - "a person riding a bicycle"
  - "a person on a skateboard"
  - "a small cart on a walkway"
  - "a person in a wheelchair"
```

## Key Modifications from Original AnomalyCLIP

### 1. Video Processing (`video_utils.py`, `dataset.py`)
- Added video reading and clip sampling
- Implemented sliding window inference
- Frame-level preprocessing and transforms

### 2. Temporal Modeling (`models.py`)
- **TemporalTransformer**: Self-attention over temporal dimension
- **VideoAnomalyCLIP**: Extended model with temporal capabilities
- Positional encoding for temporal positions

### 3. Dual-Mode Prompting (`prompt_ensemble.py`)
- **Training Phase**: Generic object-agnostic prompts
- **Testing Phase**: Specific semantic anomaly descriptions
- Learnable prompt embeddings with temporal consistency

### 4. New Scoring System
```python
# New anomaly scoring function
S(t) = max_{i=1...N}(cos_sim(V_t, T_i))
```
Where V_t is temporal-aware visual feature and T_i are semantic text features.

## Performance

The system achieves zero-shot transfer from ShanghaiTech to UCSD while maintaining competitive performance with domain-specific methods.

### Evaluation Metrics
- **Frame-level**: AUC, Average Precision
- **Video-level**: AUC, Average Precision  
- **Pixel-level**: AUC, AUPRO (when available)

## Dependencies

Key dependencies include:
- PyTorch 2.0+
- decord (efficient video reading)
- OpenCV (video processing fallback)
- scikit-learn (evaluation metrics)
- PyYAML (configuration files)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{videoanomaly2024,
  title={VideoAnomalyCLIP: Zero-shot Semantic Video Anomaly Detection},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This work builds upon the original AnomalyCLIP implementation and extends it for video anomaly detection with temporal modeling capabilities.

## License

This project follows the same license as the original AnomalyCLIP repository.