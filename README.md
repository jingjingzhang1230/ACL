# Hierarchical Multimodal ACL Severity Classification from Knee MRI

This repository contains the implementation of a hierarchical multimodal framework for anterior cruciate ligament (ACL) severity classification from knee MRI. The method combines MRI image features, ROI mask features, and radiomic features within a clinically motivated hierarchical classification structure.

## Overview

ACL severity assessment is formulated as two related binary tasks rather than a flat three-class classification problem:

1. **Head 1:** Healthy vs Injured  
2. **Head 2:** Partial Tear vs Complete Tear, evaluated only for injured cases

The framework integrates three sources of information:

- **MRI volume**
- **Predicted ACL ROI mask**
- **Radiomic features extracted from the ACL ROI**

These features are encoded separately, fused into a shared representation, and used for hierarchical prediction.

## Repository Structure

- `model_with_mask_rad_hier.py`  
  Main training script for the hierarchical multimodal classification model.

- `evaluate.py`  
  Evaluation script for the trained hierarchical model using 5-fold drop-one-subset jackknife evaluation.

- `acl_dataloader.py`  
  Custom PyTorch dataset for loading 3D MRI volumes, radiomic features, and ROI masks.

- `earlystopping.py`  
  Early stopping utility for training.

- `metrics_tracker.py`  
  Utility for tracking, printing, and saving training and evaluation metrics.

## Method Summary

### Input Modalities
- **MRI image:** 3D knee MRI volume
- **Mask:** predicted binary ACL segmentation mask
- **Radiomics:** pre-computed radiomic descriptors extracted from the ACL ROI

### Model Design
- 3D image encoder with selectable backbone:
  - ResNet
  - EfficientNet
  - DenseNet
  - Inception I3D
- Lightweight 3D CNN mask encoder
- MLP radiomics encoder
- Fusion MLP
- Two binary classification heads:
  - Healthy vs Injured
  - Partial vs Complete tear

### Loss
Training uses class-weighted binary cross-entropy objectives for the two heads, combined with either:
- **uncertainty-based weighting**, or
- **equal weighting**

## Requirements
Suggested environment:
- Python 3.10
- PyTorch 2.6.0
- CUDA 12.4
- torchvision
- MONAI
- nibabel
- pandas
- numpy
- scikit-learn
- tqdm

You may install the core dependencies with:

```bash
pip install torch torchvision monai nibabel pandas numpy scikit-learn tqdm
