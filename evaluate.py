#!/usr/bin/env python3
"""
evaluate.py – Drop-one-subset jackknife evaluation for the hierarchical
two-head ACL classification model.

Usage:
    python evaluate.py --backbone efficientnet

The script:
  1. Reproduces the same test split used during training (random_state=42).
  2. Loads the saved checkpoint from checkpoints_dir.
  3. Runs 5-fold stratified jackknife evaluation on the test set.
  4. Reports mean ± std for every metric.
  5. Saves results to results/<backbone>/jackknife_evaluation/.
"""

import os
import json
import argparse
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support,
)

from torchvision.models.video import r3d_18, R3D_18_Weights
from efficientnet_pytorch_3d import EfficientNet3D
from monai.networks.nets import DenseNet169
from pytorch_i3d import InceptionI3d
from acl_dataloader import KneeMRI917Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions  (mirrored exactly from model_with_mask_rad.py)
# ─────────────────────────────────────────────────────────────────────────────

class InceptionI3dWrapper(nn.Module):
    """Extracts a flat [B, 1024] feature vector from I3D."""
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d(
            num_classes=400,
            spatial_squeeze=True,
            final_endpoint='Logits',
            in_channels=3,
            dropout_keep_prob=0.5,
        )
        self.i3d.logits  = nn.Identity()
        self.i3d.dropout = nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.i3d.Conv3d_1a_7x7(x)
        x = self.i3d.MaxPool3d_2a_3x3(x)
        x = self.i3d.Conv3d_2b_1x1(x)
        x = self.i3d.Conv3d_2c_3x3(x)
        x = self.i3d.MaxPool3d_3a_3x3(x)
        x = self.i3d.Mixed_3b(x);  x = self.i3d.Mixed_3c(x)
        x = self.i3d.MaxPool3d_4a_3x3(x)
        x = self.i3d.Mixed_4b(x);  x = self.i3d.Mixed_4c(x)
        x = self.i3d.Mixed_4d(x);  x = self.i3d.Mixed_4e(x);  x = self.i3d.Mixed_4f(x)
        x = self.i3d.MaxPool3d_5a_2x2(x)
        x = self.i3d.Mixed_5b(x);  x = self.i3d.Mixed_5c(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone_name='efficientnet',
        radiomics_dim=107,
        fusion_dim=256,
        radiomics_encoded_dim=128,
        mask_encoded_dim=128,
    ):
        super().__init__()

        if backbone_name == 'resnet':
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'efficientnet':
            try:
                self.backbone = EfficientNet3D.from_pretrained("efficientnet-b0", in_channels=3)
            except Exception as e:
                print(f"[WARN] Pretrained EfficientNet3D unavailable ({e}). Using random init.")
                self.backbone = EfficientNet3D.from_name("efficientnet-b0", in_channels=3)
            feature_dim = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()

        elif backbone_name == 'densenet':
            feature_dim = 1024
            self.backbone = DenseNet169(
                spatial_dims=3,
                in_channels=3,
                out_channels=feature_dim,
            )

        elif backbone_name == 'inception':
            self.backbone = InceptionI3dWrapper()
            feature_dim = 1024

        else:
            raise ValueError(f"Unknown backbone '{backbone_name}'.")

        self.radiomics_encoder = nn.Sequential(
            nn.Linear(radiomics_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, radiomics_encoded_dim),
            nn.LayerNorm(radiomics_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.mask_fc = nn.Sequential(
            nn.Linear(128, mask_encoded_dim),
            nn.LayerNorm(mask_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + radiomics_encoded_dim + mask_encoded_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.head1 = nn.Linear(fusion_dim, 2)   # healthy vs injured
        self.head2 = nn.Linear(fusion_dim, 2)   # partial vs complete

        # Required to match checkpoints saved by model_with_mask_rad_hier.py
        import math
        self.log_sigma_a = nn.Parameter(torch.tensor([-math.log(0.3)], dtype=torch.float32))
        self.log_sigma_b = nn.Parameter(torch.tensor([-math.log(1.0)], dtype=torch.float32))

    def forward(self, x, radiomics_feature, mask):
        image_feat = self.backbone(x)
        rad_feat   = self.radiomics_encoder(radiomics_feature.float())
        mask_feat  = self.mask_fc(self.mask_encoder(mask))
        h = torch.cat([image_feat, rad_feat, mask_feat], dim=1)
        h = self.fusion(h)
        return self.head1(h), self.head2(h)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """
    Run model inference over a DataLoader.

    Returns
    -------
    labels   : np.ndarray (N,)    original 3-class labels {0, 1, 2}
    probs_h1 : np.ndarray (N, 2)  softmax probs from head 1
    probs_h2 : np.ndarray (N, 2)  softmax probs from head 2 (all samples)
    """
    model.eval()
    all_labels, all_probs_h1, all_probs_h2 = [], [], []

    with torch.no_grad():
        for X, r, m, y in tqdm(loader, desc="  Inference", leave=False):
            X, r, m = X.to(device), r.to(device), m.to(device)
            logits1, logits2 = model(X, r, m)
            all_probs_h1.append(F.softmax(logits1, dim=1).cpu().numpy())
            all_probs_h2.append(F.softmax(logits2, dim=1).cpu().numpy())
            all_labels.append(y.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs_h1),
        np.concatenate(all_probs_h2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical prediction construction
# ─────────────────────────────────────────────────────────────────────────────

def build_hierarchical_predictions(probs_h1, probs_h2):
    """
    Derive final 3-class predictions and probabilities from head outputs.

    Rules
    -----
    head1 predicts healthy  → final class = 0
    head1 injured + head2 partial   → final class = 1
    head1 injured + head2 complete  → final class = 2

    Probability construction
    ------------------------
    P(0) = P_h1(healthy)
    P(1) = P_h1(injured) * P_h2(partial)
    P(2) = P_h1(injured) * P_h2(complete)

    Returns
    -------
    preds_3class : (N,)    final 3-class predictions
    probs_3class : (N, 3)  3-class probability estimates
    preds_h1     : (N,)    head-1 predictions  {0=healthy, 1=injured}
    preds_h2     : (N,)    head-2 predictions  {0=partial, 1=complete}
    """
    preds_h1 = probs_h1.argmax(axis=1)
    preds_h2 = probs_h2.argmax(axis=1)

    preds_3class = np.where(preds_h1 == 0, 0, preds_h2 + 1)

    p_healthy  = probs_h1[:, 0]
    p_partial  = probs_h1[:, 1] * probs_h2[:, 0]
    p_complete = probs_h1[:, 1] * probs_h2[:, 1]
    probs_3class = np.stack([p_healthy, p_partial, p_complete], axis=1)

    return preds_3class, probs_3class, preds_h1, preds_h2


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _specificity_from_cm(cm, n_classes):
    """Per-class specificity (one-vs-rest) from a confusion matrix."""
    specs = []
    for c in range(n_classes):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        specs.append(float(tn / (tn + fp + 1e-8)))
    return specs


def compute_overall_metrics(y_true, preds, probs):
    """Overall 3-class metrics (Situation 1)."""
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
    try:
        auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return {
        'accuracy':          float(accuracy_score(y_true, preds)),
        'macro_f1':          float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, preds)),
        'macro_precision':   float(precision_score(y_true, preds, average='macro', zero_division=0)),
        'macro_recall':      float(recall_score(y_true, preds, average='macro', zero_division=0)),
        'auc_ovr':           float(auc),
        'confusion_matrix':  cm,
        'specificity':       _specificity_from_cm(cm, 3),
    }


def compute_per_class_metrics(y_true, preds):
    """Per-class precision / recall / F1 / support / specificity (Situation 2)."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
    return {
        'precision_per_class':   prec.tolist(),
        'recall_per_class':      rec.tolist(),
        'f1_per_class':          f1.tolist(),
        'support_per_class':     support.tolist(),
        'specificity_per_class': _specificity_from_cm(cm, 3),
    }


def compute_head1_metrics(y_true, preds_h1, probs_h1):
    """Binary head-1 metrics: healthy (0) vs injured (1) (Situation 3)."""
    y_h1          = (y_true > 0).astype(int)
    probs_injured = probs_h1[:, 1]
    cm = confusion_matrix(y_h1, preds_h1, labels=[0, 1])
    try:
        auc = float(roc_auc_score(y_h1, probs_injured))
    except ValueError:
        auc = float('nan')
    return {
        'accuracy':         float(accuracy_score(y_h1, preds_h1)),
        'f1':               float(f1_score(y_h1, preds_h1, average='binary', zero_division=0)),
        'precision':        float(precision_score(y_h1, preds_h1, average='binary', zero_division=0)),
        'recall':           float(recall_score(y_h1, preds_h1, average='binary', zero_division=0)),
        'auc':              auc,
        'confusion_matrix': cm,
    }


def compute_head2_metrics(y_true_injured, preds_h2_injured, probs_h2_injured):
    """
    Binary head-2 metrics: partial (0) vs complete (1).
    Evaluated on injured samples only (original labels 1 and 2).
    (Situation 4)
    """
    if len(y_true_injured) == 0:
        return {
            'accuracy': float('nan'), 'f1': float('nan'),
            'precision': float('nan'), 'recall': float('nan'),
            'auc': float('nan'), 'confusion_matrix': None,
        }

    y_h2           = (y_true_injured - 1).astype(int)   # {1,2} → {0,1}
    probs_complete = probs_h2_injured[:, 1]
    cm = confusion_matrix(y_h2, preds_h2_injured, labels=[0, 1])
    try:
        auc = float(roc_auc_score(y_h2, probs_complete))
    except ValueError:
        auc = float('nan')
    return {
        'accuracy':         float(accuracy_score(y_h2, preds_h2_injured)),
        'f1':               float(f1_score(y_h2, preds_h2_injured, average='binary', zero_division=0)),
        'precision':        float(precision_score(y_h2, preds_h2_injured, average='binary', zero_division=0)),
        'recall':           float(recall_score(y_h2, preds_h2_injured, average='binary', zero_division=0)),
        'auc':              auc,
        'confusion_matrix': cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

_EXCLUDE_KEYS = {
    'confusion_matrix', 'specificity', 'specificity_per_class',
    'precision_per_class', 'recall_per_class', 'f1_per_class', 'support_per_class',
}


def _aggregate(runs, key):
    """Collect `key` from run dicts; return (mean, std), skipping NaN entries."""
    vals = []
    for r in runs:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
            if not np.isnan(fv):
                vals.append(fv)
        except (TypeError, ValueError):
            pass
    if not vals:
        return float('nan'), float('nan')
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def summarize_runs(runs):
    """Compute mean ± std for all scalar metrics across jackknife runs."""
    if not runs:
        return {}
    scalar_keys = [k for k in runs[0] if k not in _EXCLUDE_KEYS]
    summary = {}
    for key in scalar_keys:
        mean, std = _aggregate(runs, key)
        summary[key] = {'mean': mean, 'std': std}
    return summary


def summarize_list_metric(runs, key):
    """
    Aggregate a per-class list metric (e.g. 'f1_per_class') across runs.
    Returns {'mean': [...], 'std': [...]}.
    """
    vals = np.array([r[key] for r in runs if r.get(key) is not None], dtype=float)
    if vals.ndim < 2 or len(vals) == 0:
        return {'mean': [], 'std': []}
    return {
        'mean': vals.mean(axis=0).tolist(),
        'std':  vals.std(axis=0, ddof=1).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _convert(obj):
    """Recursively convert numpy types to Python natives for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(i) for i in obj]
    return obj


def save_results_json(results, path):
    with open(path, 'w') as f:
        json.dump(_convert(results), f, indent=2)


def save_results_csv(results, path):
    """Flatten mean ± std metrics to CSV rows."""
    rows = []
    for section, data in results.items():
        if not isinstance(data, dict):
            continue
        for metric, value in data.items():
            if isinstance(value, dict) and 'mean' in value and 'std' in value:
                mean_v = value['mean']
                std_v  = value['std']
                # Per-class list metrics: expand per class
                if isinstance(mean_v, list):
                    for i, (m, s) in enumerate(zip(mean_v, std_v)):
                        rows.append({
                            'section': section,
                            'metric':  f'{metric}[{i}]',
                            'mean':    m,
                            'std':     s,
                        })
                else:
                    rows.append({
                        'section': section,
                        'metric':  metric,
                        'mean':    mean_v,
                        'std':     std_v,
                    })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['section', 'metric', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(title, summary):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    for k, v in summary.items():
        if not (isinstance(v, dict) and 'mean' in v):
            continue
        mean, std = v['mean'], v['std']
        mean_v = v['mean']
        std_v  = v['std']
        if isinstance(mean_v, list):
            for i, (m, s) in enumerate(zip(mean_v, std_v)):
                tag = f"{k}[{i}]"
                if np.isnan(m):
                    print(f"  {tag:<32s} = N/A")
                else:
                    print(f"  {tag:<32s} = {m:.4f} ± {s:.4f}")
        else:
            if np.isnan(mean_v):
                print(f"  {k:<32s} = N/A")
            else:
                print(f"  {k:<32s} = {mean_v:.4f} ± {std_v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jackknife evaluation for hierarchical ACL model")
    parser.add_argument(
        '--backbone', type=str, default='efficientnet',
        choices=['resnet', 'efficientnet', 'densenet', 'inception'],
        help='Backbone used during training',
    )
    args          = parser.parse_args()
    backbone_name = args.backbone
    print(f"Backbone: {backbone_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:   {device}")

    # ── Paths — must match model_with_mask_rad.py ─────────────────────────────
    root_dir       = "/home/yaxi/ACL_project/KneeMRI"
    csv_path       = os.path.join(root_dir, "metadata.csv")
    radiomics_path = "/home/yaxi/ACL_project/binary_classification/rediomics_results/radiomics_results_wide.csv"
    mask_path      = "/home/yaxi/ACL_project/binary_classification/predicted_masks_901"

    _script_dir     = os.path.dirname(os.path.abspath(__file__))
    results_dir     = os.path.join(_script_dir, 'results', f'results_{backbone_name}')
    checkpoints_dir = os.path.join(_script_dir, 'results', f'checkpoints_{backbone_name}')
    eval_dir        = os.path.join(results_dir, 'evaluation_metrics')
    os.makedirs(eval_dir, exist_ok=True)

    # ── Reproduce identical test split (random_state=42) ─────────────────────
    knee_df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(
        knee_df, test_size=0.5, random_state=42,
        stratify=knee_df["aclDiagnosis"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df["aclDiagnosis"],
    )
    print(f"\nTest set: {len(test_df)} samples")
    print("Class distribution in test set:")
    print(test_df['aclDiagnosis'].value_counts().sort_index())

    test_set = KneeMRI917Dataset(
        test_df,
        img_dir=root_dir,
        target_depth=64,
        cache_in_memory=True,
        use_global_normalization=True,
        radiomics_file=radiomics_path,
        mask_dir=mask_path,
        mask_mode="separate",
    )
    batch_size = 4

    # ── Build model (no DataParallel needed for evaluation) ───────────────────
    model = BinaryClassifier(backbone_name=backbone_name, radiomics_dim=107, fusion_dim=256)
    model = model.to(device)

    # ── Locate and load checkpoint ────────────────────────────────────────────
    candidate_paths = [
        os.path.join(checkpoints_dir, 'final_checkpoint.pt'),
        os.path.join(results_dir, 'best_model.pt'),
        os.path.join(results_dir,     'final_model.pt'),
    ]
    checkpoint_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found. Tried:\n" +
            "\n".join(f"  {p}" for p in candidate_paths) +
            "\nPlease train the model first."
        )
    print(f"\nLoading checkpoint: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)

    # Unwrap full-checkpoint dict if necessary
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
    else:
        state_dict = raw  # plain state dict (e.g., final_model.pt)

    # Strip 'module.' prefix saved by DataParallel
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # log_sigma_a / log_sigma_b are training-only loss-weight params;
    # they are absent in checkpoints saved before that feature was added.
    ignored = {'log_sigma_a', 'log_sigma_b'}
    real_missing = [k for k in missing if k not in ignored]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint: {unexpected}\n"
            f"Missing keys in checkpoint: {real_missing}"
        )
    model.eval()
    print("Checkpoint loaded successfully.")

    # ── Jackknife setup ───────────────────────────────────────────────────────
    test_labels_array = test_df['aclDiagnosis'].values          # shape (N,)
    test_indices      = np.arange(len(test_set))
    skf = StratifiedKFold(n_splits=5, shuffle=False)

    overall_runs, perclass_runs, head1_runs, head2_runs = [], [], [], []
    cms_3class, cms_h1, cms_h2 = [], [], []

    print("\n" + "="*60)
    print("JACKKNIFE EVALUATION  (5-fold drop-one-subset)")
    print("="*60)

    for fold_idx, (keep_idx, drop_idx) in enumerate(skf.split(test_indices, test_labels_array)):
        n_keep = len(keep_idx)
        n_drop = len(drop_idx)
        print(f"\n--- Run {fold_idx + 1}/5  "
              f"(evaluating on {n_keep} samples, dropping {n_drop}) ---")

        subset = Subset(test_set, keep_idx.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

        # ----- Inference -----
        labels, probs_h1, probs_h2 = run_inference(model, loader, device)

        # ----- Hierarchical predictions -----
        preds_3class, probs_3class, preds_h1, preds_h2 = \
            build_hierarchical_predictions(probs_h1, probs_h2)

        injured_mask = labels > 0

        # ----- Metrics -----
        overall  = compute_overall_metrics(labels, preds_3class, probs_3class)
        perclass = compute_per_class_metrics(labels, preds_3class)
        head1    = compute_head1_metrics(labels, preds_h1, probs_h1)
        head2    = compute_head2_metrics(
            labels[injured_mask],
            preds_h2[injured_mask],
            probs_h2[injured_mask],
        )

        overall_runs.append(overall)
        perclass_runs.append(perclass)
        head1_runs.append(head1)
        head2_runs.append(head2)

        cms_3class.append(overall['confusion_matrix'])
        cms_h1.append(head1['confusion_matrix'])
        cms_h2.append(head2['confusion_matrix'])   # may be None if no injured samples

        # Per-run status line
        h2_acc_str = f"{head2['accuracy']:.4f}" if not np.isnan(head2['accuracy']) else "N/A"
        h2_auc_str = f"{head2['auc']:.4f}"      if not np.isnan(head2['auc'])      else "N/A"
        print(f"  3-class Accuracy    = {overall['accuracy']:.4f}")
        print(f"  3-class Macro F1    = {overall['macro_f1']:.4f}")
        print(f"  Balanced Accuracy   = {overall['balanced_accuracy']:.4f}")
        print(f"  Head1  Acc={head1['accuracy']:.4f}  AUC={head1['auc']:.4f}")
        print(f"  Head2  Acc={h2_acc_str}  AUC={h2_auc_str}  "
              f"(n_injured={injured_mask.sum()})")

    # ── Aggregate across runs ─────────────────────────────────────────────────
    summary_overall = summarize_runs(overall_runs)
    summary_head1   = summarize_runs(head1_runs)
    summary_head2   = summarize_runs(head2_runs)

    # Per-class list metrics
    summary_perclass = {}
    for metric in ['precision_per_class', 'recall_per_class',
                   'f1_per_class', 'specificity_per_class']:
        summary_perclass[metric] = summarize_list_metric(perclass_runs, metric)

    # ── Print summaries ───────────────────────────────────────────────────────
    CLASS_NAMES = ['healthy', 'partial', 'complete']

    print_summary("SITUATION 1 — Overall 3-class  (mean ± std)", summary_overall)

    print(f"\n{'='*60}")
    print("SITUATION 2 — Per-class metrics  (mean ± std)")
    print('='*60)
    for metric_key in ['precision_per_class', 'recall_per_class',
                       'f1_per_class', 'specificity_per_class']:
        label = metric_key.replace('_per_class', '').capitalize()
        means = summary_perclass[metric_key].get('mean', [])
        stds  = summary_perclass[metric_key].get('std',  [])
        for i, cls in enumerate(CLASS_NAMES):
            if i < len(means):
                print(f"  {label:<14s} [{cls}]:  {means[i]:.4f} ± {stds[i]:.4f}")

    print_summary("SITUATION 3 — Head 1  (healthy vs injured)  (mean ± std)", summary_head1)
    print_summary("SITUATION 4 — Head 2  (partial vs complete) (mean ± std)", summary_head2)

    # ── Build full results dict ───────────────────────────────────────────────
    full_results = {
        'overall':   summary_overall,
        'per_class': summary_perclass,
        'head1':     summary_head1,
        'head2':     summary_head2,
    }

    # ── Save JSON + CSV ───────────────────────────────────────────────────────
    save_results_json(full_results, os.path.join(eval_dir, 'metrics.json'))
    save_results_csv(full_results,  os.path.join(eval_dir, 'metrics.csv'))

    # ── Save averaged confusion matrices ─────────────────────────────────────
    avg_cm_3class = np.mean(cms_3class, axis=0)
    avg_cm_h1     = np.mean(cms_h1,     axis=0)
    valid_cms_h2  = [c for c in cms_h2 if c is not None]
    avg_cm_h2     = np.mean(valid_cms_h2, axis=0) if valid_cms_h2 else np.zeros((2, 2))

    np.save(os.path.join(eval_dir, 'confusion_matrix_3class.npy'), avg_cm_3class)
    np.save(os.path.join(eval_dir, 'confusion_matrix_head1.npy'),  avg_cm_h1)
    np.save(os.path.join(eval_dir, 'confusion_matrix_head2.npy'),  avg_cm_h2)

    print(f"\nResults saved to: {eval_dir}/")
    print("  metrics.json")
    print("  metrics.csv")
    print("  confusion_matrix_3class.npy")
    print("  confusion_matrix_head1.npy")
    print("  confusion_matrix_head2.npy")
    print("\nEVALUATION COMPLETE")
