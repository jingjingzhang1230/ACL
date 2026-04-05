import os
import math
import argparse
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from efficientnet_pytorch_3d import EfficientNet3D
from monai.networks.nets import DenseNet169
from pytorch_i3d import InceptionI3d

from earlystopping import EarlyStopping
from acl_dataloader import KneeMRI917Dataset
from metrics_tracker import MetricsTracker

parser = argparse.ArgumentParser()

parser.add_argument(
    '--backbone', type=str, default='efficientnet',
    choices=['resnet', 'efficientnet', 'densenet', 'inception'],
    help='Image backbone'
)

parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

parser.add_argument('--lf-weight-A', type=float, default=0.5, help='Weight for sigma A')
parser.add_argument('--lf-weight-B', type=float, default=1.5, help='Weight for sigma B')

parser.add_argument(
    '--loss-weighting',
    type=str,
    default='uncertainty',
    choices=['uncertainty', 'equal'],
    help='How to combine head1 and head2 losses'
)

parser.add_argument(
    '--log-sigma-reg',
    type=float,
    default=0.2,
    help='Regularization coefficient for uncertainty weighting'
)

parser.add_argument(
    '--outdir',
    type=str,
    default=None,
    help='Optional output directory. If omitted, build automatically from parameter values.'
)

args = parser.parse_args()

backbone_name = args.backbone
learning_rate = args.lr
weight_decay = args.weight_decay
dropout = args.dropout
fusion_dim = 256
radiomics_encoded_dim = 128
mask_encoded_dim = 128
batch_size = args.batch_size
wA = args.lf_weight_A
wB = args.lf_weight_B
loss_weighting = args.loss_weighting # 'uncertainty' for Kendall & Gal (2018) adaptive weighting, 'equal' for simple sum
log_sigma_reg = args.log_sigma_reg

print(f"Backbone selected: {backbone_name}")
print(f"LR: {learning_rate}")
print(f"Weight decay: {weight_decay}")
print(f"Dropout: {dropout}")
print(f"Loss weighting: {loss_weighting}")


# config

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

n_class = 3

# Create model

# resnet
# model = r3d_18(weights=R3D_18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, n_class)

# efficienet
# model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': n_class}, in_channels=3)

# inception
# class I3DWrapper(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#         self.i3d = InceptionI3d(
#             num_classes=400,
#             spatial_squeeze=True,
#             final_endpoint='Logits',
#             in_channels=3,
#             dropout_keep_prob=0.5
#         )
#         self.i3d.replace_logits(num_classes)
        
#         # Add adaptive pooling to remove spatial dimensions
#         self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
#     def forward(self, x):
#         x = self.i3d(x)
        
#         # If output has spatial dimensions, pool them out
#         if len(x.shape) > 2:
#             # Current shape: [batch, classes, d, h, w]
#             x = self.pool(x)  # -> [batch, classes, 1, 1, 1]
#             x = x.view(x.size(0), -1)  # -> [batch, classes]
        
#         return x

# Use the wrapper
# model = I3DWrapper(num_classes=n_class)

# Create 3D DenseNet169
# model = DenseNet169(
#     spatial_dims=3,        # 3D (for volumetric data)
#     in_channels=3,         # RGB channels
#     out_channels=n_class   # 3 output classes
# )



class InceptionI3dWrapper(nn.Module):
    """Extracts a flat [B, 1024] feature vector from I3D (bypasses broken logits layer)."""
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
        x = self.i3d.Mixed_5b(x);  x = self.i3d.Mixed_5c(x)   # [B, 1024, t, h, w]
        x = self.pool(x)                                        # [B, 1024, 1, 1, 1]
        return x.view(x.size(0), -1)                            # [B, 1024]


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone_name='efficientnet',
        radiomics_dim=107,
        fusion_dim=256,
        radiomics_encoded_dim=128,
        mask_encoded_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        # ===== Image encoder: selectable backbone =====
        if backbone_name == 'resnet':
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            feature_dim = self.backbone.fc.in_features          # 512
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'efficientnet':
            try:
                self.backbone = EfficientNet3D.from_pretrained("efficientnet-b0", in_channels=3)
            except Exception as e:
                print(f"[WARN] Pretrained EfficientNet3D unavailable ({e}). Using random init.")
                self.backbone = EfficientNet3D.from_name("efficientnet-b0", in_channels=3)
            feature_dim = self.backbone._fc.in_features         # 1280
            self.backbone._fc = nn.Identity()

        elif backbone_name == 'densenet':
            feature_dim = 1024
            self.backbone = DenseNet169(
                spatial_dims=3,
                in_channels=3,
                out_channels=feature_dim,                       # backbone outputs feature directly
            )

        elif backbone_name == 'inception':
            self.backbone = InceptionI3dWrapper()
            feature_dim = 1024                                  # Mixed_5c output channels

        else:
            raise ValueError(f"Unknown backbone '{backbone_name}'. "
                             "Choose from: resnet, efficientnet, densenet, inception.")

        # ===== Radiomics encoder: MLP =====
        self.radiomics_encoder = nn.Sequential(
            nn.Linear(radiomics_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, radiomics_encoded_dim),
            nn.LayerNorm(radiomics_encoded_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ===== Mask encoder: lightweight 3D CNN =====
        # Input: [B, 1, D, 256, 256] → Output: [B, mask_encoded_dim]
        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),   # → [B, 16, D/2, 128, 128]
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # → [B, 32, D/4, 64, 64]
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # → [B, 64, D/8, 32, 32]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), # → [B, 128, D/16, 16, 16]
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),                                 # → [B, 128, 1, 1, 1]
            nn.Flatten(),                                            # → [B, 128]
        )
        self.mask_fc = nn.Sequential(
            nn.Linear(128, mask_encoded_dim),
            nn.LayerNorm(mask_encoded_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ===== Fusion: image + radiomics(128) + mask(128) → fusion_dim =====
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + radiomics_encoded_dim + mask_encoded_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # two separate heads for the two binary tasks
        self.head1 = nn.Linear(fusion_dim, 2)  # healthy vs injured
        self.head2 = nn.Linear(fusion_dim, 2)  # partial vs complete

        # Learnable log-uncertainty weights (Kendall & Gal, 2018)
        # init: wA=0.5 (head1 less certain), wB=1.5 (head2 more certain)
        self.log_sigma_a = nn.Parameter(torch.tensor([-math.log(wA)], dtype=torch.float32))
        self.log_sigma_b = nn.Parameter(torch.tensor([-math.log(wB)], dtype=torch.float32))
        # change sigma_a & b
        
    def forward(self, x, radiomics_feature, mask):
        image_feat = self.backbone(x)                                   # [B, feature_dim]
        rad_feat   = self.radiomics_encoder(radiomics_feature.float())  # [B, 128]
        mask_feat  = self.mask_fc(self.mask_encoder(mask))              # [B, 128]
        h = torch.cat([image_feat, rad_feat, mask_feat], dim=1)
        h = self.fusion(h)                                              # [B, fusion_dim]
        return self.head1(h), self.head2(h)                             # [B, 2], [B, 2]


def fuse_probs_to_three_classes(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Reconstruct 3-class probabilities from the two binary heads.

    head1: P(healthy=0, injured=1)
    head2: P(partial=0, complete=1)  — valid for injured samples only

    p(class=0) = P(healthy)
    p(class=1) = P(injured) * P(partial | injured)
    p(class=2) = P(injured) * P(complete | injured)
    """
    pa = torch.softmax(logits1, dim=1)          # [B, 2]
    pb = torch.softmax(logits2, dim=1)          # [B, 2]
    p_healthy  = pa[:, 0:1]
    p_injured  = pa[:, 1:2]
    p_partial  = pb[:, 0:1]
    p_complete = pb[:, 1:2]
    return torch.cat([p_healthy, p_injured * p_partial, p_injured * p_complete], dim=1)  # [B, 3]


# Create model  (via --backbone CLI argument)
model = BinaryClassifier(
    backbone_name=backbone_name,
    radiomics_dim=107,
    fusion_dim=fusion_dim,
    radiomics_encoded_dim=radiomics_encoded_dim,
    mask_encoded_dim=mask_encoded_dim,
    dropout=dropout,
)
model = model.to(device)
model_name = backbone_name
# use 2 gpu
model = nn.DataParallel(model)

class_names_h1 = ['Healthy', 'Injured']        # Head 1: 0=healthy, 1=injured
class_names_h2 = ['Partial', 'Complete']       # Head 2: 0=partial, 1=complete tear
class_names = class_names_h1                   # kept for tracker compatibility
batch_size = 4
n_epochs = 800
learning_rate = 3e-3
weight_decay = 1e-4
results_dir = os.path.join('results', f'results_{model_name}')
checkpoints_dir = os.path.join('results', f'checkpoints_{model_name}')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# Early stopping config
early_stop_patience = 150
early_stop_min_delta = 0.001

# Data split config
test_size = 0.3
random_state = 42

# Paths
# local 
# root_dir = "/Users/elioz/Documents/**/ACL_Project/KneeMRI"
# cloud server
root_dir = "/home/yaxi/ACL_project/KneeMRI"
csv_path = os.path.join(root_dir, "metadata.csv")
radiomics_pkl = "/home/yaxi/ACL_project/binary_classification/pyradiomics_all_features_central_box.csv"
mask_path = "/home/yaxi/ACL_project/binary_classification/predicted_masks_901"
radiomics_path = "/home/yaxi/ACL_project/binary_classification/rediomics_results/radiomics_results_wide.csv"

# Load and split data
knee_df = pd.read_csv(csv_path)


train_df, temp_df = train_test_split(knee_df, test_size=test_size, random_state=random_state, stratify=knee_df["aclDiagnosis"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["aclDiagnosis"])

# Create datasets
train_set = KneeMRI917Dataset(
    train_df,           
    img_dir = root_dir,           
    target_depth = 64,
    cache_in_memory = True,
    use_global_normalization = True,
    radiomics_file = radiomics_path,  # path to radiomics_results_wide.csv
    mask_dir = mask_path,             # path to predicted_masks_901/
    mask_mode="separate",          # returns image, radiomics, mask, exam_id, label
)
val_set = KneeMRI917Dataset(
    val_df,           
    img_dir = root_dir,           
    target_depth = 64,
    cache_in_memory = True,
    use_global_normalization = True,
    radiomics_file = radiomics_path,  # path to radiomics_results_wide.csv
    mask_dir = mask_path,             # path to predicted_masks_901/
    mask_mode="separate",          # returns image, radiomics, mask, exam_id, label
)
test_set = KneeMRI917Dataset(
    test_df,           
    img_dir = root_dir,           
    target_depth = 64,
    cache_in_memory = True,
    use_global_normalization = True,
    radiomics_file = radiomics_path,  # path to radiomics_results_wide.csv
    mask_dir = mask_path,             # path to predicted_masks_901/
    mask_mode="separate",          # returns image, radiomics, mask, exam_id, label
)

# Create dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Print dataset info
print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")
print(f"Test samples: {len(test_set)}")
print(f"\nClass distribution in training set:")
print(train_df['aclDiagnosis'].value_counts().sort_index())
print(f"\nClass distribution in validation set:")
print(val_df['aclDiagnosis'].value_counts().sort_index())
print(f"\nClass distribution in testing set:")
print(test_df['aclDiagnosis'].value_counts().sort_index())

# Check single samples
sample_data = train_set[0]
if len(sample_data) == 4:
    image, radiomics, mask, label = sample_data
    print(f"Single image shape: {image.shape}")
    print(f"Radiomics shape: {radiomics.shape}")
    print(f"Mask shape: {mask.shape}")
    print("Image/mask/radiomics features loaded successfully!")
else:
    print("WARNING: dataset output shape is unexpected!")

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {model_name}")
print(f"Total parameters: {total_params:,}")

# Class weights for Head 1: healthy (0) vs injured (1 or 2)
n_healthy  = int((train_df["aclDiagnosis"] == 0).sum())
n_injured  = int((train_df["aclDiagnosis"] >  0).sum())
print(f"\nHead 1 counts — Healthy: {n_healthy}, Injured: {n_injured}")
w_h1 = torch.tensor(
    [len(train_df) / (2 * n_healthy + 1e-8),
     len(train_df) / (2 * n_injured  + 1e-8)],
    dtype=torch.float, device=device)
loss_fn_h1 = nn.CrossEntropyLoss(weight=w_h1)

# Class weights for Head 2: partial (1) vs complete (2) — on injured samples only
n_partial  = int((train_df["aclDiagnosis"] == 1).sum())
n_complete = int((train_df["aclDiagnosis"] == 2).sum())
n_injured_total = n_partial + n_complete
print(f"Head 2 counts — Partial: {n_partial}, Complete: {n_complete}")
w_h2 = torch.tensor(
    [n_injured_total / (2 * n_partial  + 1e-8),
     n_injured_total / (2 * n_complete + 1e-8)],
    dtype=torch.float, device=device)
loss_fn_h2 = nn.CrossEntropyLoss(weight=w_h2)
print(f"Head 1 weights (Healthy/Injured):   {w_h1.cpu().numpy()}")
print(f"Head 2 weights (Partial/Complete):  {w_h2.cpu().numpy()}")

learning_rate = args.lr
weight_decay = args.weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=10,
    verbose=True,
    min_lr=1e-7
)

# config dictionary
config = {
    'model_name': model_name,
    'n_class': n_class,
    'class_names': class_names,
    'model_params': total_params,
    'batch_size': batch_size,
    'n_epochs': n_epochs,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss (weighted inverse-frequency)',
    'total_samples': len(knee_df),
    'train_samples': len(train_set),
    'val_samples': len(val_set),
    'test_samples': len(test_set),
    'train_batches': len(train_loader),
    'val_batches': len(val_loader),
    'test_batches': len(test_loader),
    'device': str(device),
    'early_stopping': {
        'patience': early_stop_patience,
        'min_delta': early_stop_min_delta,
        'mode': 'max',
        'restore_best_weights': True
    }
}

# Add GPU info if available
if torch.cuda.is_available():
    config['gpu_name'] = torch.cuda.get_device_name(0)
    config['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9

# Initialize MetricsTracker with config
tracker = MetricsTracker(class_names=class_names, save_dir=results_dir, config=config)

# Initialize early stopping
early_stopping = EarlyStopping(
    patience=early_stop_patience,
    min_delta=early_stop_min_delta,
    mode='max',
    restore_best_weights=True
)

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Model: {model_name}")
print(f"Batch size: {batch_size}")
print(f"Max epochs: {n_epochs}")
print(f"Learning rate: {learning_rate}")
print("="*60)


def save_training_log_excel(log_rows: list[dict], save_dir: str, filename: str = 'training_log.xlsx') -> None:
    """Save per-epoch training log to an Excel file inside save_dir.

    Args:
        log_rows: List of dicts, one per epoch, with keys:
                  epoch, train_loss, train_acc_h1, train_acc_h2,
                  val_loss, val_acc_h1, val_acc_h2, val_acc_3class,
                  weight_h1, weight_h2
        save_dir: Directory where the Excel file will be written (created if absent).
        filename: Output filename (default: training_log.xlsx).
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(log_rows, columns=[
        'epoch', 'train_loss', 'train_acc_h1', 'train_acc_h2',
        'val_loss', 'val_acc_h1', 'val_acc_h2', 'val_acc_3class',
        'weight_h1', 'weight_h2',
    ])
    out_path = os.path.join(save_dir, filename)
    df.to_excel(out_path, index=False)
    print(f"Training log saved → {out_path}")


# Mark training start
tracker.start_training()

# Training
best_val_acc_h1 = 0.0
training_log: list[dict] = []

for epoch in range(n_epochs):
    # ============================================================
    # TRAINING
    # ============================================================
    model.train()
    epoch_train_loss = 0.0
    train_correct_h1 = 0
    train_total_h1 = 0
    train_correct_h2 = 0
    train_total_h2 = 0

    for batch, (X, r, m, y) in enumerate(tqdm(train_loader, desc="Training")):
        X, r, m, y = X.to(device), r.to(device), m.to(device), y.to(device)

        # Binary labels for each head
        y_h1 = (y > 0).long()                    # 0=healthy, 1=injured
        injured_mask = y > 0
        y_h2 = (y[injured_mask] - 1).long()      # 0=partial, 1=complete

        # Forward pass
        logits1, logits2 = model(X, r, m)        # [B, 2], [B, 2]

        # Losses — uncertainty-weighted hierarchical
        loss1 = loss_fn_h1(logits1, y_h1)
        loss2 = loss_fn_h2(logits2[injured_mask], y_h2) if injured_mask.any() else logits1.new_tensor(0.0)
        if loss_weighting == 'uncertainty':
            inv_var_a = torch.exp(-model.module.log_sigma_a)
            inv_var_b = torch.exp(-model.module.log_sigma_b)
            total_loss = (
                inv_var_a * loss1
                + inv_var_b * loss2
                + log_sigma_reg * (model.module.log_sigma_a + model.module.log_sigma_b)
            )
        elif loss_weighting == 'equal':
            total_loss = loss1 + loss2
        else:
            raise ValueError(f"Unknown loss weighting mode: {loss_weighting}")

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Keep log-sigma in a numerically safe range (weight ∈ [0.2, 3.0])
        with torch.no_grad():
            model.module.log_sigma_a.clamp_(-math.log(3.0), -math.log(0.2))
            model.module.log_sigma_b.clamp_(-math.log(3.0), -math.log(0.2))

        # Calculate accuracy
        with torch.no_grad():
            preds_h1 = logits1.argmax(dim=1)
            train_correct_h1 += (preds_h1 == y_h1).sum().item()
            train_total_h1 += y.size(0)

            if injured_mask.any():
                preds_h2 = logits2[injured_mask].argmax(dim=1)
                train_correct_h2 += (preds_h2 == y_h2).sum().item()
                train_total_h2 += injured_mask.sum().item()

        epoch_train_loss += total_loss.item()

    train_acc_h1 = 100.0 * train_correct_h1 / train_total_h1
    train_acc_h2 = 100.0 * train_correct_h2 / (train_total_h2 + 1e-8)
    avg_train_loss = epoch_train_loss / len(train_loader)

    # ============================================================
    # VALIDATION
    # ============================================================
    model.eval()
    epoch_val_loss = 0.0
    val_correct_h1 = 0
    val_total_h1 = 0
    val_correct_h2 = 0
    val_total_h2 = 0
    val_probs3_list: list[torch.Tensor] = []   # collect fused 3-class probs
    val_labels_list: list[torch.Tensor] = []   # collect ground-truth labels

    with torch.no_grad():
        for batch, (X_val, r_val, m_val, y_val) in enumerate(tqdm(val_loader, desc="Validation")):
            X_val, r_val, m_val, y_val = X_val.to(device), r_val.to(device), m_val.to(device), y_val.to(device)

            y_h1_val = (y_val > 0).long()
            injured_mask_val = y_val > 0
            y_h2_val = (y_val[injured_mask_val] - 1).long()

            logits1_val, logits2_val = model(X_val, r_val, m_val)

            loss1_val = loss_fn_h1(logits1_val, y_h1_val)
            loss2_val = loss_fn_h2(logits2_val[injured_mask_val], y_h2_val) if injured_mask_val.any() else torch.tensor(0.0, device=device)
            epoch_val_loss += (loss1_val + loss2_val).item()

            # Fuse heads → 3-class probs for unified accuracy
            prob3_val = fuse_probs_to_three_classes(logits1_val, logits2_val)
            val_probs3_list.append(prob3_val.cpu())
            val_labels_list.append(y_val.cpu())

            preds_h1_val = logits1_val.argmax(dim=1)
            val_correct_h1 += (preds_h1_val == y_h1_val).sum().item()
            val_total_h1 += y_val.size(0)

            if injured_mask_val.any():
                preds_h2_val = logits2_val[injured_mask_val].argmax(dim=1)
                val_correct_h2 += (preds_h2_val == y_h2_val).sum().item()
                val_total_h2 += injured_mask_val.sum().item()

    val_acc_h1 = 100.0 * val_correct_h1 / val_total_h1
    val_acc_h2 = 100.0 * val_correct_h2 / (val_total_h2 + 1e-8)
    avg_val_loss = epoch_val_loss / len(val_loader)

    # 3-class accuracy from fused head probabilities
    all_probs3  = torch.cat(val_probs3_list, dim=0)   # [N, 3]
    all_labels  = torch.cat(val_labels_list, dim=0)   # [N]
    val_acc_3cl = 100.0 * (all_probs3.argmax(dim=1) == all_labels).float().mean().item()

    scheduler.step(val_acc_h1)  # use head-1 accuracy as primary metric

    # Save best model
    if val_acc_h1 > best_val_acc_h1:
        best_val_acc_h1 = val_acc_h1
        torch.save(model.module.state_dict(), 'best_fusion_model.pt')
        print(f"  → New best model saved! Val Acc H1: {val_acc_h1:.2f}%")

    # Update metrics tracker (use H1 acc as the primary scalar)
    train_acc = train_acc_h1
    val_acc = val_acc_h1
    tracker.update_history(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc)
    tracker.save_best_model(model, val_acc_h1)

    # ========== EPOCH SUMMARY ==========
    print(f"\n{'─'*60}")
    print(f"EPOCH {epoch + 1} SUMMARY, {model_name}")
    print(f"{'─'*60}")
    print(f"  Train Loss: {avg_train_loss:.4f} | H1 Acc: {train_acc_h1:.2f}% | H2 Acc: {train_acc_h2:.2f}%")
    print(f"  Val   Loss: {avg_val_loss:.4f} | H1 Acc: {val_acc_h1:.2f}% | H2 Acc: {val_acc_h2:.2f}% | 3-class Acc: {val_acc_3cl:.2f}%")
    wA = float(torch.exp(-model.module.log_sigma_a).detach().cpu())
    wB = float(torch.exp(-model.module.log_sigma_b).detach().cpu())
    print(f"  Task weights → wA(H1)={wA:.3f}  wB(H2)={wB:.3f}")

    training_log.append({
        'epoch': epoch + 1,
        'train_loss': round(avg_train_loss, 6),
        'train_acc_h1': round(train_acc_h1, 4),
        'train_acc_h2': round(train_acc_h2, 4),
        'val_loss': round(avg_val_loss, 6),
        'val_acc_h1': round(val_acc_h1, 4),
        'val_acc_h2': round(val_acc_h2, 4),
        'val_acc_3class': round(val_acc_3cl, 4),
        'weight_h1': round(wA, 6),
        'weight_h2': round(wB, 6),
    })

    # early stop
    if epoch > 100:
        early_stopping(val_acc_h1, model, epoch)

    if early_stopping.early_stop:
        print(f" Early stopping triggered after {epoch + 1} epochs")
        print(f" Best validation H1 accuracy: {early_stopping.best_score:.2f}%")
        break  # Exit the training loop

# Mark training end
tracker.end_training()

# Save per-epoch training log to Excel
save_training_log_excel(training_log, save_dir=results_dir)

# ========== SAVE FINAL CHECKPOINT ==========
torch.save({
    'epoch': tracker.total_epochs_trained,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, os.path.join(checkpoints_dir, 'final_checkpoint.pt'))

torch.save(model.module.state_dict(), os.path.join(results_dir, 'final_model.pt'))
print(f"\nSaved {results_dir}/final_model.pt")
print(f"Saved {checkpoints_dir}/final_checkpoint.pt")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print(f"Model: {model_name}")
print("="*60)
print(f"Total epochs trained: {tracker.total_epochs_trained}/{n_epochs}")
print(f"Best validation accuracy: {tracker.best_val_acc:.2f}%")
print("="*60)
print("Launching evaluate.py for jackknife evaluation...")

import subprocess, sys
result = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), 'evaluate_hier.py'),
     '--backbone', backbone_name],
    check=False,
)
if result.returncode != 0:
    print(f"[WARN] evaluate_hier.py exited with code {result.returncode}.")
