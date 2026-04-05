import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
from datetime import datetime


class MetricsTracker:
    """
    A class to track and save training metrics for classification models.
    """
    
    def __init__(self, class_names=None, save_dir='results', config=None):
        """
        Initialize the MetricsTracker.
        
        Args:
            class_names (list): List of class names for reporting
            save_dir (str): Directory to save results
            config (dict): Training configuration dictionary
        """
        self.class_names = class_names or ['Class 0', 'Class 1', 'Class 2']
        self.save_dir = save_dir
        self.config = config or {}
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize history tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0
        self.training_start_time = None
        self.training_end_time = None
        self.total_epochs_trained = 0
        
    def update_history(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """
        Update training history after each epoch.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Average training loss
            train_acc (float): Training accuracy
            val_loss (float): Average validation loss
            val_acc (float): Validation accuracy
        """
        self.history['epoch'].append(epoch + 1)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.total_epochs_trained = epoch + 1
        
    def save_best_model(self, model, val_acc, filepath=None):
        """
        Save model if it achieves best validation accuracy.
        
        Args:
            model: PyTorch model
            val_acc (float): Current validation accuracy
            filepath (str): Optional custom filepath
            
        Returns:
            bool: True if model was saved, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, 'best_model.pt')
            
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(model.state_dict(), filepath)
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            return True
        return False
    
    def start_training(self):
        """Mark the start of training."""
        self.training_start_time = datetime.now()
        
    def end_training(self):
        """Mark the end of training."""
        self.training_end_time = datetime.now()
    
    def print_metrics(self, metrics):
        """
        Print all metrics in a formatted way.
        
        Args:
            metrics (dict): Dictionary containing all metrics
        """
        print(f"\n{'='*60}")
        print("BASIC METRICS")
        print(f"{'='*60}")
        print(f"Loss: {metrics['avg_test_loss']:.4f}")
        print(f"Accuracy: {metrics['test_acc']:.2f}%")
        
        print(f"\n{'='*60}")
        print("DETAILED METRICS")
        print(f"{'='*60}")
        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"F1 Score (Macro):    {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (Macro):   {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):      {metrics['recall_macro']:.4f}")
        
        if not np.isnan(metrics['auc_ovr']):
            print(f"AUC (OvR):           {metrics['auc_ovr']:.4f}")
            print(f"AUC (OvO):           {metrics['auc_ovo']:.4f}")
        
        print(f"\n{'='*60}")
        print("PER-CLASS METRICS")
        print(f"{'='*60}")
        print(metrics['classification_report'])
        
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX")
        print(f"{'='*60}")
        print(metrics['confusion_matrix'])
    
    def save_all_metrics(self, metrics):
        """
        Save all metrics to files.
        
        Args:
            metrics (dict): Dictionary containing all metrics
        """
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        
        # 1. Save training configuration
        self._save_config()
        
        # 2. Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        print(f"✓ Saved {self.save_dir}/training_history.csv")
        
        # 3. Save overall metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision (Macro)', 
                       'Recall (Macro)', 'AUC (OvR)', 'AUC (OvO)', 'Test Loss'],
            'Value': [metrics['accuracy'], metrics['f1_macro'], metrics['f1_weighted'], 
                      metrics['precision_macro'], metrics['recall_macro'], 
                      metrics['auc_ovr'], metrics['auc_ovo'], metrics['avg_test_loss']]
        })
        metrics_df.to_csv(os.path.join(self.save_dir, 'test_metrics.csv'), index=False)
        print(f"✓ Saved {self.save_dir}/test_metrics.csv")
        
        # 4. Save confusion matrix
        cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                            index=self.class_names, 
                            columns=self.class_names)
        cm_df.to_csv(os.path.join(self.save_dir, 'confusion_matrix.csv'))
        print(f"✓ Saved {self.save_dir}/confusion_matrix.csv")
        
        # 5. Save classification report
        with open(os.path.join(self.save_dir, 'classification_report.txt'), 'w') as f:
            f.write("Classification Report\n")
            f.write("="*60 + "\n\n")
            f.write(metrics['classification_report'])
            f.write("\n\nConfusion Matrix\n")
            f.write("="*60 + "\n")
            f.write(str(metrics['confusion_matrix']))
        print(f"✓ Saved {self.save_dir}/classification_report.txt")
        
        # 6. Save per-class metrics
        per_class_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1-Score': metrics['f1_per_class']
        })
        per_class_df.to_csv(os.path.join(self.save_dir, 'per_class_metrics.csv'), index=False)
        print(f"✓ Saved {self.save_dir}/per_class_metrics.csv")
        
        # 7. Save comprehensive summary
        self._save_summary(metrics)
        
        print(f"\n{'='*60}")
        print(f"ALL RESULTS SAVED TO '{self.save_dir}/' FOLDER")
        print(f"{'='*60}")
        print("\nSaved files:")
        print("  - training_config.txt")
        print("  - training_history.csv")
        print("  - test_metrics.csv")
        print("  - confusion_matrix.csv")
        print("  - classification_report.txt")
        print("  - per_class_metrics.csv")
        print("  - training_summary.txt")
        print("  - best_model.pt")
    
    def _save_config(self):
        """Save training configuration to file."""
        config_path = os.path.join(self.save_dir, 'training_config.txt')
        
        with open(config_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" " * 20 + "TRAINING CONFIGURATION\n")
            f.write("="*70 + "\n\n")
            
            # Timestamp
            if self.training_start_time:
                f.write(f"Training Start: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.training_end_time:
                f.write(f"Training End: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.training_start_time:
                    duration = self.training_end_time - self.training_start_time
                    f.write(f"Total Duration: {duration}\n")
            f.write("\n")
            
            # Model configuration
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Model: {self.config.get('model_name', 'N/A')}\n")
            f.write(f"  Number of Classes: {self.config.get('n_class', 'N/A')}\n")
            f.write(f"  Class Names: {self.config.get('class_names', 'N/A')}\n")
            if 'model_params' in self.config:
                f.write(f"  Total Parameters: {self.config['model_params']:,}\n")
            f.write("\n")
            
            # Training configuration
            f.write("TRAINING HYPERPARAMETERS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Batch Size: {self.config.get('batch_size', 'N/A')}\n")
            f.write(f"  Max Epochs: {self.config.get('n_epochs', 'N/A')}\n")
            f.write(f"  Epochs Trained: {self.total_epochs_trained}\n")
            f.write(f"  Learning Rate: {self.config.get('learning_rate', 'N/A')}\n")
            f.write(f"  Weight Decay: {self.config.get('weight_decay', 'N/A')}\n")
            f.write(f"  Optimizer: {self.config.get('optimizer', 'N/A')}\n")
            f.write(f"  Loss Function: {self.config.get('loss_function', 'N/A')}\n")
            f.write("\n")
            
            # Dataset information
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total Samples: {self.config.get('total_samples', 'N/A')}\n")
            f.write(f"  Training Samples: {self.config.get('train_samples', 'N/A')}\n")
            f.write(f"  Validation Samples: {self.config.get('val_samples', 'N/A')}\n")
            f.write(f"  Test Samples: {self.config.get('test_samples', 'N/A')}\n")
            f.write(f"  Training Batches: {self.config.get('train_batches', 'N/A')}\n")
            f.write(f"  Validation Batches: {self.config.get('val_batches', 'N/A')}\n")
            f.write(f"  Test Batches: {self.config.get('test_batches', 'N/A')}\n")
            f.write("\n")
            
            # Early stopping
            if 'early_stopping' in self.config:
                f.write("EARLY STOPPING CONFIGURATION:\n")
                f.write("-" * 70 + "\n")
                es = self.config['early_stopping']
                f.write(f"  Patience: {es.get('patience', 'N/A')} epochs\n")
                f.write(f"  Min Delta: {es.get('min_delta', 'N/A')}\n")
                f.write(f"  Mode: {es.get('mode', 'N/A')}\n")
                f.write(f"  Restore Best Weights: {es.get('restore_best_weights', 'N/A')}\n")
                f.write("\n")
            
            # Device info
            if 'device' in self.config:
                f.write("DEVICE INFORMATION:\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Device: {self.config['device']}\n")
                if 'gpu_name' in self.config:
                    f.write(f"  GPU: {self.config['gpu_name']}\n")
                if 'gpu_memory' in self.config:
                    f.write(f"  GPU Memory: {self.config['gpu_memory']:.2f} GB\n")
                f.write("\n")
            
            # Best validation accuracy
            f.write("TRAINING RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Best Validation Accuracy: {self.best_val_acc:.2f}%\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
        
        print(f"✓ Saved {self.save_dir}/training_config.txt")
    
    def _save_summary(self, metrics):
        """Save a comprehensive training summary."""
        summary_path = os.path.join(self.save_dir, 'training_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" " * 22 + "TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Training info
            f.write("TRAINING INFORMATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Model: {self.config.get('model_name', 'N/A')}\n")
            f.write(f"  Batch Size: {self.config.get('batch_size', 'N/A')}\n")
            f.write(f"  Epochs Trained: {self.total_epochs_trained} / {self.config.get('n_epochs', 'N/A')}\n")
            f.write(f"  Learning Rate: {self.config.get('learning_rate', 'N/A')}\n")
            f.write(f"  Best Val Accuracy: {self.best_val_acc:.2f}%\n")
            
            if self.training_start_time and self.training_end_time:
                duration = self.training_end_time - self.training_start_time
                f.write(f"  Training Duration: {duration}\n")
            f.write("\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total Samples: {self.config.get('total_samples', 'N/A')}\n")
            f.write(f"  Train/Val/Test: {self.config.get('train_samples', 'N/A')} / ")
            f.write(f"{self.config.get('val_samples', 'N/A')} / ")
            f.write(f"{self.config.get('test_samples', 'N/A')}\n\n")
            
            # Test results
            f.write("TEST SET RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Test Loss: {metrics['avg_test_loss']:.4f}\n")
            f.write(f"  Test Accuracy: {metrics['test_acc']:.2f}%\n")
            f.write(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"  Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall (Macro): {metrics['recall_macro']:.4f}\n")
            if not np.isnan(metrics['auc_ovr']):
                f.write(f"  AUC (OvR): {metrics['auc_ovr']:.4f}\n")
                f.write(f"  AUC (OvO): {metrics['auc_ovo']:.4f}\n")
            f.write("\n")
            
            # Per-class results
            f.write("PER-CLASS RESULTS:\n")
            f.write("-" * 70 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"    Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"    F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
            f.write("\n")
            
            # Confusion matrix
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 70 + "\n")
            cm = metrics['confusion_matrix']
            f.write("           Predicted\n")
            f.write("           " + "  ".join([f"{cn:>10}" for cn in self.class_names]) + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"Actual {class_name:>10}  ")
                f.write("  ".join([f"{cm[i][j]:>10}" for j in range(len(self.class_names))]))
                f.write("\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
        
        print(f"✓ Saved {self.save_dir}/training_summary.txt")