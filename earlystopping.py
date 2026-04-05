# Add this import at the top
from sklearn.metrics import f1_score

# Early stopping class (unchanged - it's already flexible)
class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience, min_delta=0, mode='max', restore_best_weights=True , warmup_epochs = 20):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'max' for accuracy/F1 (higher is better), 'min' for loss (lower is better)
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.warmup_epochs = warmup_epochs
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None
        
    def __call__(self, score, model, epoch):
        """
        Args:
            score: Current metric value (F1 score, accuracy, or loss)
            model: The model being trained
            epoch: Current epoch number
        """
        # start recording only after number of warmup_epochs epochs
        if epoch < self.warmup_epochs:

            return
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
        else:
            if self._is_improvement(score):
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
                if self.restore_best_weights:
                    self.best_model_state = model.state_dict().copy()
                print(f"✓ Validation metric improved to {score:.4f}")
            else:
                self.counter += 1
                print(f"⚠ EarlyStopping counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f} at epoch {self.best_epoch + 1})")
                
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"\n{'='*60}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"{'='*60}")
                    print(f"No improvement for {self.patience} epochs")
                    print(f"Best metric: {self.best_score:.4f} at epoch {self.best_epoch + 1}")
                    
                    if self.restore_best_weights:
                        model.load_state_dict(self.best_model_state)
                        print(f"Restored model weights from epoch {self.best_epoch + 1}")
    
    def _is_improvement(self, score):
        """Check if current score is an improvement over best score."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:  # mode == 'min'
            return score < self.best_score - self.min_delta