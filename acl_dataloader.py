import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from typing import Optional, Callable, Union
import warnings
import pickle

class KneeMRI917Dataset(Dataset):
    """
    Fully Optimized Dataset for 3D Knee MRI volumes.
    
    Key optimizations:
    1. File path caching during initialization (10-50x speedup)
    2. Optional in-memory caching for small datasets (100-1000x speedup)
    3. Global normalization for stable training (better for deep learning)
    4. Efficient tensor operations
    """
    
    def __init__(
        self,
        csv_path: Union[str, pd.DataFrame],
        img_dir: str,
        transform: Optional[Callable] = None,
        target_depth: int = 32,
        cache_in_memory: bool = False,
        use_global_normalization: bool = True,
        use_trilinear_interpolation: bool = True,
        radiomics_file: Optional[str] = None,
        radiomics_key: str = "volumeFilename",
        # If True, return (image, radiomics, label). If False, return (image, label, radiomics)
        # Keeping this switch makes it easier to stay compatible with older scripts.
        return_radiomics_first: bool = True,
        mask_dir: Optional[str] = None,
        mask_mode: str = "separate",  # "concat", "mul", "separate"
    ):
        """
        Args:
            csv_path: Path to metadata CSV file OR a pandas DataFrame.
            img_dir: Root directory containing volXX_reg folders.
            transform: Optional transform to apply to each image.
            target_depth: Target depth for interpolation (default: 64).
            cache_in_memory: If True, cache all volumes in RAM (only for small datasets).
            use_global_normalization: If True, use dataset-level statistics for normalization.
                                     If False, normalize each image independently.
            use_trilinear_interpolation: If True, interpolate depth. If False, use original depth.
        """
        # Handle both DataFrame and CSV file path
        if isinstance(csv_path, pd.DataFrame):
            self.data_frame = csv_path.reset_index(drop=True)
        elif isinstance(csv_path, str):
            self.data_frame = pd.read_csv(csv_path)
        else:
            raise TypeError("csv_path must be either a pandas DataFrame or a path to a CSV file")
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_depth = target_depth
        self.cache_in_memory = cache_in_memory
        self.use_global_normalization = use_global_normalization 
        self.use_trilinear_interpolation = use_trilinear_interpolation

        # Radiomics options
        self.radiomics_file = radiomics_file
        self.radiomics_key = radiomics_key
        self.return_radiomics_first = return_radiomics_first

        # Mask options
        self.mask_dir = mask_dir
        self.mask_mode = mask_mode

        # ========== OPTIMIZATION 1: Build file path cache during init ==========
        self.file_paths = self._build_file_path_cache()

        # ========== Build mask path cache if mask_dir provided ==========
        self.case_id_map = {}
        if self.mask_dir is not None:
            self.mask_paths = self._build_mask_path_cache()
        else:
            self.mask_paths = {}
        
        # ========== OPTIMIZATION 2: Compute global normalization stats ==========
        if self.use_global_normalization:
            self.global_mean, self.global_std = self._compute_global_stats()
        else:
            self.global_mean = None
            self.global_std = None
        
        # ========== OPTIMIZATION 3: Optional in-memory caching ==========
        self.memory_cache = {}
        if self.cache_in_memory:
            self._cache_all_volumes()
        

        if self.radiomics_file is not None:
            self._load_radiomics_features()


    def _load_radiomics_features(self):
        """Load pre-computed radiomics features from disk.

        Expected formats:
        - .csv: one row per sample, containing a lookup key (preferably volumeFilename)
                plus radiomics feature columns.
        - .pkl: a pickled dict produced by your preprocessing pipeline. This dataset supports
                either:
                  * a dict mapping keys -> feature tensors/arrays
                  * a dict with fields {"radiomics_features": ..., "feature_names": ...}

        The dataset will try to index radiomics by `self.radiomics_key` (default: volumeFilename).
        """
        
        print(f"\nLoading pre-computed radiomics from: {self.radiomics_file}")
        
        try:
            # Check file extension
            if self.radiomics_file.endswith('.pkl'):
                # Load PKL file
                with open(self.radiomics_file, 'rb') as f:
                    data = pickle.load(f)

                # Common formats:
                # 1) {"radiomics_features": <dict or list>, "feature_names": [...]}
                # 2) <dict> directly mapping key -> feature vector
                if isinstance(data, dict) and 'radiomics_features' in data:
                    self.radiomics_cache = data['radiomics_features']
                    self.radiomics_feature_names = data.get('feature_names', [])
                else:
                    self.radiomics_cache = data
                    self.radiomics_feature_names = []

                # If values are numpy arrays, convert lazily in __getitem__.
                # Try to infer feature dimension
                try:
                    first_val = next(iter(self.radiomics_cache.values())) if isinstance(self.radiomics_cache, dict) else self.radiomics_cache[0]
                    if torch.is_tensor(first_val):
                        self.radiomics_dim = int(first_val.numel())
                    else:
                        self.radiomics_dim = int(np.asarray(first_val).size)
                except Exception:
                    self.radiomics_dim = 0
                
                print(f"✓ Loaded {len(self.radiomics_cache)} radiomics feature vectors")
                print(f"✓ Feature dimension: {len(self.radiomics_feature_names)}")
                
            elif self.radiomics_file.endswith('.csv'):
                # Load CSV file
                df = pd.read_csv(self.radiomics_file)
                
                print(f"✓ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
                
                # Identify metadata columns (exclude from features)
                meta_cols = {
                    'patient_id', 'sample_idx', 'label', 'volumeFilename',
                    'aclDiagnosis', 'examId', 'seriesNo', 'roi_name', 'case_id',
                }

                # Pick key column
                if self.radiomics_key in df.columns:
                    key_col = self.radiomics_key
                elif 'volumeFilename' in df.columns:
                    key_col = 'volumeFilename'
                elif 'sample_idx' in df.columns:
                    key_col = 'sample_idx'
                else:
                    # Fall back to row index
                    key_col = None

                # Get all feature columns (everything except metadata + key)
                drop_cols = set(meta_cols)
                if key_col is not None:
                    drop_cols.add(key_col)
                feature_cols = [c for c in df.columns if c not in drop_cols]
                
                print(f"✓ Identified {len(feature_cols)} radiomics features")
                
                # Create cache dictionary indexed by chosen key
                self.radiomics_cache = {}

                for row_idx, row in df.iterrows():
                    if key_col is None:
                        key = int(row_idx)
                    else:
                        key = row[key_col]
                        # volumeFilename keys should match metadata.csv strings
                        if isinstance(key, str):
                            key = key.strip()
                        # numeric keys (sample_idx) should be int
                        elif pd.notna(key):
                            try:
                                key = int(key)
                            except Exception:
                                pass

                    feature_values = row[feature_cols].to_numpy(dtype=np.float32)
                    self.radiomics_cache[key] = torch.tensor(feature_values, dtype=torch.float32)
                
                # Store feature names for reference
                self.radiomics_feature_names = feature_cols
                self.radiomics_dim = len(feature_cols)
                
                print(f"✓ Loaded {len(self.radiomics_cache)} radiomics feature vectors")
                print(f"✓ Feature dimension: {len(self.radiomics_feature_names)}")
                # Print a tiny sanity check
                try:
                    k0 = next(iter(self.radiomics_cache.keys()))
                    print(f"✓ Radiomics key: {key_col if key_col is not None else 'row_index'} (example key: {k0})")
                except Exception:
                    pass
                
            else:
                raise ValueError(f"Unsupported file format: {self.radiomics_file}. Use .pkl or .csv")
            
        except Exception as e:
            warnings.warn(f"Failed to load radiomics file: {e}")
            import traceback
            traceback.print_exc()
            self.radiomics_cache = {}
            self.radiomics_feature_names = []
            self.radiomics_dim = 0


    def _build_file_path_cache(self):
        """
        Build a dictionary mapping indices to full file paths.
        This is done ONCE during initialization instead of on every data load.
        """
        file_paths = {}
        
        # Build a lookup dictionary for faster file searching
        file_lookup = {}
        for folder in os.listdir(self.img_dir):
            if folder.startswith("vol") and folder.endswith("_reg"):
                folder_path = os.path.join(self.img_dir, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.nii.gz') or file.endswith('.nii'):
                            file_lookup[file] = os.path.join(folder_path, file)
        
        # Map each index to its full path
        missing_files = []
        for idx in range(len(self.data_frame)):
            filename = str(self.data_frame.iloc[idx]["volumeFilename"]).strip()
            
            # Try direct path first
            full_path = os.path.join(self.img_dir, filename)
            if os.path.exists(full_path):
                file_paths[idx] = full_path
            # Try lookup dictionary
            elif filename in file_lookup:
                file_paths[idx] = file_lookup[filename]
            else:
                missing_files.append(filename)
                file_paths[idx] = None
        
        if missing_files:
            warnings.warn(f"Could not find {len(missing_files)} files. First few: {missing_files[:5]}")

        return file_paths

    def _build_mask_path_cache(self):
        """
        Scan mask_dir and map each dataset index to a mask file.
        Mask filenames follow pattern: *_{examId}_s{seriesNo}.nii.gz
        Also builds self.case_id_map: idx -> case_id string (for radiomics lookup).
        """
        import re

        mask_lookup = {}
        for fname in os.listdir(self.mask_dir):
            if not (fname.endswith('.nii.gz') or fname.endswith('.nii')):
                continue
            base = fname.replace('.nii.gz', '').replace('.nii', '')
            match = re.search(r'_(\d+)_s(\d+)$', base)
            if match:
                exam_id = int(match.group(1))
                series_no = int(match.group(2))
                mask_lookup[(exam_id, series_no)] = (
                    os.path.join(self.mask_dir, fname),
                    base,  # case_id e.g. vol01_reg_329637_s8
                )

        mask_paths = {}
        missing = 0
        for idx in range(len(self.data_frame)):
            row = self.data_frame.iloc[idx]
            exam_id = int(row['examId'])
            series_no = int(row['seriesNo'])
            key = (exam_id, series_no)
            if key in mask_lookup:
                mask_paths[idx] = mask_lookup[key][0]
                self.case_id_map[idx] = mask_lookup[key][1]
            else:
                mask_paths[idx] = None
                missing += 1

        if missing > 0:
            warnings.warn(f"Missing masks for {missing}/{len(self.data_frame)} samples")
        else:
            print(f"✓ Found masks for all {len(self.data_frame)} samples")

        return mask_paths

    def _load_and_process_mask(self, idx: int) -> torch.Tensor:
        """
        Load a binary mask file and resize to match image dimensions.
        Returns: [1, target_depth, 256, 256] tensor with values 0 or 1.
        """
        mask_path = self.mask_paths.get(idx)
        if mask_path is None:
            return torch.zeros(1, self.target_depth, 256, 256)

        mask = nib.load(mask_path).get_fdata()
        mask = (mask > 0).astype(np.float32)

        # Transpose from [H, W, D] to [D, H, W]
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(mask)

        # Resize to match image: [D, H, W] -> [1, 1, D, H, W] -> interpolate -> [1, D, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.target_depth, 256, 256),
            mode='nearest'
        )
        mask = mask.squeeze(0)  # [1, D, H, W]

        return mask

    def _compute_global_stats(self, sample_size: int = 100):
        """
        Compute dataset-level mean and standard deviation.
        
        This provides more stable normalization than per-image min-max scaling.
        Uses a sample of images to estimate statistics efficiently.
        
        Args:
            sample_size: Number of images to sample for statistics (default: 100)
        
        Returns:
            tuple: (mean, std) computed across sampled images
        """
        valid_indices = [idx for idx in range(len(self.data_frame)) if self.file_paths[idx] is not None]
        
        if len(valid_indices) == 0:
            warnings.warn("No valid files found for computing global stats. Using default values.")
            return 0.0, 1.0
        
        # Sample a subset of images
        sample_size = min(sample_size, len(valid_indices))
        sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
        
        all_values = []
        for idx in sample_indices:
            try:
                image = nib.load(self.file_paths[idx]).get_fdata()
                all_values.append(image.flatten())
            except Exception as e:
                warnings.warn(f"Error loading file at index {idx}: {e}")
                continue
        
        if len(all_values) == 0:
            warnings.warn("Could not load any images for stats. Using default values.")
            return 0.0, 1.0
        
        # Concatenate all values and compute statistics
        all_values = np.concatenate(all_values)
        mean = np.mean(all_values)
        std = np.std(all_values)
        
        return float(mean), float(std)
    
    def _cache_all_volumes(self):
        """Load all volumes into memory. Only use for small datasets."""
        for idx in range(len(self)):
            if self.file_paths[idx] is not None:
                try:
                    image = self._load_and_process_volume(idx)
                    self.memory_cache[idx] = image
                except Exception as e:
                    warnings.warn(f"Error caching volume at index {idx}: {e}")
    
    def _load_and_process_volume(self, idx: int) -> torch.Tensor:
        """
        Load and process a single volume.
        This is separated out for caching purposes.
        """
        full_path = self.file_paths[idx]
        
        if full_path is None:
            raise FileNotFoundError(f"File path not found for index {idx}")
        
        # Load MRI volume
        image = nib.load(full_path).get_fdata()
        image = image.astype(np.float32)
        
        # ========== NORMALIZATION ==========
        if self.use_global_normalization:
            # Use dataset-level statistics (more stable for deep learning)
            image = (image - self.global_mean) / (self.global_std + 1e-8)
            # Clip extreme values after standardization
            image = np.clip(image, -5, 5)
            # Scale to [0, 1]
            image = (image + 5) / 10
        else:
            # Per-image min-max normalization (original approach)
            image_min = image.min()
            image_max = image.max()
            image = (image - image_min) / (image_max - image_min + 1e-8)
        
        # Transpose from [H, W, D] to [D, H, W] and convert to tensor
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        
        # ========== INTERPOLATION AND CHANNEL REPLICATION ==========
        if self.use_trilinear_interpolation:
            # Add batch and channel dims: [1, 1, D, H, W]
            image = image.unsqueeze(0).unsqueeze(0)
            
            # Interpolate depth
            image = torch.nn.functional.interpolate(
                image, 
                size=(self.target_depth, 256, 256), 
                mode='trilinear', 
                align_corners=False
            )
            
            # Remove batch dim and replicate channels: [1, D, H, W] -> [3, D, H, W]
            image = image.squeeze(0).repeat(3, 1, 1, 1)
        else:
            # No interpolation - just replicate channels
            # [D, H, W] -> [3, D, H, W]
            image = image.unsqueeze(0).repeat(3, 1, 1, 1)
        
        return image
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # ========== Load from cache or disk ==========
        if self.cache_in_memory and idx in self.memory_cache:
            image = self.memory_cache[idx]
        else:
            image = self._load_and_process_volume(idx)
        
        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)

        # ========== Combine with mask if mask_dir provided ==========
        mask = None
        if self.mask_dir is not None:
            mask = self._load_and_process_mask(idx)  # [1, D, H, W]
            if self.mask_mode == "mul":
                # Multiply: broadcast mask across all 3 channels
                image = image * mask.expand_as(image)  # [3, D, H, W]
                mask = None  # mask already applied, don't return separately
            elif self.mask_mode == "concat":
                # Concat: append mask as extra channel
                image = torch.cat([image, mask], dim=0)  # [4, D, H, W]
                mask = None  # mask merged into image, don't return separately
            # else: mask_mode == "separate" → keep mask as separate tensor

        # Get label
        if "aclDiagnosis" in self.data_frame.columns:
            label = torch.tensor(self.data_frame.iloc[idx]["aclDiagnosis"], dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        # ========== Load radiomics features ==========
        radiomics_features = None
        if self.radiomics_file is not None and hasattr(self, "radiomics_cache") and len(self.radiomics_cache) > 0:
            # Try case_id from mask mapping first (e.g. vol01_reg_329637_s8)
            key = None
            if idx in self.case_id_map:
                key = self.case_id_map[idx]

            # Fall back to metadata columns
            if key is None or key not in self.radiomics_cache:
                if self.radiomics_key in self.data_frame.columns:
                    key = self.data_frame.iloc[idx][self.radiomics_key]
                elif "volumeFilename" in self.data_frame.columns:
                    key = self.data_frame.iloc[idx]["volumeFilename"]

            if isinstance(key, str):
                key = key.strip()
            elif key is not None and pd.notna(key):
                try:
                    key = int(key)
                except Exception:
                    pass

            # Fallback to idx if no key or missing in cache
            if key is not None and key in self.radiomics_cache:
                radiomics_features = self.radiomics_cache[key]
            elif idx in self.radiomics_cache:
                radiomics_features = self.radiomics_cache[idx]
            else:
                dim = getattr(self, "radiomics_dim", len(getattr(self, "radiomics_feature_names", [])))
                radiomics_features = torch.zeros(dim, dtype=torch.float32)

            # Ensure tensor
            if not torch.is_tensor(radiomics_features):
                radiomics_features = torch.tensor(np.asarray(radiomics_features, dtype=np.float32), dtype=torch.float32)

        # ========== Return ==========
        if mask is not None and radiomics_features is not None:
            return image, radiomics_features, mask, label
        elif mask is not None:
            return image, mask, label
        elif radiomics_features is not None:
            if self.return_radiomics_first:
                return image, radiomics_features, label
            return image, label, radiomics_features
        else:
            return image, label
    
    def get_class_distribution(self):
        """Helper method to get class distribution."""
        if "aclDiagnosis" in self.data_frame.columns:
            return self.data_frame["aclDiagnosis"].value_counts().sort_index()
        return None


# ========== ADDITIONAL OPTIMIZATION: DataLoader helper ==========
def get_optimized_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """
    Create an optimized DataLoader with best practices.
    
    Args:
        dataset: The dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes. If None, auto-determines based on caching:
            - 0 if dataset uses cache_in_memory (no disk I/O)
            - 4 otherwise (parallel disk loading)
    
    Returns:
        DataLoader instance
    """
    # Auto-determine num_workers if not specified
    if num_workers is None:
        num_workers = 0 if dataset.cache_in_memory else 4
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches
    )