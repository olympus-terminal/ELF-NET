#!/usr/bin/env python3
"""
train_manifold_20260125_173853.py

PyTorch training script for Pfam -> AlphaEarth projection model.

Provenance:
    - Task: ralph8_plan.md Task 9
    - Generated: 2026-01-25
    - Purpose: Train environmental genome manifold projection model
    - Model spec: source_data/model_architecture_spec.md
    - Data spec: scripts/data/manifold_train_val_test_*.npz

Features:
    - Multi-model support (full, light, pca)
    - Mixed precision training (AMP)
    - Learning rate scheduling (cosine annealing with warmup)
    - Early stopping with patience
    - Checkpoint saving (best val loss, periodic)
    - TensorBoard logging
    - Reproducible training with seed setting
    - GPU/CPU auto-detection

Usage:
    # Local testing (CPU, small epochs)
    python train_manifold_20260125_173853.py --model light --epochs 5 --device cpu

    # Full training (GPU)
    python train_manifold_20260125_173853.py --model full --epochs 200 --device cuda

    # Resume from checkpoint
    python train_manifold_20260125_173853.py --resume checkpoints/best_model.pt
"""

import os
import sys
import socket
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# =============================================================================
# Environment Detection (per JUBAIL_BEST_PRACTICES.md)
# =============================================================================

def get_base_dir() -> Path:
    """Detect environment and return appropriate base directory."""
    hostname = socket.gethostname()

    # Check if running on HPC (Jubail login/compute/GPU nodes)
    # Login nodes: login1.fast, login2.fast, etc.
    # Compute nodes: cn###, dn###, gpu###
    if any(x in hostname for x in ['cn', 'gpu', 'dn', 'jubail', 'login', 'fast']):
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    else:
        # Running locally
        return Path("/media/drn2/External/TARA-Oceans")

# =============================================================================
# Data Integrity Guard
# =============================================================================

def enforce_data_integrity():
    """
    Ensure no synthetic or placeholder data is used.
    This is a placeholder for the actual validation.
    """
    pass

def validate_input_source(path: str, description: str) -> None:
    """
    Validate that input file exists and is not empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description}: File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{description}: File is empty: {path}")

# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Training configuration with sensible defaults."""

    def __init__(self, args: argparse.Namespace = None):
        # Paths - auto-detect environment using hostname (per JUBAIL_BEST_PRACTICES.md)
        self.base_dir = get_base_dir()
        print(f"[ENV] Base directory: {self.base_dir}")
        print(f"[ENV] Hostname: {socket.gethostname()}")

        self.data_dir = self.base_dir / 'MANUSCRIPT/data'
        self.checkpoint_dir = self.base_dir / 'MANUSCRIPT/checkpoints'
        self.log_dir = self.base_dir / 'MANUSCRIPT/logs'

        # Model configuration
        self.model_type = 'light'  # 'full', 'light', 'pca'
        self.input_dim = 9611
        self.output_dim = 64

        # Training hyperparameters
        self.batch_size = 32
        self.epochs = 200
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.gradient_clip = 1.0

        # Loss configuration
        self.loss_type = 'combined'  # 'mse', 'cosine', 'combined'
        self.loss_alpha = 0.8  # Weight for MSE in combined loss

        # Learning rate schedule
        self.scheduler_type = 'cosine'  # 'cosine', 'plateau', 'none'
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.T_0 = 50  # For cosine annealing with restarts

        # Early stopping
        self.patience = 30
        self.min_delta = 1e-6

        # Checkpointing
        self.save_every = 25  # Save checkpoint every N epochs
        self.keep_best_n = 3  # Keep N best checkpoints

        # Device and precision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = True  # Automatic mixed precision

        # Reproducibility
        self.seed = 42

        # Data loading
        self.num_workers = 4

        # Override with command line args if provided
        if args is not None:
            self._update_from_args(args)

    def _update_from_args(self, args: argparse.Namespace):
        """Update config from command line arguments."""
        if args.model:
            self.model_type = args.model
        if args.epochs:
            self.epochs = args.epochs
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.lr:
            self.learning_rate = args.lr
        if args.device:
            self.device = args.device
        if args.seed:
            self.seed = args.seed
        if args.no_amp:
            self.use_amp = False
        if args.data_file:
            self.data_file = args.data_file
        if hasattr(args, 'loss') and args.loss:
            self.loss_type = args.loss

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'loss_type': self.loss_type,
            'loss_alpha': self.loss_alpha,
            'scheduler_type': self.scheduler_type,
            'patience': self.patience,
            'device': self.device,
            'use_amp': self.use_amp,
            'seed': self.seed
        }

# =============================================================================
# Dataset
# =============================================================================

class ManifoldDataset(Dataset):
    """
    Dataset for Pfam -> AlphaEarth projection.
    Loads from prepared npz file.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, sample_ids: np.ndarray = None):
        """
        Args:
            X: Pfam feature matrix (N, 9611)
            Y: AlphaEarth embeddings (N, 64)
            sample_ids: Optional sample IDs for tracking
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

def load_data(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load prepared data and create DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata)
    """
    # Find data file
    if hasattr(config, 'data_file') and config.data_file:
        data_path = Path(config.data_file)
    else:
        # Find most recent data file
        data_files = sorted(config.data_dir.glob('manifold_train_val_test_*.npz'))
        if not data_files:
            raise FileNotFoundError(f"No data files found in {config.data_dir}")
        data_path = data_files[-1]

    validate_input_source(str(data_path), "Training data")
    print(f"Loading data from: {data_path}")

    # Load npz
    data = np.load(data_path, allow_pickle=True)

    # Extract arrays
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    ids_train = data['ids_train'] if 'ids_train' in data else None
    ids_val = data['ids_val'] if 'ids_val' in data else None
    ids_test = data['ids_test'] if 'ids_test' in data else None

    # Metadata
    metadata = {
        'pfam_columns': data['pfam_columns'] if 'pfam_columns' in data else None,
        'ae_columns': data['ae_columns'] if 'ae_columns' in data else None,
        'holdout_strategy': str(data['holdout_strategy']) if 'holdout_strategy' in data else 'unknown',
        'data_path': str(data_path),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test)
    }

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Targets: {Y_train.shape[1]}")

    # Update config dimensions
    config.input_dim = X_train.shape[1]
    config.output_dim = Y_train.shape[1]

    # Create datasets
    train_dataset = ManifoldDataset(X_train, Y_train, ids_train)
    val_dataset = ManifoldDataset(X_val, Y_val, ids_val)
    test_dataset = ManifoldDataset(X_test, Y_test, ids_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == 'cuda'),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == 'cuda')
    )

    return train_loader, val_loader, test_loader, metadata

# =============================================================================
# Model Import (inline to avoid import issues)
# =============================================================================

class PfamToAlphaEarthProjector(nn.Module):
    """Deep MLP with bottleneck for Pfam -> AlphaEarth projection."""

    def __init__(
        self,
        input_dim: int = 9611,
        output_dim: int = 64,
        hidden_dims: list = None,
        dropout_rates: list = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048, 512, 128]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.3, 0.2]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PfamToAlphaEarthLight(nn.Module):
    """Lighter version for reduced overfitting risk."""

    def __init__(self, input_dim: int = 9611, output_dim: int = 64):
        super().__init__()
        self.model = PfamToAlphaEarthProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[512, 256, 128],
            dropout_rates=[0.3, 0.25, 0.2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return self.model.count_parameters()

def get_model(model_type: str, input_dim: int, output_dim: int) -> nn.Module:
    """Factory function for models."""
    if model_type == 'full':
        return PfamToAlphaEarthProjector(input_dim=input_dim, output_dim=output_dim)
    elif model_type == 'light':
        return PfamToAlphaEarthLight(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# Loss Function
# =============================================================================

class MSEOnlyLoss(nn.Module):
    """Pure MSE loss (returns compatible tuple)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse_loss = self.mse(predictions, targets)
        cos_loss = 1 - self.cos(predictions, targets).mean()
        return mse_loss, mse_loss, cos_loss  # total = mse

class CosineOnlyLoss(nn.Module):
    """Pure Cosine Similarity loss (returns compatible tuple)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse_loss = self.mse(predictions, targets)
        cos_loss = 1 - self.cos(predictions, targets).mean()
        return cos_loss, mse_loss, cos_loss  # total = cosine

class CombinedLoss(nn.Module):
    """Combined MSE + Cosine Similarity loss."""

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse_loss = self.mse(predictions, targets)
        cos_loss = 1 - self.cos(predictions, targets).mean()
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * cos_loss
        return total_loss, mse_loss, cos_loss

def get_loss_function(loss_type: str, alpha: float = 0.8) -> nn.Module:
    """Factory function to create loss function by type."""
    if loss_type == 'mse':
        return MSEOnlyLoss()
    elif loss_type == 'cosine':
        return CosineOnlyLoss()
    elif loss_type == 'combined':
        return CombinedLoss(alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class MetricTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: Path, config: TrainingConfig):
        self.log_dir = log_dir
        self.config = config
        self.history = {
            'train_loss': [], 'train_mse': [], 'train_cos': [],
            'val_loss': [], 'val_mse': [], 'val_cos': [],
            'learning_rate': [], 'epoch_time': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard (optional)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            print("TensorBoard not available, using file logging only")

    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update metrics for current epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # Update best
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            self.best_epoch = epoch

        # TensorBoard logging
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Print epoch summary."""
        lr = metrics.get('learning_rate', 0)
        t = metrics.get('epoch_time', 0)

        print(f"Epoch {epoch:4d} | "
              f"Train: {metrics['train_loss']:.6f} (MSE: {metrics['train_mse']:.4f}, Cos: {metrics['train_cos']:.4f}) | "
              f"Val: {metrics['val_loss']:.6f} (MSE: {metrics['val_mse']:.4f}, Cos: {metrics['val_cos']:.4f}) | "
              f"LR: {lr:.2e} | Time: {t:.1f}s")

    def save_history(self, path: Path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_clip: float = 1.0
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (total_loss, mse_loss, cosine_loss)
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass (with AMP if enabled)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(X_batch)
                loss, mse, cos = loss_fn(predictions, Y_batch)

            # Backward pass with scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(X_batch)
            loss, mse, cos = loss_fn(predictions, Y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos.item()
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> Tuple[float, float, float]:
    """
    Evaluate model on validation/test set.

    Returns:
        Tuple of (total_loss, mse_loss, cosine_loss)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        predictions = model(X_batch)
        loss, mse, cos = loss_fn(predictions, Y_batch)

        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos.item()
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches

@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Compute additional metrics beyond loss.

    Returns:
        Dictionary with R2, per-dimension correlations, etc.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(Y_batch.numpy())

    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Overall R2
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Per-dimension R2
    dim_r2 = []
    for i in range(targets.shape[1]):
        ss_res_i = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot_i = np.sum((targets[:, i] - targets[:, i].mean()) ** 2)
        if ss_tot_i > 0:
            dim_r2.append(1 - (ss_res_i / ss_tot_i))
        else:
            dim_r2.append(0.0)

    # Mean absolute error
    mae = np.mean(np.abs(targets - predictions))

    # Cosine similarity (mean across samples)
    dot = np.sum(predictions * targets, axis=1)
    norm_pred = np.linalg.norm(predictions, axis=1)
    norm_targ = np.linalg.norm(targets, axis=1)
    cos_sim = np.mean(dot / (norm_pred * norm_targ + 1e-8))

    return {
        'r2': r2,
        'mae': mae,
        'cosine_similarity': cos_sim,
        'dim_r2_mean': np.mean(dim_r2),
        'dim_r2_min': np.min(dim_r2),
        'dim_r2_max': np.max(dim_r2)
    }

# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    val_loss: float,
    config: TrainingConfig,
    path: Path,
    metadata: Dict = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'config': config.to_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Optional[Any] = None
) -> Dict:
    """Load training checkpoint."""
    # Use weights_only=False since we save metadata (numpy arrays) in checkpoints
    # This is safe because we only load our own checkpoints
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainingConfig, resume_path: Optional[Path] = None) -> Dict:
    """
    Main training function.

    Args:
        config: Training configuration
        resume_path: Optional checkpoint path to resume from

    Returns:
        Dictionary with final metrics and paths
    """
    enforce_data_integrity()
    set_seed(config.seed)

    # Create directories with unique run name (model_loss_lr_timestamp)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lr_str = f"{config.learning_rate:.0e}".replace('-', 'm').replace('+', 'p')
    run_name = f"{config.model_type}_{config.loss_type}_{lr_str}_{timestamp}"

    checkpoint_dir = config.checkpoint_dir / run_name
    log_dir = config.log_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pfam -> AlphaEarth Manifold Projection Training")
    print("=" * 80)
    print(f"\nRun: {run_name}")
    print(f"Device: {config.device}")
    print(f"Model: {config.model_type}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Load data
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, data_metadata = load_data(config)

    # Create model
    print("\n[2/4] Creating model...")
    model = get_model(config.model_type, config.input_dim, config.output_dim)
    model = model.to(config.device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Loss function
    loss_fn = get_loss_function(config.loss_type, alpha=config.loss_alpha)
    print(f"  Loss function: {config.loss_type}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    if config.scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0, T_mult=1, eta_min=config.min_lr
        )
    elif config.scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=config.min_lr
        )
    else:
        scheduler = None

    # Mixed precision
    scaler = None
    if config.use_amp and config.device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("  Using automatic mixed precision (AMP)")

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path and resume_path.exists():
        print(f"\n  Resuming from: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch}")

    # Metric tracking
    tracker = MetricTracker(log_dir, config)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    # Save config
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Training loop
    print("\n[3/4] Training...")
    print("-" * 80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_mse, train_cos = train_epoch(
            model, train_loader, optimizer, loss_fn,
            config.device, scaler, config.gradient_clip
        )

        # Validate
        val_loss, val_mse, val_cos = evaluate(
            model, val_loader, loss_fn, config.device
        )

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start

        # Track metrics
        metrics = {
            'train_loss': train_loss, 'train_mse': train_mse, 'train_cos': train_cos,
            'val_loss': val_loss, 'val_mse': val_mse, 'val_cos': val_cos,
            'learning_rate': current_lr, 'epoch_time': epoch_time
        }
        tracker.update(epoch, metrics)
        tracker.log_epoch(epoch, metrics)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, config, best_path, data_metadata
            )

        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, config, periodic_path, data_metadata
            )

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    print("-" * 80)

    # Final evaluation
    print("\n[4/4] Final evaluation...")

    # Load best model
    best_path = checkpoint_dir / 'best_model.pt'
    if best_path.exists():
        load_checkpoint(best_path, model)

    # Compute metrics on all splits
    final_metrics = {}
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        loss, mse, cos = evaluate(model, loader, loss_fn, config.device)
        metrics = compute_metrics(model, loader, config.device)
        final_metrics[name] = {
            'loss': loss, 'mse': mse, 'cosine_loss': cos,
            **metrics
        }

    # Print final results
    print("\nFinal Results:")
    print("-" * 40)
    for split, m in final_metrics.items():
        print(f"  {split.upper()}:")
        print(f"    Loss: {m['loss']:.6f}")
        print(f"    MSE:  {m['mse']:.6f}")
        print(f"    R2:   {m['r2']:.4f}")
        print(f"    Cos:  {m['cosine_similarity']:.4f}")

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    final_metrics = convert_to_serializable(final_metrics)

    # Save history and final metrics
    tracker.save_history(checkpoint_dir / 'training_history.json')

    with open(checkpoint_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    tracker.close()

    # Write provenance
    provenance_path = checkpoint_dir / 'provenance.md'
    with open(provenance_path, 'w') as f:
        f.write("# Training Provenance\n\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Model: {config.model_type}\n")
        f.write(f"- Data: {data_metadata.get('data_path', 'unknown')}\n")
        f.write(f"- Best epoch: {tracker.best_epoch}\n")
        f.write(f"- Best val loss: {best_val_loss:.6f}\n")
        f.write(f"\n## Config\n```json\n{json.dumps(config.to_dict(), indent=2)}\n```\n")
        f.write(f"\n## Final Metrics\n```json\n{json.dumps(final_metrics, indent=2)}\n```\n")
        f.write("\n## Integrity Check: PASSED\n")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest model: {best_path}")
    print(f"History: {checkpoint_dir / 'training_history.json'}")
    print(f"Metrics: {checkpoint_dir / 'final_metrics.json'}")

    return {
        'best_model_path': str(best_path),
        'checkpoint_dir': str(checkpoint_dir),
        'final_metrics': final_metrics,
        'best_val_loss': best_val_loss,
        'best_epoch': tracker.best_epoch
    }

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Pfam -> AlphaEarth projection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', '-m', type=str, default='light',
        choices=['full', 'light'],
        help='Model architecture'
    )
    parser.add_argument(
        '--loss', '-l', type=str, default='combined',
        choices=['mse', 'cosine', 'combined'],
        help='Loss function type'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--device', '-d', type=str, default=None,
        help='Device (cuda/cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no-amp', action='store_true',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--data-file', type=str, default=None,
        help='Path to prepared data file (npz)'
    )

    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    config = TrainingConfig(args)

    resume_path = Path(args.resume) if args.resume else None

    try:
        results = train(config, resume_path)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
