#!/usr/bin/env python3
"""
==============================================================================
Training: Environment → Pfam Prediction Model
==============================================================================

Purpose: Train neural network to predict Pfam domain profiles from
         environmental features (GEE + AlphaEarth embeddings).

Input:  94 dimensions (30 GEE + 64 AlphaEarth)
Output: ~9600 dimensions (Pfam CLR profiles)

Models:
  - light:  94 → 256 → 512 → 1024 → 2048 → Pfam
  - full:   94 → 512 → 1024 → 2048 → 4096 → Pfam

Author: TARA-LA4SR Analysis Pipeline
Date: 2026-01-25
==============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import json
import argparse
import socket
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Environment Detection
# =============================================================================

def get_base_dir() -> Path:
    """Detect environment and return appropriate base directory."""
    hostname = socket.gethostname()
    if any(x in hostname for x in ['cn', 'gpu', 'dn', 'jubail', 'login', 'fast']):
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    else:
        return Path("/media/drn2/External/TARA-Oceans")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Training configuration."""
    # Model
    model_type: str = 'light'  # 'light' or 'full'
    dropout: float = 0.2

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-6

    # Loss
    loss_type: str = 'mse'  # Only MSE makes sense for high-dim regression

    # Data
    dataset: str = 'pythia'  # 'pythia' or 'algagpt'
    data_file: Optional[str] = None

    # System
    device: str = 'cuda'
    seed: int = 42

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

# =============================================================================
# Models
# =============================================================================

class EnvToPfamLight(nn.Module):
    """Light model: Environment → Pfam prediction."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        # Expanding architecture for high-dim output
        # 94 → 256 → 512 → 1024 → 2048 → output
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(2048, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EnvToPfamFull(nn.Module):
    """Full model: Environment → Pfam prediction with more capacity."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        # Larger expanding architecture
        # 94 → 512 → 1024 → 2048 → 4096 → output
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(4096, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def get_model(model_type: str, input_dim: int, output_dim: int, dropout: float = 0.2) -> nn.Module:
    """Model factory."""
    if model_type == 'light':
        return EnvToPfamLight(input_dim, output_dim, dropout)
    elif model_type == 'full':
        return EnvToPfamFull(input_dim, output_dim, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# Training
# =============================================================================

class Trainer:
    """Model trainer with early stopping."""

    def __init__(self, model: nn.Module, config: Config, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.criterion = nn.MSELoss()

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train': [], 'val': []}

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'loss': total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)

        # Calculate metrics
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))

        # R² per output dimension
        ss_res = np.sum((targets - preds) ** 2, axis=0)
        ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
        r2_per_dim = 1 - ss_res / (ss_tot + 1e-8)

        # Overall R²
        ss_res_total = np.sum((targets - preds) ** 2)
        ss_tot_total = np.sum((targets - np.mean(targets)) ** 2)
        r2_overall = 1 - ss_res_total / (ss_tot_total + 1e-8)

        # Cosine similarity (row-wise, then average)
        dot = np.sum(preds * targets, axis=1)
        norm_pred = np.linalg.norm(preds, axis=1) + 1e-8
        norm_target = np.linalg.norm(targets, axis=1) + 1e-8
        cos_sim = np.mean(dot / (norm_pred * norm_target))

        return {
            'loss': float(total_loss / n_batches),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2_overall),
            'r2_mean': float(np.mean(r2_per_dim)),
            'r2_median': float(np.median(r2_per_dim)),
            'r2_min': float(np.min(r2_per_dim)),
            'r2_max': float(np.max(r2_per_dim)),
            'cosine_similarity': float(cos_sim),
        }

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            checkpoint_dir: Path) -> Dict:
        """Full training loop."""

        print(f"\nTraining {self.config.model_type} model...")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")

        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            self.scheduler.step(val_metrics['loss'])

            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self._save_checkpoint(checkpoint_dir / 'best_model.pt')
            else:
                self.patience_counter += 1

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:3d}: train_loss={train_metrics['loss']:.6f}, "
                      f"val_loss={val_metrics['loss']:.6f}, val_r2={val_metrics['r2']:.4f}, "
                      f"val_cos={val_metrics['cosine_similarity']:.4f}, lr={lr:.2e}")

            if self.patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        return self.history

    def _save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)

# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_file: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Load prepared data and create DataLoaders."""

    print(f"Loading data: {data_file}")
    data = np.load(data_file, allow_pickle=True)

    # Create tensors
    X_train = torch.FloatTensor(data['X_train'])
    X_val = torch.FloatTensor(data['X_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_train = torch.FloatTensor(data['y_train'])
    y_val = torch.FloatTensor(data['y_val'])
    y_test = torch.FloatTensor(data['y_test'])

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    info = {
        'input_dim': X_train.shape[1],
        'output_dim': y_train.shape[1],
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'input_features': data['input_feature_names'].tolist() if 'input_feature_names' in data else None,
        'output_features': data['output_feature_names'].tolist() if 'output_feature_names' in data else None,
    }

    print(f"  Input dim: {info['input_dim']}")
    print(f"  Output dim: {info['output_dim']}")
    print(f"  Train/Val/Test: {info['n_train']}/{info['n_val']}/{info['n_test']}")

    return train_loader, val_loader, test_loader, info

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Environment → Pfam model')
    parser.add_argument('--model', type=str, default='light', choices=['light', 'full'])
    parser.add_argument('--dataset', type=str, default='pythia', choices=['pythia', 'algagpt'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-file', type=str, default=None)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Configuration
    config = Config(
        model_type=args.model,
        dataset=args.dataset,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        data_file=args.data_file
    )

    # Device
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Paths
    base_dir = get_base_dir()
    data_dir = base_dir / "MANUSCRIPT" / "data"
    checkpoint_dir = base_dir / "MANUSCRIPT" / "checkpoints"

    # Find data file
    if config.data_file:
        data_file = Path(config.data_file)
    else:
        pattern = f"env_to_pfam_{config.dataset}_*.npz"
        data_files = sorted(data_dir.glob(pattern))
        if not data_files:
            print(f"ERROR: No data file found matching {pattern}")
            return
        data_file = data_files[-1]  # Most recent

    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lr_str = f"{config.learning_rate:.0e}".replace('-', 'm').replace('+', 'p')
    run_name = f"env2pfam_{config.dataset}_{config.model_type}_{lr_str}_{timestamp}"
    run_dir = checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Environment → Pfam Training")
    print(f"{'='*60}")
    print(f"Run: {run_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model_type}")

    # Load data
    train_loader, val_loader, test_loader, data_info = load_data(data_file, config.batch_size)

    # Create model
    model = get_model(
        config.model_type,
        data_info['input_dim'],
        data_info['output_dim'],
        config.dropout
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Save config
    config_dict = config.to_dict()
    config_dict.update(data_info)
    config_dict['n_parameters'] = n_params
    config_dict['run_name'] = run_name
    config_dict['data_file'] = str(data_file)

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Train
    trainer = Trainer(model, config, device)
    history = trainer.fit(train_loader, val_loader, run_dir)

    # Load best model and evaluate on test
    checkpoint = torch.load(run_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model = model

    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    test_metrics = trainer.evaluate(test_loader)

    # Final metrics
    final_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Save training history
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Train: MSE={train_metrics['mse']:.6f}, R²={train_metrics['r2']:.4f}, Cos={train_metrics['cosine_similarity']:.4f}")
    print(f"Val:   MSE={val_metrics['mse']:.6f}, R²={val_metrics['r2']:.4f}, Cos={val_metrics['cosine_similarity']:.4f}")
    print(f"Test:  MSE={test_metrics['mse']:.6f}, R²={test_metrics['r2']:.4f}, Cos={test_metrics['cosine_similarity']:.4f}")
    print(f"\nCheckpoint: {run_dir}")

if __name__ == "__main__":
    main()
