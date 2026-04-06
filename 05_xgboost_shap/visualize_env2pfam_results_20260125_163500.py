#!/usr/bin/env python3
"""
==============================================================================
Visualization: Environment → Pfam Model Results
==============================================================================

Generates:
1. Learning curves (train/val loss vs epochs)
2. AUC analysis (binarized Pfam presence/absence)

Author: TARA-LA4SR Analysis Pipeline
Date: 2026-01-25
==============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Model Definition (must match training script)
# ==============================================================================

class EnvToPfamLight(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
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

    def forward(self, x):
        return self.network(x)

class EnvToPfamFull(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
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

    def forward(self, x):
        return self.network(x)

# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_learning_curves(checkpoint_dirs, output_path):
    """Plot learning curves for all checkpoints."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'algagpt': '#2ecc71', 'pythia': '#3498db'}
    linestyles = {'light': '-', 'full': '--'}

    for ckpt_dir in checkpoint_dirs:
        history_file = ckpt_dir / 'training_history.json'
        config_file = ckpt_dir / 'config.json'

        if not history_file.exists() or not config_file.exists():
            continue

        with open(history_file) as f:
            history = json.load(f)
        with open(config_file) as f:
            config = json.load(f)

        dataset = config.get('dataset', 'unknown')
        model_type = config.get('model_type', 'unknown')
        lr = config.get('learning_rate', 0)

        label = f"{dataset}-{model_type} (lr={lr})"
        color = colors.get(dataset, 'gray')
        ls = linestyles.get(model_type, '-')

        train_loss = [e['loss'] for e in history['train']]
        val_loss = [e['loss'] for e in history['val']]
        epochs = range(1, len(train_loss) + 1)

        # Train loss
        axes[0, 0].plot(epochs, train_loss, color=color, linestyle=ls,
                       alpha=0.7, label=label)

        # Val loss
        axes[0, 1].plot(epochs, val_loss, color=color, linestyle=ls,
                       alpha=0.7, label=label)

        # Val R² (if available)
        if 'r2' in history['val'][0]:
            val_r2 = [e['r2'] for e in history['val']]
            axes[1, 0].plot(epochs, val_r2, color=color, linestyle=ls,
                           alpha=0.7, label=label)

        # Val Cosine (if available)
        if 'cosine_similarity' in history['val'][0]:
            val_cos = [e['cosine_similarity'] for e in history['val']]
            axes[1, 1].plot(epochs, val_cos, color=color, linestyle=ls,
                           alpha=0.7, label=label)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss (MSE)')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val Loss (MSE)')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Val R²')
    axes[1, 0].set_title('Validation R²')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Cosine Similarity')
    axes[1, 1].set_title('Validation Cosine Similarity')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

def compute_auc_metrics(checkpoint_dir, data_dir, device='cpu'):
    """Compute AUC for binarized Pfam predictions."""

    config_file = checkpoint_dir / 'config.json'
    model_file = checkpoint_dir / 'best_model.pt'

    if not config_file.exists() or not model_file.exists():
        return None

    with open(config_file) as f:
        config = json.load(f)

    dataset = config.get('dataset', 'pythia')
    model_type = config.get('model_type', 'light')
    input_dim = config.get('input_dim', 94)
    output_dim = config.get('output_dim', 17245)

    # Find data file
    data_files = sorted(data_dir.glob(f"env_to_pfam_{dataset}_*.npz"))
    if not data_files:
        print(f"No data file found for {dataset}")
        return None
    data_file = data_files[-1]

    # Load data
    data = np.load(data_file)
    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']

    # Load model
    if model_type == 'light':
        model = EnvToPfamLight(input_dim, output_dim)
    else:
        model = EnvToPfamFull(input_dim, output_dim)

    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Predict
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = model(X_test).cpu().numpy()

    # Binarize: CLR > 0 means "present" (above geometric mean)
    y_true_binary = (y_test > 0).astype(int)
    y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))  # Convert to probabilities

    # Compute per-domain AUC (only for domains with variance)
    aucs = []
    valid_domains = 0
    for i in range(y_test.shape[1]):
        if y_true_binary[:, i].sum() > 0 and y_true_binary[:, i].sum() < len(y_true_binary):
            try:
                auc = roc_auc_score(y_true_binary[:, i], y_pred_sigmoid[:, i])
                aucs.append(auc)
                valid_domains += 1
            except:
                pass

    if not aucs:
        return None

    results = {
        'dataset': dataset,
        'model_type': model_type,
        'n_test_samples': len(y_test),
        'n_pfam_domains': output_dim,
        'valid_domains_for_auc': valid_domains,
        'mean_auc': np.mean(aucs),
        'median_auc': np.median(aucs),
        'std_auc': np.std(aucs),
        'min_auc': np.min(aucs),
        'max_auc': np.max(aucs),
        'auc_per_domain': aucs
    }

    return results

def plot_auc_comparison(auc_results, output_path):
    """Plot AUC comparison between models."""

    # Deduplicate: keep only unique dataset+model_type combinations (best AUC)
    unique_results = {}
    for r in auc_results:
        key = f"{r['dataset']}_{r['model_type']}"
        if key not in unique_results or r['mean_auc'] > unique_results[key]['mean_auc']:
            unique_results[key] = r
    auc_results = list(unique_results.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of mean AUC
    labels = [f"{r['dataset']}\n{r['model_type']}" for r in auc_results]
    mean_aucs = [r['mean_auc'] for r in auc_results]
    std_aucs = [r['std_auc'] for r in auc_results]

    colors = ['#2ecc71' if 'algagpt' in r['dataset'] else '#3498db' for r in auc_results]

    x_pos = np.arange(len(labels))
    bars = axes[0].bar(x_pos, mean_aucs, yerr=std_aucs, capsize=5, color=colors, alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Mean AUC')
    axes[0].set_title('Mean AUC Across Pfam Domains')
    axes[0].set_ylim(0.5, 1.0)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add values on bars (with offset for visibility)
    for bar, val in zip(bars, mean_aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Histogram of per-domain AUC (only unique configs)
    for r in auc_results:
        color = '#2ecc71' if 'algagpt' in r['dataset'] else '#3498db'
        label = f"{r['dataset']}-{r['model_type']}"
        axes[1].hist(r['auc_per_domain'], bins=50, alpha=0.5, color=color, label=label)

    axes[1].set_xlabel('AUC per Pfam Domain')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Per-Domain AUC')
    axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

def plot_roc_curves(checkpoint_dir, data_dir, output_path, n_domains=5, device='cpu'):
    """Plot ROC curves for top domains."""

    config_file = checkpoint_dir / 'config.json'
    model_file = checkpoint_dir / 'best_model.pt'

    if not config_file.exists() or not model_file.exists():
        return

    with open(config_file) as f:
        config = json.load(f)

    dataset = config.get('dataset', 'pythia')
    model_type = config.get('model_type', 'light')
    input_dim = config.get('input_dim', 94)
    output_dim = config.get('output_dim', 17245)

    # Find data file
    data_files = sorted(data_dir.glob(f"env_to_pfam_{dataset}_*.npz"))
    if not data_files:
        return
    data_file = data_files[-1]

    # Load data
    data = np.load(data_file)
    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']
    pfam_names = data['output_feature_names'] if 'output_feature_names' in data else None

    # Load model
    if model_type == 'light':
        model = EnvToPfamLight(input_dim, output_dim)
    else:
        model = EnvToPfamFull(input_dim, output_dim)

    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Predict
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = model(X_test).cpu().numpy()

    # Binarize
    y_true_binary = (y_test > 0).astype(int)
    y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))

    # Find domains with best AUC
    aucs = []
    for i in range(y_test.shape[1]):
        if y_true_binary[:, i].sum() > 5 and y_true_binary[:, i].sum() < len(y_true_binary) - 5:
            try:
                auc = roc_auc_score(y_true_binary[:, i], y_pred_sigmoid[:, i])
                aucs.append((i, auc))
            except:
                pass

    aucs.sort(key=lambda x: x[1], reverse=True)
    top_domains = aucs[:n_domains]

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_domains))

    for (idx, auc), color in zip(top_domains, colors):
        fpr, tpr, _ = roc_curve(y_true_binary[:, idx], y_pred_sigmoid[:, idx])
        label = pfam_names[idx] if pfam_names is not None else f"Domain {idx}"
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - Top {n_domains} Pfam Domains\n({dataset}-{model_type})')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT")
    checkpoint_dir = base_dir / "checkpoints"
    data_dir = base_dir / "data"
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Environment → Pfam Results Visualization")
    print("=" * 60)

    # Find env2pfam checkpoints
    ckpt_dirs = sorted(checkpoint_dir.glob("env2pfam_*"))
    print(f"Found {len(ckpt_dirs)} checkpoint directories")

    if not ckpt_dirs:
        print("No checkpoints found. Run rsync first.")
        return

    # 1. Learning curves
    print("\n1. Plotting learning curves...")
    learning_curves_path = figures_dir / f"env2pfam_learning_curves_{timestamp}.png"
    plot_learning_curves(ckpt_dirs, learning_curves_path)

    # 2. AUC analysis
    print("\n2. Computing AUC metrics...")
    auc_results = []
    for ckpt_dir in ckpt_dirs:
        print(f"  Processing: {ckpt_dir.name}")
        result = compute_auc_metrics(ckpt_dir, data_dir, device='cpu')
        if result:
            auc_results.append(result)
            print(f"    Mean AUC: {result['mean_auc']:.4f}")

    if auc_results:
        # Plot AUC comparison
        auc_comparison_path = figures_dir / f"env2pfam_auc_comparison_{timestamp}.png"
        plot_auc_comparison(auc_results, auc_comparison_path)

        # Save AUC results
        auc_results_json = figures_dir / f"env2pfam_auc_results_{timestamp}.json"
        # Remove numpy arrays for JSON serialization
        for r in auc_results:
            r['auc_per_domain'] = [float(x) for x in r['auc_per_domain'][:100]]  # Keep first 100
        with open(auc_results_json, 'w') as f:
            json.dump(auc_results, f, indent=2)
        print(f"Saved: {auc_results_json}")

    # 3. ROC curves for best model
    print("\n3. Plotting ROC curves...")
    if ckpt_dirs:
        # Find best checkpoint (algagpt preferred)
        algagpt_dirs = [d for d in ckpt_dirs if 'algagpt' in d.name]
        best_ckpt = algagpt_dirs[0] if algagpt_dirs else ckpt_dirs[0]

        roc_path = figures_dir / f"env2pfam_roc_curves_{timestamp}.png"
        plot_roc_curves(best_ckpt, data_dir, roc_path, n_domains=5)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
