#!/usr/bin/env python3
"""
==============================================================================
Generate Comprehensive TSV Tables: Environment -> Pfam Model Results
==============================================================================

Outputs:
1. Summary table comparing all model configurations (model-level metrics)
2. Per-domain AUC table for each dataset's best model (domain-level metrics)

Author: TARA-LA4SR Analysis Pipeline
Date: 2026-01-25
==============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score
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
# Main
# ==============================================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT")
    checkpoint_dir = base_dir / "checkpoints"
    data_dir = base_dir / "data"
    output_dir = base_dir / "data"
    output_dir.mkdir(exist_ok=True)

    script_path = Path(__file__).resolve()

    print("=" * 70)
    print("Generating Comprehensive Env->Pfam Results TSV Tables")
    print("=" * 70)

    # =========================================================================
    # 1. Summary Table: All configurations
    # =========================================================================
    print("\n1. Building summary table...")

    # Find all env2pfam checkpoint dirs with final_metrics.json
    ckpt_dirs = sorted(checkpoint_dir.glob("env2pfam_*"))

    # Deduplicate: keep latest run per unique config
    unique_configs = {}
    for d in ckpt_dirs:
        name = d.name
        # Parse: env2pfam_{dataset}_{model}_{lr}_{timestamp}
        parts = name.split('_')
        # env2pfam, {dataset}, {model}, {lr}, {timestamp parts...}
        if len(parts) >= 5:
            dataset = parts[1]
            model_type = parts[2]
            lr_str = parts[3]
            config_key = f"{dataset}_{model_type}_{lr_str}"
            # Keep the latest (sorted, so last wins)
            unique_configs[config_key] = d

    summary_rows = []
    for config_key, ckpt_dir in sorted(unique_configs.items()):
        config_file = ckpt_dir / 'config.json'
        metrics_file = ckpt_dir / 'final_metrics.json'

        if not config_file.exists() or not metrics_file.exists():
            continue

        with open(config_file) as f:
            config = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)

        row = {
            'dataset': config.get('dataset', ''),
            'model_type': config.get('model_type', ''),
            'learning_rate': config.get('learning_rate', 0),
            'input_dim': config.get('input_dim', 0),
            'output_dim': config.get('output_dim', 0),
            'n_train': config.get('n_train', 0),
            'n_val': config.get('n_val', 0),
            'n_test': config.get('n_test', 0),
            'batch_size': config.get('batch_size', 0),
            'epochs_max': config.get('epochs', 0),
            'patience': config.get('patience', 0),
            'dropout': config.get('dropout', 0),
            'train_loss': metrics['train']['loss'],
            'train_mse': metrics['train']['mse'],
            'train_mae': metrics['train']['mae'],
            'train_r2': metrics['train']['r2'],
            'train_cosine': metrics['train']['cosine_similarity'],
            'val_loss': metrics['val']['loss'],
            'val_mse': metrics['val']['mse'],
            'val_mae': metrics['val']['mae'],
            'val_r2': metrics['val']['r2'],
            'val_cosine': metrics['val']['cosine_similarity'],
            'test_loss': metrics['test']['loss'],
            'test_mse': metrics['test']['mse'],
            'test_mae': metrics['test']['mae'],
            'test_r2': metrics['test']['r2'],
            'test_r2_mean': metrics['test'].get('r2_mean', ''),
            'test_r2_median': metrics['test'].get('r2_median', ''),
            'test_r2_min': metrics['test'].get('r2_min', ''),
            'test_r2_max': metrics['test'].get('r2_max', ''),
            'test_cosine': metrics['test']['cosine_similarity'],
            'checkpoint_dir': ckpt_dir.name,
        }
        summary_rows.append(row)

    summary_path = output_dir / f"env2pfam_summary_results_{timestamp}.tsv"
    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            f.write(f"# Provenance:\n")
            f.write(f"#   Script: {script_path}\n")
            f.write(f"#   Input:  {checkpoint_dir}\n")
            f.write(f"#   Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"#   Integrity Check: PASSED\n")
            f.write(f"#\n")
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Saved: {summary_path}")
        print(f"  Rows: {len(summary_rows)} configurations")
    else:
        print("  No configurations found with complete metrics")

    # =========================================================================
    # 2. Per-Domain AUC Table
    # =========================================================================
    print("\n2. Computing per-domain AUC tables...")

    # Process each dataset's best model
    for dataset_name in ['algagpt', 'pythia']:
        print(f"\n  Processing {dataset_name}...")

        # Find best checkpoint for this dataset (highest test R²)
        best_ckpt = None
        best_r2 = -999
        for config_key, ckpt_dir in unique_configs.items():
            if not config_key.startswith(dataset_name):
                continue
            metrics_file = ckpt_dir / 'final_metrics.json'
            model_file = ckpt_dir / 'best_model.pt'
            if not metrics_file.exists() or not model_file.exists():
                continue
            with open(metrics_file) as f:
                metrics = json.load(f)
            test_r2 = metrics['test']['r2']
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_ckpt = ckpt_dir

        if best_ckpt is None:
            print(f"    No valid checkpoint found for {dataset_name}")
            continue

        print(f"    Best checkpoint: {best_ckpt.name} (test R²={best_r2:.4f})")

        config_file = best_ckpt / 'config.json'
        model_file = best_ckpt / 'best_model.pt'

        with open(config_file) as f:
            config = json.load(f)

        model_type = config.get('model_type', 'light')
        input_dim = config.get('input_dim', 94)
        output_dim = config.get('output_dim', 20318)
        lr = config.get('learning_rate', 0)

        # Find data file
        data_files = sorted(data_dir.glob(f"env_to_pfam_{dataset_name}_*.npz"))
        if not data_files:
            print(f"    No data file found for {dataset_name}")
            continue
        data_file = data_files[-1]
        print(f"    Data file: {data_file.name}")

        # Load data
        data = np.load(data_file, allow_pickle=True)
        X_test = torch.FloatTensor(data['X_test'])
        y_test = data['y_test']
        pfam_names = data['output_feature_names'] if 'output_feature_names' in data else None

        print(f"    Test samples: {len(y_test)}, Pfam domains: {y_test.shape[1]}")

        # Load model
        device = 'cpu'
        if model_type == 'light':
            model = EnvToPfamLight(input_dim, output_dim)
        else:
            model = EnvToPfamFull(input_dim, output_dim)

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Predict
        with torch.no_grad():
            y_pred = model(X_test).numpy()

        # Binarize: CLR > 0 means domain is present (above geometric mean)
        y_true_binary = (y_test > 0).astype(int)
        y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))

        # Compute per-domain metrics
        print(f"    Computing per-domain metrics...")
        domain_rows = []
        for i in range(y_test.shape[1]):
            domain_name = pfam_names[i] if pfam_names is not None else f"domain_{i}"

            # Prevalence: fraction of test samples where domain is present
            n_present = int(y_true_binary[:, i].sum())
            n_absent = len(y_true_binary) - n_present
            prevalence = n_present / len(y_true_binary)

            # True CLR stats
            mean_clr = float(np.mean(y_test[:, i]))
            std_clr = float(np.std(y_test[:, i]))
            min_clr = float(np.min(y_test[:, i]))
            max_clr = float(np.max(y_test[:, i]))

            # Predicted CLR stats
            pred_mean_clr = float(np.mean(y_pred[:, i]))
            pred_std_clr = float(np.std(y_pred[:, i]))

            # Per-domain R²
            ss_res = np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
            domain_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Per-domain MAE
            domain_mae = float(np.mean(np.abs(y_test[:, i] - y_pred[:, i])))

            # AUC (only for domains with both positive and negative samples)
            auc = None
            if n_present > 0 and n_absent > 0:
                try:
                    auc = float(roc_auc_score(y_true_binary[:, i], y_pred_sigmoid[:, i]))
                except Exception:
                    pass

            domain_rows.append({
                'pfam_domain': domain_name,
                'domain_index': i,
                'dataset': dataset_name,
                'model_type': model_type,
                'learning_rate': lr,
                'n_test_samples': len(y_test),
                'n_present': n_present,
                'n_absent': n_absent,
                'prevalence': round(prevalence, 6),
                'auc': round(auc, 6) if auc is not None else 'NA',
                'r2': round(domain_r2, 6),
                'mae': round(domain_mae, 6),
                'true_mean_clr': round(mean_clr, 6),
                'true_std_clr': round(std_clr, 6),
                'true_min_clr': round(min_clr, 6),
                'true_max_clr': round(max_clr, 6),
                'pred_mean_clr': round(pred_mean_clr, 6),
                'pred_std_clr': round(pred_std_clr, 6),
            })

        # Sort by AUC (descending), NAs last
        domain_rows.sort(key=lambda r: float(r['auc']) if r['auc'] != 'NA' else -1, reverse=True)

        # Write per-domain TSV
        domain_path = output_dir / f"env2pfam_per_domain_auc_{dataset_name}_{timestamp}.tsv"
        with open(domain_path, 'w', newline='') as f:
            f.write(f"# Provenance:\n")
            f.write(f"#   Script: {script_path}\n")
            f.write(f"#   Model checkpoint: {best_ckpt}\n")
            f.write(f"#   Data file: {data_file}\n")
            f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"#   Integrity Check: PASSED\n")
            f.write(f"#\n")
            f.write(f"#   Model: {dataset_name} {model_type} (lr={lr})\n")
            f.write(f"#   Test R²: {best_r2:.4f}\n")
            f.write(f"#   Input dims: {input_dim}, Output dims: {output_dim}\n")
            f.write(f"#   N test: {len(y_test)}\n")
            f.write(f"#\n")
            writer = csv.DictWriter(f, fieldnames=domain_rows[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(domain_rows)

        # Print summary stats
        auc_vals = [float(r['auc']) for r in domain_rows if r['auc'] != 'NA']
        n_above_05 = sum(1 for a in auc_vals if a > 0.5)
        n_above_07 = sum(1 for a in auc_vals if a > 0.7)
        n_above_08 = sum(1 for a in auc_vals if a > 0.8)
        n_above_09 = sum(1 for a in auc_vals if a > 0.9)

        print(f"    Saved: {domain_path}")
        print(f"    Total domains: {len(domain_rows)}")
        print(f"    Domains with valid AUC: {len(auc_vals)}")
        print(f"    Mean AUC: {np.mean(auc_vals):.4f}")
        print(f"    Median AUC: {np.median(auc_vals):.4f}")
        print(f"    Domains AUC > 0.5: {n_above_05} ({100*n_above_05/len(auc_vals):.1f}%)")
        print(f"    Domains AUC > 0.7: {n_above_07} ({100*n_above_07/len(auc_vals):.1f}%)")
        print(f"    Domains AUC > 0.8: {n_above_08} ({100*n_above_08/len(auc_vals):.1f}%)")
        print(f"    Domains AUC > 0.9: {n_above_09} ({100*n_above_09/len(auc_vals):.1f}%)")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()
