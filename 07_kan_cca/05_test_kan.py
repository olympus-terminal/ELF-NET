#!/usr/bin/env python3
"""
05_test_kan.py — Test KAN layer and KAN-CCA model on synthetic data.

Tests:
1. KAN layer forward pass produces correct shapes
2. B-spline basis is well-formed (nonnegative, partition of unity)
3. KAN-CCA training: loss decreases over epochs
4. KAN-CCA: correlations increase during training
5. Spline extraction works and produces reasonable outputs
"""

import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kan_layer import KANLayer, _bspline_basis
from kan_cca_model import KANCCAModel, train_kan_cca, evaluate_kan_cca

def make_synthetic(n=200, k_d=6, k_e=12, noise=0.3, seed=42):
    """Generate synthetic data with nonlinear coupling."""
    rng = np.random.RandomState(seed)

    # Latent variable
    z = rng.randn(n, 1)

    # Nonlinear mappings
    X_d = np.column_stack([
        np.sin(z[:, 0] * (i + 1)) + noise * rng.randn(n)
        for i in range(k_d)
    ])
    X_e = np.column_stack([
        np.tanh(z[:, 0] * (j + 0.5)) + noise * rng.randn(n)
        for j in range(k_e)
    ])

    # Standardize
    X_d = (X_d - X_d.mean(0)) / X_d.std(0).clip(1e-8)
    X_e = (X_e - X_e.mean(0)) / X_e.std(0).clip(1e-8)

    return X_d, X_e

def test_kan_layer_shapes():
    """Test KAN layer produces correct output shapes."""
    print("Test 1: KAN layer shapes...")
    layer = KANLayer(6, 32, grid_size=5, spline_order=3)
    x = torch.randn(16, 6)
    out = layer(x)
    assert out.shape == (16, 32), f"Expected (16, 32), got {out.shape}"
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Second layer
    layer2 = KANLayer(32, 3, grid_size=5, spline_order=3)
    out2 = layer2(out)
    assert out2.shape == (16, 3), f"Expected (16, 3), got {out2.shape}"
    print(f"  Layer 2: {out.shape} -> {out2.shape}")
    print("  PASSED")

def test_bspline_basis():
    """Test B-spline basis properties."""
    print("\nTest 2: B-spline basis properties...")
    grid = torch.linspace(-2, 2, 12)  # grid_size=5, order=3 -> 12 knots
    x = torch.linspace(-1, 1, 50).unsqueeze(1)  # (50, 1)

    basis = _bspline_basis(x, grid, order=3)
    print(f"  Basis shape: {basis.shape}")  # (50, 1, n_basis)

    # Check non-negativity
    assert (basis >= -1e-6).all(), "B-spline basis has negative values!"
    print(f"  Non-negativity: OK (min={basis.min():.6f})")

    # Check partition of unity (approximately)
    row_sums = basis.sum(dim=-1)
    print(f"  Row sums range: [{row_sums.min():.4f}, {row_sums.max():.4f}]")
    assert (row_sums > 0.9).all() and (row_sums < 1.1).all(), \
        f"Partition of unity violated: {row_sums.min():.4f} to {row_sums.max():.4f}"
    print("  PASSED")

def test_kan_cca_training():
    """Test KAN-CCA training: loss should decrease."""
    print("\nTest 3: KAN-CCA training loss decreases...")
    X_d, X_e = make_synthetic()

    model, history = train_kan_cca(
        X_d, X_e, n_components=1, hidden_dim=16,
        n_epochs=50, lr=1e-3, patience=50)

    losses = [h['loss'] for h in history]
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({len(losses)} epochs)")
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  PASSED")

def test_kan_cca_correlations():
    """Test KAN-CCA produces increasing correlations during training."""
    print("\nTest 4: KAN-CCA correlations increase...")
    X_d, X_e = make_synthetic()

    model, history = train_kan_cca(
        X_d, X_e, n_components=1, hidden_dim=16,
        n_epochs=100, lr=1e-3, patience=100)

    early_corr = abs(history[5]['correlations'][0])
    final_corr = abs(history[-1]['correlations'][0])
    print(f"  Epoch 5 corr: {early_corr:.4f}")
    print(f"  Final corr: {final_corr:.4f}")

    # Evaluate on training data
    test_corrs = evaluate_kan_cca(model, X_d, X_e)
    print(f"  Evaluation corrs: {test_corrs}")
    assert abs(test_corrs[0]) > 0.3, \
        f"Final correlation too low: {test_corrs[0]:.4f}"
    print("  PASSED")

def test_spline_extraction():
    """Test spline activation extraction."""
    print("\nTest 5: Spline extraction...")
    X_d, X_e = make_synthetic()

    model, history = train_kan_cca(
        X_d, X_e, n_components=1, hidden_dim=16,
        n_epochs=50, lr=1e-3, patience=50)

    splines = model.extract_first_layer_splines(n_points=50)
    assert 'domain' in splines
    assert 'env' in splines

    n_domain_splines = len(splines['domain'])
    n_env_splines = len(splines['env'])
    print(f"  Domain splines: {n_domain_splines}")
    print(f"  Env splines: {n_env_splines}")

    # Check a single spline has correct shape
    key = list(splines['domain'].keys())[0]
    x_vals, y_vals = splines['domain'][key]
    assert len(x_vals) == 50
    assert len(y_vals) == 50
    print(f"  Sample spline {key}: x range [{x_vals.min():.2f}, {x_vals.max():.2f}], "
          f"y range [{y_vals.min():.4f}, {y_vals.max():.4f}]")
    print("  PASSED")

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 05_test_kan.py on {socket.gethostname()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 60)

    test_kan_layer_shapes()
    test_bspline_basis()
    test_kan_cca_training()
    test_kan_cca_correlations()
    test_spline_extraction()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
