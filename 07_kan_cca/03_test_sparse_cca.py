#!/usr/bin/env python3
"""
03_test_sparse_cca.py — Test sparse CCA implementation on synthetic data.

Tests:
1. Convergence: sparse CCA converges and produces valid correlations in [-1, 1]
2. Sparsity: smaller L1 constraint produces sparser vectors
3. Comparison: dense sparse CCA approximates sklearn CCA
4. Multi-component: deflation produces decreasing correlations
"""

import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA

sys.path.insert(0, str(Path(__file__).parent))
from sparse_cca import sparse_cca

def make_synthetic(n=200, p=50, q=20, n_latent=3, noise=0.3, seed=42):
    """Generate synthetic data with known latent structure."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, n_latent)
    A = np.zeros((n_latent, p))
    for k in range(n_latent):
        idx = rng.choice(p, size=10, replace=False)
        A[k, idx] = rng.randn(10)
    B = rng.randn(n_latent, q) * 0.5
    X = Z @ A + noise * rng.randn(n, p)
    Y = Z @ B + noise * rng.randn(n, q)
    return X, Y

def test_convergence():
    """Test that sparse CCA converges with valid correlations."""
    print("Test 1: Convergence...")
    X, Y = make_synthetic()
    p, q = X.shape[1], Y.shape[1]

    c_u = 3.0  # sqrt(50) ~= 7.07, so 3.0 is moderate
    c_v = 2.5  # sqrt(20) ~= 4.47, so 2.5 is moderate

    U, V, corrs = sparse_cca(X, Y, c_u, c_v, n_components=3)

    print(f"  Correlations: {corrs}")
    assert all(np.isfinite(corrs)), "Non-finite correlations!"
    assert all(-1 <= c <= 1 for c in corrs), f"Correlations outside [-1,1]: {corrs}"
    assert corrs[0] > 0.3, f"First correlation too low: {corrs[0]:.4f}"
    print("  PASSED")

def test_sparsity():
    """Test that smaller L1 constraint produces sparser vectors."""
    print("\nTest 2: Sparsity...")
    X, Y = make_synthetic()
    p, q = X.shape[1], Y.shape[1]

    # Very sparse
    c_u_sparse = 1.5
    c_v_sparse = 1.5
    U_sparse, V_sparse, corrs_sparse = sparse_cca(
        X, Y, c_u_sparse, c_v_sparse, n_components=1)

    # Dense (no effective sparsity)
    c_u_dense = np.sqrt(p)
    c_v_dense = np.sqrt(q)
    U_dense, V_dense, corrs_dense = sparse_cca(
        X, Y, c_u_dense, c_v_dense, n_components=1)

    nnz_sparse = np.sum(np.abs(U_sparse[:, 0]) > 1e-10)
    nnz_dense = np.sum(np.abs(U_dense[:, 0]) > 1e-10)

    print(f"  Sparse u: {nnz_sparse}/{p} nonzero (c_u={c_u_sparse})")
    print(f"  Dense u:  {nnz_dense}/{p} nonzero (c_u={c_u_dense:.2f})")
    print(f"  Sparse corr: {corrs_sparse[0]:.4f}, Dense corr: {corrs_dense[0]:.4f}")

    assert nnz_sparse < nnz_dense, \
        f"Sparse ({nnz_sparse}) should have fewer nonzero than dense ({nnz_dense})!"
    print("  PASSED")

def test_vs_sklearn():
    """Compare dense sparse CCA to sklearn CCA."""
    print("\nTest 3: Comparison with sklearn CCA...")
    X, Y = make_synthetic(n=200, p=20, q=10)
    p, q = X.shape[1], Y.shape[1]

    U_pmd, V_pmd, corrs_pmd = sparse_cca(
        X, Y, np.sqrt(p), np.sqrt(q), n_components=3)

    cca = CCA(n_components=3, max_iter=1000, tol=1e-6)
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    cca.fit(X_c, Y_c)
    Xs, Ys = cca.transform(X_c, Y_c)
    corrs_sklearn = np.array([
        np.corrcoef(Xs[:, k], Ys[:, k])[0, 1] for k in range(3)
    ])

    print(f"  PMD correlations:    {corrs_pmd}")
    print(f"  sklearn correlations: {corrs_sklearn}")

    # First correlation should be similar (within 0.20)
    diff = abs(abs(corrs_pmd[0]) - abs(corrs_sklearn[0]))
    print(f"  |Difference| (comp 1): {diff:.4f}")
    assert diff < 0.20, f"First component too different: {diff:.4f}"
    print("  PASSED")

def test_multicomponent():
    """Test multi-component deflation produces non-increasing correlations."""
    print("\nTest 4: Multi-component deflation...")
    X, Y = make_synthetic()
    p, q = X.shape[1], Y.shape[1]

    c_u = np.sqrt(p) * 0.5
    c_v = np.sqrt(q) * 0.5
    U, V, corrs = sparse_cca(X, Y, c_u, c_v, n_components=3)

    print(f"  Correlations: {corrs}")
    for k in range(len(corrs) - 1):
        # Allow small tolerance for deflation imprecision
        assert corrs[k] >= corrs[k + 1] - 0.05, \
            f"Correlations not roughly decreasing: {corrs[k]:.4f} < {corrs[k+1]:.4f}"
    print("  PASSED")

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 03_test_sparse_cca.py on {socket.gethostname()}")
    print("=" * 60)

    test_convergence()
    test_sparsity()
    test_vs_sklearn()
    test_multicomponent()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
