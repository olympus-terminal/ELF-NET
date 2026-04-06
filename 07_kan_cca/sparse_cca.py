#!/usr/bin/env python3
"""
sparse_cca.py — L1-penalized CCA via Penalized Matrix Decomposition (PMD).

Implements the sparse CCA method of Witten, Tibshirani & Hastie (2009).
Biometrika 96(4): 857-880.

The PMD approach finds sparse canonical vectors u, v that maximize:
    u' S v  where S = X' Y / n
subject to:
    ||u||_2 <= 1, ||v||_2 <= 1, ||u||_1 <= c_u, ||v||_1 <= c_v

where c_u in [1, sqrt(p)] and c_v in [1, sqrt(q)].

Uses cross-covariance deflation for multiple components.
"""

import numpy as np

def _soft_threshold(x, lam):
    """Soft-thresholding operator: sign(x) * max(|x| - lam, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def _project_l1_l2(z, c):
    """Project z onto {x : ||x||_2 <= 1, ||x||_1 <= c}.

    1. L2-normalize z
    2. If ||z_norm||_1 <= c, return z_norm (no sparsity needed)
    3. Otherwise, binary search for soft-threshold lambda that yields
       ||st(z, lam)||_1 / ||st(z, lam)||_2 = c, then L2-normalize
    """
    nrm = np.linalg.norm(z)
    if nrm < 1e-15:
        return z

    z_norm = z / nrm
    if np.sum(np.abs(z_norm)) <= c + 1e-10:
        return z_norm

    lo, hi = 0.0, np.max(np.abs(z))
    for _ in range(200):
        mid = (lo + hi) / 2
        st = _soft_threshold(z, mid)
        st_nrm = np.linalg.norm(st)
        if st_nrm < 1e-15:
            hi = mid
            continue
        ratio = np.sum(np.abs(st)) / st_nrm
        if ratio > c:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break

    st = _soft_threshold(z, (lo + hi) / 2)
    st_nrm = np.linalg.norm(st)
    if st_nrm < 1e-15:
        return np.zeros_like(z)
    return st / st_nrm

def _sparse_cca_single_from_S(S, c_u, c_v, max_iter=500, tol=1e-6, seed=42):
    """Compute one sparse CCA component from cross-covariance matrix S.

    Parameters
    ----------
    S : ndarray (p, q)
        Cross-covariance matrix X'Y/n (or residual).
    c_u, c_v : float
        L1 constraints.
    max_iter, tol : convergence params.
    seed : random seed.

    Returns
    -------
    u : ndarray (p,), v : ndarray (q,), d : float (u'Sv), n_iter : int
    """
    p, q = S.shape

    rng = np.random.RandomState(seed)
    v = rng.randn(q)
    v = v / np.linalg.norm(v)

    prev_obj = -np.inf

    for it in range(max_iter):
        # u step: maximize u' S v s.t. ||u||_2<=1, ||u||_1<=c_u
        z_u = S @ v
        u = _project_l1_l2(z_u, c_u)

        # v step: maximize u' S v s.t. ||v||_2<=1, ||v||_1<=c_v
        z_v = S.T @ u
        v = _project_l1_l2(z_v, c_v)

        obj = u @ S @ v
        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj

    d = u @ S @ v
    return u, v, d, it + 1

def sparse_cca(X, Y, c_u, c_v, n_components=3, max_iter=500, tol=1e-6):
    """Multi-component sparse CCA with cross-covariance deflation.

    Parameters
    ----------
    X : ndarray (n, p) — first data matrix (will be centered).
    Y : ndarray (n, q) — second data matrix (will be centered).
    c_u : float — L1 constraint for X vectors. Range: [1, sqrt(p)].
    c_v : float — L1 constraint for Y vectors. Range: [1, sqrt(q)].
    n_components : int — number of canonical components.
    max_iter, tol : convergence params.

    Returns
    -------
    U : ndarray (p, n_components) — canonical weight matrix for X.
    V : ndarray (q, n_components) — canonical weight matrix for Y.
    correlations : ndarray (n_components,) — canonical correlations.
    """
    n, p = X.shape
    _, q = Y.shape

    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    # Cross-covariance matrix
    S = X_c.T @ Y_c / n  # (p, q)

    U = np.zeros((p, n_components))
    V = np.zeros((q, n_components))
    correlations = np.zeros(n_components)

    for k in range(n_components):
        u, v, d, n_iter = _sparse_cca_single_from_S(
            S, c_u, c_v, max_iter, tol, seed=42 + k)

        U[:, k] = u
        V[:, k] = v

        # Compute canonical correlation = corr(X_c @ u, Y_c @ v)
        Xu = X_c @ u
        Yv = Y_c @ v
        xu_std = np.std(Xu)
        yv_std = np.std(Yv)
        if xu_std > 1e-15 and yv_std > 1e-15:
            correlations[k] = np.corrcoef(Xu, Yv)[0, 1]
        else:
            correlations[k] = 0.0

        # Cross-covariance deflation: S <- S - d * u * v'
        S = S - d * np.outer(u, v)

    return U, V, correlations
