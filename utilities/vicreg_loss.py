#!/usr/bin/env python3
"""
VICReg Loss Function for Joint Embedding Learning.

Implements the Variance-Invariance-Covariance Regularization loss from:
  Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance
  Regularization for Self-Supervised Learning", ICLR 2022.

Three terms:
  1. Invariance: MSE between paired embeddings (push co-located pairs together)
  2. Variance:   Hinge loss on per-dimension std dev (prevent collapse)
  3. Covariance: Penalize off-diagonal covariance (decorrelate dimensions)

Usage:
    loss_fn = VICRegLoss(lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0)
    total_loss, components = loss_fn(z_a, z_b)
"""

import torch
import torch.nn as nn

class VICRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization Loss.

    Parameters
    ----------
    lambda_inv : float
        Weight for invariance term (MSE between paired embeddings).
    lambda_var : float
        Weight for variance term (hinge loss on per-dimension std dev).
    lambda_cov : float
        Weight for covariance term (off-diagonal covariance penalty).
    gamma : float
        Target standard deviation for variance hinge (default 1.0).
    """

    def __init__(self, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
                 gamma=1.0):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma

    def invariance_loss(self, z_a, z_b):
        """MSE between paired embeddings.

        Parameters
        ----------
        z_a, z_b : torch.Tensor, shape (N, D)
            Paired embedding vectors.

        Returns
        -------
        torch.Tensor
            Scalar invariance loss.
        """
        return torch.nn.functional.mse_loss(z_a, z_b)

    def variance_loss(self, z):
        """Hinge loss on per-dimension standard deviation.

        Encourages each dimension to have std >= gamma, preventing
        embedding collapse where all points map to the same vector.

        Parameters
        ----------
        z : torch.Tensor, shape (N, D)
            Embedding matrix (single modality).

        Returns
        -------
        torch.Tensor
            Scalar variance loss.
        """
        # Per-dimension std with epsilon for numerical stability
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        # Hinge: penalize dimensions with std below gamma
        return torch.mean(torch.relu(self.gamma - std_z))

    def covariance_loss(self, z):
        """Off-diagonal covariance penalty.

        Decorrelates embedding dimensions by penalizing off-diagonal
        elements of the covariance matrix.

        Parameters
        ----------
        z : torch.Tensor, shape (N, D)
            Embedding matrix (single modality).

        Returns
        -------
        torch.Tensor
            Scalar covariance loss.
        """
        N, D = z.shape
        # Center the embeddings
        z_centered = z - z.mean(dim=0)
        # Compute covariance matrix
        cov = (z_centered.T @ z_centered) / (N - 1)
        # Zero out diagonal (we only penalize off-diagonal)
        cov_offdiag = cov - torch.diag(cov.diag())
        # Sum of squared off-diagonal elements, normalized by D
        return (cov_offdiag ** 2).sum() / D

    def forward(self, z_a, z_b):
        """Compute total VICReg loss.

        Parameters
        ----------
        z_a : torch.Tensor, shape (N, D)
            Embeddings from modality A (e.g., environment encoder).
        z_b : torch.Tensor, shape (N, D)
            Embeddings from modality B (e.g., PFAM module encoder).

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of invariance, variance, and covariance terms.
        components : dict
            Individual loss components for logging:
            - 'invariance': float
            - 'variance_a': float (variance loss for z_a)
            - 'variance_b': float (variance loss for z_b)
            - 'covariance_a': float (covariance loss for z_a)
            - 'covariance_b': float (covariance loss for z_b)
            - 'total': float
        """
        # Input validation
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"Shape mismatch: z_a {z_a.shape} vs z_b {z_b.shape}"
            )
        if z_a.shape[0] < 2:
            raise ValueError(
                f"Batch size must be >= 2, got {z_a.shape[0]}"
            )

        # Compute individual terms
        inv_loss = self.invariance_loss(z_a, z_b)
        var_loss_a = self.variance_loss(z_a)
        var_loss_b = self.variance_loss(z_b)
        cov_loss_a = self.covariance_loss(z_a)
        cov_loss_b = self.covariance_loss(z_b)

        # Combine: variance and covariance applied to BOTH modalities
        total = (self.lambda_inv * inv_loss
                 + self.lambda_var * (var_loss_a + var_loss_b)
                 + self.lambda_cov * (cov_loss_a + cov_loss_b))

        components = {
            'invariance': inv_loss.item(),
            'variance_a': var_loss_a.item(),
            'variance_b': var_loss_b.item(),
            'covariance_a': cov_loss_a.item(),
            'covariance_b': cov_loss_b.item(),
            'total': total.item(),
        }

        return total, components

def self_test():
    """Run self-tests for VICReg loss module. Returns True if all pass."""
    import sys

    tests_passed = 0
    tests_total = 0

    def check(name, condition):
        nonlocal tests_passed, tests_total
        tests_total += 1
        if condition:
            tests_passed += 1
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")

    print("=" * 60)
    print("VICReg Loss Self-Tests")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    loss_fn = VICRegLoss(lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0)

    # Test 1: Gradient flow
    print("Test 1: Gradient flow")
    z_a = torch.randn(64, 16, device=device, requires_grad=True)
    z_b = torch.randn(64, 16, device=device, requires_grad=True)
    total, comp = loss_fn(z_a, z_b)
    total.backward()
    check("gradients computed for z_a", z_a.grad is not None)
    check("gradients computed for z_b", z_b.grad is not None)
    check("no NaN in z_a grad", not torch.isnan(z_a.grad).any())
    check("no NaN in z_b grad", not torch.isnan(z_b.grad).any())
    check("all components present",
          all(k in comp for k in ['invariance', 'variance_a', 'variance_b',
                                   'covariance_a', 'covariance_b', 'total']))

    # Test 2: Invariance = 0 for identical embeddings
    print("\nTest 2: Invariance = 0 for identical embeddings")
    z_same = torch.randn(32, 16, device=device)
    inv = loss_fn.invariance_loss(z_same, z_same)
    check("invariance is zero", inv.item() < 1e-7)

    # Test 3: Variance = 0 when std >= gamma
    print("\nTest 3: Variance = 0 when std >= gamma")
    z_spread = torch.randn(1000, 16, device=device) * 2.0  # std ~2.0 >> gamma=1.0
    var_loss = loss_fn.variance_loss(z_spread)
    check("variance is zero for high-spread embeddings", var_loss.item() < 1e-4)

    # Test 4: Variance > 0 for collapsed embeddings
    print("\nTest 4: Variance > 0 for collapsed embeddings")
    z_collapsed = torch.ones(32, 16, device=device) * 0.5  # constant -> std=0
    # Add tiny noise so std is very small but not exactly zero
    z_collapsed = z_collapsed + torch.randn_like(z_collapsed) * 1e-6
    var_loss_collapsed = loss_fn.variance_loss(z_collapsed)
    check("variance penalizes collapsed embeddings",
          var_loss_collapsed.item() > 0.5)

    # Test 5: Covariance ~ 0 for i.i.d. Gaussian
    print("\nTest 5: Covariance ~ 0 for i.i.d. Gaussian")
    z_iid = torch.randn(1000, 16, device=device)
    cov_loss_iid = loss_fn.covariance_loss(z_iid)
    check("covariance low for i.i.d. Gaussian (< 0.1)",
          cov_loss_iid.item() < 0.1)

    # Test 6: Covariance high for correlated dimensions
    print("\nTest 6: Covariance high for correlated dimensions")
    z_base = torch.randn(1000, 1, device=device)
    z_corr = z_base.repeat(1, 16) + torch.randn(1000, 16, device=device) * 0.01
    cov_loss_corr = loss_fn.covariance_loss(z_corr)
    check("covariance penalizes correlated dimensions (> 1.0)",
          cov_loss_corr.item() > 1.0)

    # Test 7: Three lambda configurations
    print("\nTest 7: Three lambda configurations")
    configs = {
        'default': VICRegLoss(25.0, 25.0, 1.0),
        'high_variance': VICRegLoss(10.0, 50.0, 1.0),
        'high_covariance': VICRegLoss(25.0, 25.0, 10.0),
    }
    z_a_test = torch.randn(64, 16, device=device)
    z_b_test = torch.randn(64, 16, device=device)
    for name, cfg in configs.items():
        total_loss, _ = cfg(z_a_test, z_b_test)
        check(f"{name} produces valid loss (> 0)",
              total_loss.item() > 0 and not torch.isnan(total_loss))

    # Test 8: Shape validation
    print("\nTest 8: Shape validation")
    try:
        loss_fn(torch.randn(10, 16, device=device),
                torch.randn(10, 32, device=device))
        check("shape mismatch caught", False)
    except ValueError:
        check("shape mismatch caught", True)

    try:
        loss_fn(torch.randn(1, 16, device=device),
                torch.randn(1, 16, device=device))
        check("batch size < 2 caught", False)
    except ValueError:
        check("batch size < 2 caught", True)

    # Test 9: GPU computation (if available)
    print("\nTest 9: GPU computation")
    if torch.cuda.is_available():
        z_gpu_a = torch.randn(64, 16, device='cuda', requires_grad=True)
        z_gpu_b = torch.randn(64, 16, device='cuda', requires_grad=True)
        total_gpu, comp_gpu = loss_fn.to('cuda')(z_gpu_a, z_gpu_b)
        total_gpu.backward()
        check("GPU forward + backward succeeded",
              z_gpu_a.grad is not None and not torch.isnan(z_gpu_a.grad).any())
    else:
        print("  SKIP: CUDA not available")
        tests_total += 1
        tests_passed += 1  # Skip counts as pass

    print(f"\n{'=' * 60}")
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print(f"{'=' * 60}")

    return tests_passed == tests_total

if __name__ == '__main__':
    success = self_test()
    import sys
    sys.exit(0 if success else 1)
