#!/usr/bin/env python3
"""
kan_layer.py — Minimal Kolmogorov-Arnold Network layer in PyTorch.

Each edge has a learnable B-spline activation function instead of a fixed
activation + linear weight. This provides interpretable nonlinear mappings.

Architecture:
  For input dim d_in and output dim d_out:
  - d_in * d_out B-spline functions, one per edge
  - Each B-spline has (grid_size + spline_order) learnable coefficients
  - Output j = sum_i spline_{i,j}(x_i)

B-spline parameters:
  - grid_size: number of intervals (default 5)
  - spline_order: B-spline order (default 3 = cubic)
  - grid range: [-1, 1] (data should be standardized)
"""

import torch
import torch.nn as nn
import numpy as np

def _bspline_basis(x, grid, order):
    """Compute B-spline basis values using Cox-de Boor recursion.

    Parameters
    ----------
    x : Tensor (batch, d_in)
    grid : Tensor (n_knots,) — knot positions
    order : int — spline order (0=constant, 1=linear, 3=cubic)

    Returns
    -------
    basis : Tensor (batch, d_in, n_basis) where n_basis = len(grid) - order - 1
    """
    # Expand x: (batch, d_in, 1)
    x = x.unsqueeze(-1)
    # grid: (n_knots,)
    grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n_knots)

    n_knots = grid.shape[-1]

    # Order 0: indicator functions
    # basis_k(x) = 1 if grid[k] <= x < grid[k+1]
    bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
    # (batch, d_in, n_knots-1)

    # Handle rightmost point: include it in the last basis function
    bases[:, :, -1] += (x[:, :, 0] == grid[0, 0, -1]).float()

    for p in range(1, order + 1):
        # Recursion: B_{i,p}(x) = w1 * B_{i,p-1} + w2 * B_{i+1,p-1}
        knots_left = grid[:, :, :-(p + 1)]   # grid[i]
        knots_mid = grid[:, :, p:-1]          # grid[i+p]
        knots_right = grid[:, :, (p + 1):]    # grid[i+p+1]
        knots_next = grid[:, :, 1:-(p)]       # grid[i+1]

        denom1 = knots_mid - knots_left
        denom2 = knots_right - knots_next

        # Avoid division by zero
        safe1 = denom1.abs() > 1e-10
        safe2 = denom2.abs() > 1e-10

        w1 = torch.where(safe1, (x - knots_left) / denom1.clamp(min=1e-10), torch.zeros_like(denom1))
        w2 = torch.where(safe2, (knots_right - x) / denom2.clamp(min=1e-10), torch.zeros_like(denom2))

        bases = w1 * bases[:, :, :-1] + w2 * bases[:, :, 1:]

    return bases  # (batch, d_in, n_basis)

class KANLayer(nn.Module):
    """Single KAN layer: learnable B-spline activations per edge.

    Parameters
    ----------
    d_in : int — input dimension
    d_out : int — output dimension
    grid_size : int — number of grid intervals (default 5)
    spline_order : int — B-spline order (default 3)
    grid_range : tuple — range for uniform grid (default (-1, 1))
    """

    def __init__(self, d_in, d_out, grid_size=5, spline_order=3,
                 grid_range=(-1.0, 1.0)):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.grid_size = grid_size
        self.spline_order = spline_order

        n_knots = grid_size + 2 * spline_order + 1
        n_basis = grid_size + spline_order

        # Uniform knot vector with extra knots for boundary
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            n_knots
        )
        self.register_buffer('grid', grid)

        # Learnable spline coefficients: (d_in, d_out, n_basis)
        self.coeffs = nn.Parameter(
            torch.randn(d_in, d_out, n_basis) * 0.1
        )

        # Residual linear connection (SiLU activation as in original KAN)
        self.base_weight = nn.Parameter(torch.randn(d_in, d_out) * 0.1)

    def forward(self, x):
        """
        x : Tensor (batch, d_in)
        Returns: Tensor (batch, d_out)
        """
        batch = x.shape[0]

        # Compute B-spline basis: (batch, d_in, n_basis)
        basis = _bspline_basis(x, self.grid, self.spline_order)

        # Spline output: sum over basis functions
        # basis: (batch, d_in, n_basis)
        # coeffs: (d_in, d_out, n_basis)
        # Result: (batch, d_in, d_out)
        spline_out = torch.einsum('bin,ion->bio', basis, self.coeffs)

        # Sum over input dimension: (batch, d_out)
        out = spline_out.sum(dim=1)

        # Add residual linear connection: x @ base_weight
        # Using SiLU activation on x as in original KAN paper
        out = out + torch.nn.functional.silu(x) @ self.base_weight

        return out

    def spline_regularization(self):
        """L1 + smoothness regularization on spline coefficients.

        Returns (l1_reg, smooth_reg) as separate terms.
        """
        l1 = self.coeffs.abs().mean()

        # Smoothness: penalize second differences of coefficients
        if self.coeffs.shape[-1] >= 3:
            diff2 = self.coeffs[:, :, 2:] - 2 * self.coeffs[:, :, 1:-1] + self.coeffs[:, :, :-2]
            smooth = (diff2 ** 2).mean()
        else:
            smooth = torch.tensor(0.0, device=self.coeffs.device)

        return l1, smooth

    def get_spline_activations(self, x_range=(-1, 1), n_points=100):
        """Evaluate learned spline activations on a grid for visualization.

        Returns dict mapping (i, j) -> (x_vals, y_vals) for each edge.
        """
        x_vals = torch.linspace(x_range[0], x_range[1], n_points)
        # Evaluate each input dimension independently
        activations = {}
        for i in range(self.d_in):
            x_single = torch.zeros(n_points, self.d_in)
            x_single[:, i] = x_vals
            with torch.no_grad():
                basis = _bspline_basis(x_single, self.grid, self.spline_order)
                # basis: (n_points, d_in, n_basis)
                # Only care about dimension i
                basis_i = basis[:, i, :]  # (n_points, n_basis)
                for j in range(self.d_out):
                    y_spline = (basis_i * self.coeffs[i, j, :]).sum(dim=-1)
                    y_linear = torch.nn.functional.silu(x_vals) * self.base_weight[i, j]
                    y_total = y_spline + y_linear
                    activations[(i, j)] = (x_vals.numpy(), y_total.numpy())

        return activations
