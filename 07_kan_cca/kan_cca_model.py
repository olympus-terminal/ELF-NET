#!/usr/bin/env python3
"""
kan_cca_model.py — Two-branch KAN-CCA model.

Architecture:
  Domain branch:  KAN(k_d -> hidden -> n_components)
  Env branch:     KAN(k_e -> hidden -> n_components)

Loss: negative correlation between latent projections + regularization.
  L = -sum_k corr(z_d_k, z_e_k) + lambda_l1 * L1_reg + lambda_smooth * smooth_reg

The model learns nonlinear canonical variates via B-spline activations,
providing interpretable mappings from features to latent space.
"""

import torch
import torch.nn as nn
import numpy as np

from kan_layer import KANLayer

class KANCCAModel(nn.Module):
    """Two-branch KAN-CCA model.

    Parameters
    ----------
    d_domain : int — number of domain features
    d_env : int — number of environment features
    n_components : int — number of canonical components
    hidden_dim : int — hidden layer size (default 32)
    grid_size : int — B-spline grid intervals (default 5)
    spline_order : int — B-spline order (default 3)
    """

    def __init__(self, d_domain, d_env, n_components=3, hidden_dim=32,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.n_components = n_components

        # Domain branch: KAN(d_domain -> hidden -> n_components)
        self.domain_kan1 = KANLayer(d_domain, hidden_dim, grid_size, spline_order)
        self.domain_kan2 = KANLayer(hidden_dim, n_components, grid_size, spline_order)

        # Env branch: KAN(d_env -> hidden -> n_components)
        self.env_kan1 = KANLayer(d_env, hidden_dim, grid_size, spline_order)
        self.env_kan2 = KANLayer(hidden_dim, n_components, grid_size, spline_order)

    def forward(self, x_domain, x_env):
        """
        Returns z_domain, z_env: (batch, n_components) each.
        """
        z_d = self.domain_kan2(self.domain_kan1(x_domain))
        z_e = self.env_kan2(self.env_kan1(x_env))
        return z_d, z_e

    def correlation_loss(self, z_d, z_e):
        """Negative sum of per-component correlations.

        corr_k = cov(z_d_k, z_e_k) / (std(z_d_k) * std(z_e_k))
        loss = -sum_k corr_k
        """
        n = z_d.shape[0]
        z_d_c = z_d - z_d.mean(dim=0, keepdim=True)
        z_e_c = z_e - z_e.mean(dim=0, keepdim=True)

        # Per-component covariance
        cov = (z_d_c * z_e_c).sum(dim=0) / (n - 1)
        std_d = z_d_c.std(dim=0).clamp(min=1e-8)
        std_e = z_e_c.std(dim=0).clamp(min=1e-8)

        corrs = cov / (std_d * std_e)

        return -corrs.sum(), corrs.detach()

    def regularization_loss(self):
        """Sum of L1 and smoothness regularization across all KAN layers."""
        total_l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        total_smooth = torch.tensor(0.0, device=next(self.parameters()).device)

        for layer in [self.domain_kan1, self.domain_kan2,
                      self.env_kan1, self.env_kan2]:
            l1, smooth = layer.spline_regularization()
            total_l1 = total_l1 + l1
            total_smooth = total_smooth + smooth

        return total_l1, total_smooth

    def extract_first_layer_splines(self, x_range=(-1, 1), n_points=100):
        """Extract spline activations from the first KAN layer of each branch.

        Returns dict with 'domain' and 'env' keys, each mapping
        (input_idx, output_idx) -> (x_vals, y_vals).
        """
        return {
            'domain': self.domain_kan1.get_spline_activations(x_range, n_points),
            'env': self.env_kan1.get_spline_activations(x_range, n_points),
        }

def train_kan_cca(X_domain, X_env, n_components=3, hidden_dim=32,
                  n_epochs=200, lr=1e-3, weight_decay=1e-4,
                  lambda_l1=1e-3, lambda_smooth=1e-3,
                  patience=20, device='cpu'):
    """Train KAN-CCA model.

    Parameters
    ----------
    X_domain : ndarray (n, k_d) — domain features (standardized)
    X_env : ndarray (n, k_e) — env features (standardized)
    n_components : int
    hidden_dim : int
    n_epochs : int
    lr : float
    weight_decay : float
    lambda_l1 : float — L1 regularization weight
    lambda_smooth : float — smoothness regularization weight
    patience : int — early stopping patience
    device : str

    Returns
    -------
    model : trained KANCCAModel
    history : list of dicts with per-epoch metrics
    """
    n, k_d = X_domain.shape
    _, k_e = X_env.shape

    model = KANCCAModel(k_d, k_e, n_components, hidden_dim).to(device)

    X_d_t = torch.tensor(X_domain, dtype=torch.float32, device=device)
    X_e_t = torch.tensor(X_env, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        z_d, z_e = model(X_d_t, X_e_t)
        corr_loss, corrs = model.correlation_loss(z_d, z_e)
        l1_reg, smooth_reg = model.regularization_loss()

        loss = corr_loss + lambda_l1 * l1_reg + lambda_smooth * smooth_reg
        loss.backward()
        optimizer.step()

        epoch_info = {
            'epoch': epoch,
            'loss': loss.item(),
            'corr_loss': corr_loss.item(),
            'l1_reg': l1_reg.item(),
            'smooth_reg': smooth_reg.item(),
            'correlations': corrs.cpu().numpy().tolist(),
        }
        history.append(epoch_info)

        # Early stopping on total loss
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, history

def evaluate_kan_cca(model, X_domain, X_env, device='cpu'):
    """Evaluate a trained KAN-CCA model, returning per-component correlations."""
    X_d_t = torch.tensor(X_domain, dtype=torch.float32, device=device)
    X_e_t = torch.tensor(X_env, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        z_d, z_e = model(X_d_t, X_e_t)

    z_d_np = z_d.cpu().numpy()
    z_e_np = z_e.cpu().numpy()

    corrs = []
    for k in range(z_d_np.shape[1]):
        if np.std(z_d_np[:, k]) > 1e-10 and np.std(z_e_np[:, k]) > 1e-10:
            c = np.corrcoef(z_d_np[:, k], z_e_np[:, k])[0, 1]
        else:
            c = 0.0
        corrs.append(c)

    return np.array(corrs)
