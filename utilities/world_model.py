#!/usr/bin/env python3
"""
WorldModel: Joint Environment-Genome Embedding for Productivity Prediction.

Architecture:
  - Encoder_E: Environment MLP (env_dim -> 128 -> latent_dim)
  - Encoder_P: PFAM Module MLP (pfam_dim -> 256 -> 128 -> latent_dim)
  - Predictor: Productivity head (latent_dim -> 64 -> 3)

Training:
  Loss = VICReg(z_env, z_pfam) + alpha * MSE(Predictor(z_env), bio_targets)

Inference (environment-only):
  env -> Encoder_E -> z_env -> Predictor -> productivity (chl-a, POC, NFLH)

Designed for:
  - 1,810 ocean samples with 24 environmental variables, 20 PFAM modules, 3 bio targets
  - Spatial block CV (leave-one-basin-out)
  - VICReg non-contrastive alignment (Bardes et al., ICLR 2022)

Author: World Model RALPH Loop
Date: 2026-01-27
"""

import sys
import os
import torch
import torch.nn as nn

# Import VICReg loss from sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vicreg_loss import VICRegLoss

class EncoderE(nn.Module):
    """Environment encoder MLP.

    Architecture: env_dim -> 128 -> latent_dim
    Each layer: Linear -> BatchNorm1d -> ReLU -> Dropout

    Parameters
    ----------
    env_dim : int
        Number of environment input features (default 24).
    latent_dim : int
        Latent embedding dimension (default 16).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(self, env_dim=24, latent_dim=16, dropout=0.3):
        super().__init__()
        self.env_dim = env_dim
        self.latent_dim = latent_dim

        self.layers = nn.Sequential(
            # Block 1: env_dim -> 128
            nn.Linear(env_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2: 128 -> latent_dim
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, env_dim)
            Standardized environment features.

        Returns
        -------
        z_env : torch.Tensor, shape (N, latent_dim)
            Environment embedding.
        """
        return self.layers(x)

class EncoderP(nn.Module):
    """PFAM module encoder MLP.

    Architecture: pfam_dim -> 256 -> 128 -> latent_dim
    Each layer: Linear -> BatchNorm1d -> ReLU -> Dropout
    Deeper than EncoderE because PFAM modules encode richer combinatorial
    information.

    Parameters
    ----------
    pfam_dim : int
        Number of PFAM module input features (default 20).
    latent_dim : int
        Latent embedding dimension (default 16).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(self, pfam_dim=20, latent_dim=16, dropout=0.3):
        super().__init__()
        self.pfam_dim = pfam_dim
        self.latent_dim = latent_dim

        self.layers = nn.Sequential(
            # Block 1: pfam_dim -> 256
            nn.Linear(pfam_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 3: 128 -> latent_dim
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, pfam_dim)
            Standardized PFAM module features.

        Returns
        -------
        z_pfam : torch.Tensor, shape (N, latent_dim)
            PFAM module embedding.
        """
        return self.layers(x)

class Predictor(nn.Module):
    """Productivity prediction head.

    Architecture: input_dim -> 64 -> bio_dim
    Simple head: Linear -> ReLU -> Linear (no BatchNorm/Dropout).

    Parameters
    ----------
    input_dim : int
        Input dimension (latent_dim for env-only, 2*latent_dim for joint).
    bio_dim : int
        Number of bio-response targets (default 3: chl-a, POC, NFLH).
    """

    def __init__(self, input_dim=16, bio_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.bio_dim = bio_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bio_dim),
        )

    def forward(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor, shape (N, input_dim)
            Latent embedding (z_env or [z_env, z_pfam]).

        Returns
        -------
        y_pred : torch.Tensor, shape (N, bio_dim)
            Predicted productivity (chl-a, POC, NFLH).
        """
        return self.layers(z)

class WorldModel(nn.Module):
    """Joint Environment-Genome Embedding Model.

    Wraps Encoder_E, Encoder_P, Predictor, and VICRegLoss into a single
    module for training and inference.

    Training flow:
        env -> Encoder_E -> z_env --|
                                     |--> VICReg(z_env, z_pfam)
        pfam -> Encoder_P -> z_pfam--|
                                     |--> Predictor(z_env) -> y_pred
                                          MSE(y_pred, bio_targets)

    Inference flow (env-only):
        env -> Encoder_E -> z_env -> Predictor -> productivity

    Parameters
    ----------
    env_dim : int
        Number of environment input features (default 24).
    pfam_dim : int
        Number of PFAM module input features (default 20).
    bio_dim : int
        Number of bio-response targets (default 3).
    latent_dim : int
        Latent embedding dimension (default 16).
    dropout : float
        Dropout probability (default 0.3).
    lambda_inv : float
        VICReg invariance weight (default 25.0).
    lambda_var : float
        VICReg variance weight (default 25.0).
    lambda_cov : float
        VICReg covariance weight (default 1.0).
    pred_alpha : float
        Weight for productivity prediction loss (default 1.0).
    """

    def __init__(self, env_dim=24, pfam_dim=20, bio_dim=3, latent_dim=16,
                 dropout=0.3, lambda_inv=25.0, lambda_var=25.0,
                 lambda_cov=1.0, pred_alpha=1.0):
        super().__init__()

        self.env_dim = env_dim
        self.pfam_dim = pfam_dim
        self.bio_dim = bio_dim
        self.latent_dim = latent_dim
        self.pred_alpha = pred_alpha

        # Sub-modules
        self.encoder_e = EncoderE(env_dim, latent_dim, dropout)
        self.encoder_p = EncoderP(pfam_dim, latent_dim, dropout)
        self.predictor = Predictor(latent_dim, bio_dim)
        self.vicreg = VICRegLoss(lambda_inv, lambda_var, lambda_cov)

        # Store config for serialization
        self.config = {
            'env_dim': env_dim,
            'pfam_dim': pfam_dim,
            'bio_dim': bio_dim,
            'latent_dim': latent_dim,
            'dropout': dropout,
            'lambda_inv': lambda_inv,
            'lambda_var': lambda_var,
            'lambda_cov': lambda_cov,
            'pred_alpha': pred_alpha,
        }

    def forward(self, env, pfam, bio_targets=None, bio_valid=None):
        """Full training forward pass.

        Parameters
        ----------
        env : torch.Tensor, shape (N, env_dim)
            Standardized environment features.
        pfam : torch.Tensor, shape (N, pfam_dim)
            Standardized PFAM module features.
        bio_targets : torch.Tensor or None, shape (N, bio_dim)
            Standardized bio-response targets. If None, skip pred loss.
        bio_valid : torch.Tensor or None, shape (N,)
            Boolean mask: True where all bio targets are valid.
            If None and bio_targets given, assume all valid.

        Returns
        -------
        result : dict
            'z_env': (N, latent_dim) environment embedding
            'z_pfam': (N, latent_dim) PFAM module embedding
            'y_pred': (N, bio_dim) predicted productivity
            'total_loss': scalar total loss
            'vicreg_loss': scalar VICReg loss
            'pred_loss': scalar prediction MSE loss (0 if no targets)
            'vicreg_components': dict of individual VICReg terms
        """
        # Encode both modalities
        z_env = self.encoder_e(env)
        z_pfam = self.encoder_p(pfam)

        # Predict productivity from environment embedding
        y_pred = self.predictor(z_env)

        # Compute VICReg alignment loss
        vicreg_loss, vicreg_components = self.vicreg(z_env, z_pfam)

        # Compute prediction loss (only on bio_valid samples)
        pred_loss = torch.tensor(0.0, device=env.device)
        if bio_targets is not None:
            if bio_valid is not None:
                valid_mask = bio_valid.bool()
                if valid_mask.sum() > 0:
                    pred_loss = nn.functional.mse_loss(
                        y_pred[valid_mask], bio_targets[valid_mask]
                    )
            else:
                pred_loss = nn.functional.mse_loss(y_pred, bio_targets)

        # Total loss
        total_loss = vicreg_loss + self.pred_alpha * pred_loss

        return {
            'z_env': z_env,
            'z_pfam': z_pfam,
            'y_pred': y_pred,
            'total_loss': total_loss,
            'vicreg_loss': vicreg_loss,
            'pred_loss': pred_loss,
            'vicreg_components': vicreg_components,
        }

    def encode_env(self, env):
        """Encode environment features to latent space.

        Parameters
        ----------
        env : torch.Tensor, shape (N, env_dim)

        Returns
        -------
        z_env : torch.Tensor, shape (N, latent_dim)
        """
        return self.encoder_e(env)

    def encode_pfam(self, pfam):
        """Encode PFAM module features to latent space.

        Parameters
        ----------
        pfam : torch.Tensor, shape (N, pfam_dim)

        Returns
        -------
        z_pfam : torch.Tensor, shape (N, latent_dim)
        """
        return self.encoder_p(pfam)

    def inference(self, env):
        """Environment-only inference path.

        Parameters
        ----------
        env : torch.Tensor, shape (N, env_dim)
            Standardized environment features.

        Returns
        -------
        y_pred : torch.Tensor, shape (N, bio_dim)
            Predicted productivity.
        """
        z_env = self.encoder_e(env)
        return self.predictor(z_env)

    def count_parameters(self):
        """Count total trainable parameters.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self):
        """Count trainable parameters per sub-module.

        Returns
        -------
        dict
            {'encoder_e': int, 'encoder_p': int, 'predictor': int, 'total': int}
        """
        counts = {}
        for name, module in [('encoder_e', self.encoder_e),
                             ('encoder_p', self.encoder_p),
                             ('predictor', self.predictor)]:
            counts[name] = sum(p.numel() for p in module.parameters()
                               if p.requires_grad)
        counts['total'] = sum(counts.values())
        return counts

def self_test():
    """Run comprehensive self-tests for WorldModel. Returns True if all pass."""
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

    print("=" * 70)
    print("WorldModel Self-Tests")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Test 1: Instantiation with default parameters ──
    print("\nTest 1: Instantiation with default parameters")
    model = WorldModel(env_dim=24, pfam_dim=20, bio_dim=3,
                       latent_dim=16, dropout=0.3).to(device)
    params = model.count_parameters()
    param_detail = model.count_parameters_by_component()
    print(f"  Total parameters: {params:,}")
    print(f"  Encoder_E: {param_detail['encoder_e']:,}")
    print(f"  Encoder_P: {param_detail['encoder_p']:,}")
    print(f"  Predictor: {param_detail['predictor']:,}")
    check("model instantiates", model is not None)
    check("total params > 0", params > 0)
    check("param counts sum correctly",
          param_detail['total'] == params)

    # ── Test 2: Forward pass shapes ──
    print("\nTest 2: Forward pass shapes")
    N = 64
    env = torch.randn(N, 24, device=device)
    pfam = torch.randn(N, 20, device=device)
    bio = torch.randn(N, 3, device=device)
    bio_valid = torch.ones(N, dtype=torch.bool, device=device)

    model.train()
    result = model(env, pfam, bio, bio_valid)
    check("z_env shape", result['z_env'].shape == (N, 16))
    check("z_pfam shape", result['z_pfam'].shape == (N, 16))
    check("y_pred shape", result['y_pred'].shape == (N, 3))
    check("total_loss is scalar", result['total_loss'].dim() == 0)
    check("vicreg_loss is scalar", result['vicreg_loss'].dim() == 0)
    check("pred_loss is scalar", result['pred_loss'].dim() == 0)
    check("vicreg_components present",
          all(k in result['vicreg_components']
              for k in ['invariance', 'variance_a', 'variance_b',
                        'covariance_a', 'covariance_b', 'total']))

    # ── Test 3: Forward without bio targets (VICReg-only mode) ──
    print("\nTest 3: Forward without bio targets (VICReg-only)")
    result_no_bio = model(env, pfam, bio_targets=None)
    check("works without bio targets", result_no_bio['total_loss'].item() > 0)
    check("pred_loss is zero", result_no_bio['pred_loss'].item() == 0.0)

    # ── Test 4: Forward with partial bio_valid mask ──
    print("\nTest 4: Forward with partial bio_valid mask")
    partial_valid = torch.zeros(N, dtype=torch.bool, device=device)
    partial_valid[:32] = True  # Only half valid
    result_partial = model(env, pfam, bio, partial_valid)
    check("works with partial bio_valid", result_partial['total_loss'].item() > 0)
    check("pred_loss computed on valid subset",
          result_partial['pred_loss'].item() >= 0)

    # Forward with all-invalid bio_valid mask
    all_invalid = torch.zeros(N, dtype=torch.bool, device=device)
    result_novalid = model(env, pfam, bio, all_invalid)
    check("works with all-invalid mask",
          result_novalid['pred_loss'].item() == 0.0)

    # ── Test 5: Gradient flow ──
    print("\nTest 5: Gradient flow")
    model.zero_grad()
    result = model(env, pfam, bio, bio_valid)
    result['total_loss'].backward()
    all_params = list(model.named_parameters())
    grad_count = sum(1 for _, p in all_params
                     if p.grad is not None and p.grad.abs().sum() > 0)
    check(f"all {len(all_params)} param tensors receive gradients",
          grad_count == len(all_params))
    no_nan = all(not torch.isnan(p.grad).any()
                 for _, p in all_params if p.grad is not None)
    check("no NaN in any gradient", no_nan)

    # ── Test 6: Inference mode (env-only) ──
    print("\nTest 6: Inference mode (env-only)")
    model.eval()
    with torch.no_grad():
        y_pred_inf = model.inference(env)
    check("inference returns correct shape", y_pred_inf.shape == (N, 3))
    check("no NaN in inference output", not torch.isnan(y_pred_inf).any())

    # ── Test 7: Training convergence (50 steps) ──
    print("\nTest 7: Training convergence (50 steps)")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(50):
        optimizer.zero_grad()
        result = model(env, pfam, bio, bio_valid)
        result['total_loss'].backward()
        optimizer.step()
        losses.append(result['total_loss'].item())
    reduction = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Loss: {losses[0]:.2f} -> {losses[-1]:.2f} ({reduction:.1f}% reduction)")
    check("loss decreases over 50 steps", losses[-1] < losses[0])
    check("no NaN in loss", all(not (l != l) for l in losses))

    # ── Test 8: Different latent dimensions ──
    print("\nTest 8: Different latent dimensions {16, 32, 64}")
    for ld in [16, 32, 64]:
        m = WorldModel(env_dim=24, pfam_dim=20, latent_dim=ld).to(device)
        m.train()
        r = m(env, pfam, bio, bio_valid)
        check(f"latent_dim={ld}: z_env shape ({N},{ld})",
              r['z_env'].shape == (N, ld))
        check(f"latent_dim={ld}: valid loss",
              r['total_loss'].item() > 0 and not torch.isnan(r['total_loss']))

    # ── Test 9: Custom VICReg configs ──
    print("\nTest 9: Custom VICReg configurations")
    configs = {
        'default': dict(lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0),
        'high_variance': dict(lambda_inv=10.0, lambda_var=50.0, lambda_cov=1.0),
        'high_covariance': dict(lambda_inv=25.0, lambda_var=25.0, lambda_cov=10.0),
    }
    for name, cfg in configs.items():
        m = WorldModel(env_dim=24, pfam_dim=20, **cfg).to(device)
        m.train()
        r = m(env, pfam, bio, bio_valid)
        check(f"{name}: valid loss",
              r['total_loss'].item() > 0 and not torch.isnan(r['total_loss']))

    # ── Test 10: Minimum batch size (N=2) ──
    print("\nTest 10: Minimum batch size (N=2)")
    env_small = torch.randn(2, 24, device=device)
    pfam_small = torch.randn(2, 20, device=device)
    bio_small = torch.randn(2, 3, device=device)
    valid_small = torch.ones(2, dtype=torch.bool, device=device)
    model.train()
    r_small = model(env_small, pfam_small, bio_small, valid_small)
    check("batch size 2 works", not torch.isnan(r_small['total_loss']))

    # ── Test 11: Standalone encoder methods ──
    print("\nTest 11: Standalone encoder methods")
    model.eval()
    with torch.no_grad():
        ze = model.encode_env(env)
        zp = model.encode_pfam(pfam)
    check("encode_env shape", ze.shape == (N, 16))
    check("encode_pfam shape", zp.shape == (N, 16))

    # ── Test 12: GPU computation (if available) ──
    print("\nTest 12: GPU computation")
    if torch.cuda.is_available():
        m_gpu = WorldModel(env_dim=24, pfam_dim=20).to('cuda')
        m_gpu.train()
        e_gpu = torch.randn(32, 24, device='cuda')
        p_gpu = torch.randn(32, 20, device='cuda')
        b_gpu = torch.randn(32, 3, device='cuda')
        v_gpu = torch.ones(32, dtype=torch.bool, device='cuda')
        r_gpu = m_gpu(e_gpu, p_gpu, b_gpu, v_gpu)
        r_gpu['total_loss'].backward()
        check("GPU forward + backward succeeded",
              not torch.isnan(r_gpu['total_loss']))
    else:
        print("  SKIP: CUDA not available")
        tests_total += 1
        tests_passed += 1

    # ── Test 13: Model serialization (save/load) ──
    print("\nTest 13: Model serialization (save/load)")
    import tempfile
    model.eval()
    with torch.no_grad():
        y_before = model.inference(env)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
        torch.save(checkpoint, f)

    # Load into fresh model
    loaded = torch.load(tmp_path, map_location=device, weights_only=False)
    model2 = WorldModel(**loaded['config']).to(device)
    model2.load_state_dict(loaded['model_state_dict'])
    model2.eval()
    with torch.no_grad():
        y_after = model2.inference(env)

    max_diff = (y_before - y_after).abs().max().item()
    print(f"  Max prediction diff after save/load: {max_diff:.2e}")
    check("save/load produces identical predictions", max_diff < 1e-6)

    os.unlink(tmp_path)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print(f"{'=' * 70}")

    return tests_passed == tests_total

if __name__ == '__main__':
    success = self_test()
    sys.exit(0 if success else 1)
