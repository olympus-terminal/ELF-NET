#!/usr/bin/env python3
"""
manifold_model_20260125_173659.py

PyTorch model definitions for Pfam -> AlphaEarth projection.

Provenance:
    - Task: ralph8_plan.md Task 8
    - Generated: 2026-01-25
    - Purpose: Model architecture for environmental genome manifold projection
    - Architecture spec: source_data/model_architecture_spec.md

Models:
    - PfamToAlphaEarthProjector: Main MLP with bottleneck architecture
    - PfamToAlphaEarthLight: Reduced capacity variant for overfitting control
    - PfamToAlphaEarthPCA: PCA preprocessing + lightweight projection

Data Dimensions:
    - Input: 9,611 CLR-normalized Pfam domain counts
    - Output: 64 AlphaEarth satellite embedding dimensions
    - Training samples: 995 (Train=603, Val=150, Test=242 Mediterranean holdout)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# =============================================================================
# Main Model: Deep MLP with Bottleneck
# =============================================================================

class PfamToAlphaEarthProjector(nn.Module):
    """
    Projects Pfam domain profiles to AlphaEarth satellite embeddings.

    Architecture:
        Input (9,611) -> BatchNorm -> [Linear(2048) -> ReLU -> Dropout(0.3)]
                                   -> [Linear(512)  -> ReLU -> Dropout(0.3)]
                                   -> [Linear(128)  -> ReLU -> Dropout(0.2)]
                                   -> Linear(64) -> Output

    Args:
        input_dim: Number of input Pfam features (default: 9611)
        output_dim: Number of output AlphaEarth dimensions (default: 64)
        hidden_dims: List of hidden layer dimensions (default: [2048, 512, 128])
        dropout_rates: List of dropout rates per hidden layer (default: [0.3, 0.3, 0.2])

    Example:
        >>> model = PfamToAlphaEarthProjector()
        >>> x = torch.randn(32, 9611)  # batch of 32 samples
        >>> y = model(x)  # shape: (32, 64)
    """

    def __init__(
        self,
        input_dim: int = 9611,
        output_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048, 512, 128]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.3, 0.2]

        assert len(hidden_dims) == len(dropout_rates), \
            "hidden_dims and dropout_rates must have same length"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation - AlphaEarth values can be negative)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal for ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = self.input_norm(x)
        return self.network(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embedding(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get intermediate representation from a specific layer.

        Args:
            x: Input tensor
            layer_idx: Index of hidden layer (0-indexed), -1 for final hidden

        Returns:
            Intermediate representation tensor
        """
        x = self.input_norm(x)

        # Each hidden "block" has 3 modules: Linear, ReLU, Dropout
        # Plus final Linear layer
        n_hidden = len(self.hidden_dims)

        if layer_idx == -1:
            layer_idx = n_hidden - 1

        # Go through layers up to requested index
        for i in range(layer_idx + 1):
            block_start = i * 3
            for j in range(3):  # Linear, ReLU, Dropout
                x = self.network[block_start + j](x)

        return x

# =============================================================================
# Light Model: Reduced Capacity Variant
# =============================================================================

class PfamToAlphaEarthLight(nn.Module):
    """
    Lighter version for reduced overfitting risk.

    Architecture:
        Input (9,611) -> BatchNorm -> [Linear(512) -> ReLU -> Dropout(0.3)]
                                   -> [Linear(256) -> ReLU -> Dropout(0.25)]
                                   -> [Linear(128) -> ReLU -> Dropout(0.2)]
                                   -> Linear(64) -> Output

    ~5.2M parameters vs ~20.8M in full model.

    Args:
        input_dim: Number of input Pfam features (default: 9611)
        output_dim: Number of output AlphaEarth dimensions (default: 64)
    """

    def __init__(self, input_dim: int = 9611, output_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = PfamToAlphaEarthProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[512, 256, 128],
            dropout_rates=[0.3, 0.25, 0.2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return self.model.count_parameters()

# =============================================================================
# PCA + Projection Model (For comparison)
# =============================================================================

class PfamToAlphaEarthPCA(nn.Module):
    """
    PCA preprocessing + lightweight MLP projection.

    Applies pretrained PCA transformation before neural projection.
    Useful when high dimensionality causes issues despite regularization.

    Architecture:
        Input (9,611) -> PCA (n_components) -> [Linear(256) -> ReLU -> Dropout]
                                            -> [Linear(128) -> ReLU -> Dropout]
                                            -> Linear(64) -> Output

    Args:
        input_dim: Original input dimension (default: 9611)
        output_dim: Output AlphaEarth dimensions (default: 64)
        pca_components: Number of PCA components (default: 500)
        pca_mean: Pretrained PCA mean vector (fitted on training data)
        pca_components_matrix: Pretrained PCA components (fitted on training data)
    """

    def __init__(
        self,
        input_dim: int = 9611,
        output_dim: int = 64,
        pca_components: int = 500,
        pca_mean: Optional[torch.Tensor] = None,
        pca_components_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pca_components = pca_components

        # PCA transform (frozen, fitted externally)
        # If not provided, will be set later via set_pca_transform()
        if pca_mean is not None:
            self.register_buffer('pca_mean', pca_mean)
        else:
            self.register_buffer('pca_mean', torch.zeros(input_dim))

        if pca_components_matrix is not None:
            self.register_buffer('pca_comp', pca_components_matrix)
        else:
            # Placeholder - must be set before training
            self.register_buffer('pca_comp', torch.zeros(pca_components, input_dim))

        # Post-PCA network
        self.network = nn.Sequential(
            nn.BatchNorm1d(pca_components),
            nn.Linear(pca_components, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_pca_transform(self, pca_mean: torch.Tensor, pca_components: torch.Tensor):
        """
        Set PCA transform from sklearn PCA fitted on training data.

        Args:
            pca_mean: Mean vector from fitted PCA, shape (input_dim,)
            pca_components: Components matrix from fitted PCA, shape (n_components, input_dim)
        """
        self.pca_mean = pca_mean.to(self.pca_mean.device)
        self.pca_comp = pca_components.to(self.pca_comp.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply PCA transform
        x_centered = x - self.pca_mean
        x_pca = torch.matmul(x_centered, self.pca_comp.T)

        # Neural projection
        return self.network(x_pca)

    def count_parameters(self) -> int:
        # Only count trainable parameters (exclude frozen PCA buffers)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# =============================================================================
# Loss Functions
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined MSE + Cosine Similarity loss.

    Loss = alpha * MSE + (1 - alpha) * (1 - cosine_similarity)

    Args:
        alpha: Weight for MSE component (default: 0.8)
    """

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Returns:
            Tuple of (total_loss, mse_loss, cosine_loss)
        """
        mse_loss = self.mse(predictions, targets)
        cos_loss = 1 - self.cos(predictions, targets).mean()
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * cos_loss

        return total_loss, mse_loss, cos_loss

# =============================================================================
# Utilities
# =============================================================================

def get_model(
    model_type: str = 'full',
    input_dim: int = 9611,
    output_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to get model by type.

    Args:
        model_type: One of 'full', 'light', 'pca'
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Model instance
    """
    models = {
        'full': PfamToAlphaEarthProjector,
        'light': PfamToAlphaEarthLight,
        'pca': PfamToAlphaEarthPCA
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](input_dim=input_dim, output_dim=output_dim, **kwargs)

def model_summary(model: nn.Module) -> str:
    """
    Generate a summary of model architecture.

    Returns:
        String summary of model
    """
    lines = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    lines.append(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    lines.append("")
    lines.append("Layers:")

    for name, module in model.named_modules():
        if name:
            if isinstance(module, nn.Linear):
                lines.append(f"  {name}: Linear({module.in_features} -> {module.out_features})")
            elif isinstance(module, nn.BatchNorm1d):
                lines.append(f"  {name}: BatchNorm1d({module.num_features})")
            elif isinstance(module, nn.Dropout):
                lines.append(f"  {name}: Dropout(p={module.p})")
            elif isinstance(module, nn.ReLU):
                lines.append(f"  {name}: ReLU")

    return "\n".join(lines)

# =============================================================================
# Testing
# =============================================================================

def _test_models():
    """Test model instantiation and forward pass."""
    print("Testing model definitions...\n")

    # Test dimensions
    batch_size = 32
    input_dim = 9611
    output_dim = 64

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Test full model
    print("=" * 50)
    print("Testing PfamToAlphaEarthProjector (full)")
    print("=" * 50)
    model_full = PfamToAlphaEarthProjector(input_dim=input_dim, output_dim=output_dim)
    y_full = model_full(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_full.shape}")
    print(f"Parameters:   {model_full.count_parameters():,}")
    assert y_full.shape == (batch_size, output_dim), "Output shape mismatch"
    print("PASSED\n")

    # Test light model
    print("=" * 50)
    print("Testing PfamToAlphaEarthLight")
    print("=" * 50)
    model_light = PfamToAlphaEarthLight(input_dim=input_dim, output_dim=output_dim)
    y_light = model_light(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_light.shape}")
    print(f"Parameters:   {model_light.count_parameters():,}")
    assert y_light.shape == (batch_size, output_dim), "Output shape mismatch"
    print("PASSED\n")

    # Test PCA model
    print("=" * 50)
    print("Testing PfamToAlphaEarthPCA")
    print("=" * 50)
    pca_components = 500
    model_pca = PfamToAlphaEarthPCA(
        input_dim=input_dim,
        output_dim=output_dim,
        pca_components=pca_components,
        pca_mean=torch.zeros(input_dim),
        pca_components_matrix=torch.randn(pca_components, input_dim)
    )
    y_pca = model_pca(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_pca.shape}")
    print(f"Parameters:   {model_pca.count_parameters():,}")
    assert y_pca.shape == (batch_size, output_dim), "Output shape mismatch"
    print("PASSED\n")

    # Test combined loss
    print("=" * 50)
    print("Testing CombinedLoss")
    print("=" * 50)
    loss_fn = CombinedLoss(alpha=0.8)
    target = torch.randn(batch_size, output_dim)
    total_loss, mse_loss, cos_loss = loss_fn(y_full, target)
    print(f"Total loss:  {total_loss.item():.4f}")
    print(f"MSE loss:    {mse_loss.item():.4f}")
    print(f"Cosine loss: {cos_loss.item():.4f}")
    assert total_loss.ndim == 0, "Loss should be scalar"
    print("PASSED\n")

    # Test factory function
    print("=" * 50)
    print("Testing get_model factory")
    print("=" * 50)
    for mtype in ['full', 'light']:
        m = get_model(mtype, input_dim=input_dim, output_dim=output_dim)
        print(f"  {mtype}: {m.__class__.__name__}, params={m.count_parameters():,}")
    print("PASSED\n")

    # Print model summary
    print("=" * 50)
    print("Model Summary (full)")
    print("=" * 50)
    print(model_summary(model_full))

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)

if __name__ == '__main__':
    _test_models()
