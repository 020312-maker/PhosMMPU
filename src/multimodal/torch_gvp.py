"""Rotation-invariant local geometric encoder without a torch-geometric dependency."""

from __future__ import annotations

import torch
from torch import nn


class GVPBlock(nn.Module):
    """A compact geometric vector perceptron operating on scalar/vector features."""

    def __init__(self, scalar_in: int, scalar_out: int, vector_dim: int):
        super().__init__()
        self.vector_projection = nn.Linear(vector_dim, vector_dim, bias=False)
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_in + vector_dim, scalar_out),
            nn.SiLU(),
            nn.Linear(scalar_out, scalar_out),
            nn.SiLU(),
        )
        self.vector_gate = nn.Linear(scalar_out, vector_dim)

    def forward(self, scalar: torch.Tensor, vectors: torch.Tensor):
        if vectors.ndim != scalar.ndim + 1 or vectors.shape[-1] != 3:
            raise ValueError("vectors must have one vector-channel dimension and final xyz dimension")
        projected = self.vector_projection(vectors.transpose(-1, -2)).transpose(-1, -2)
        vector_norm = torch.linalg.vector_norm(projected, dim=-1)
        scalar_out = self.scalar_projection(torch.cat([scalar, vector_norm], dim=-1))
        gate = torch.sigmoid(self.vector_gate(scalar_out)).unsqueeze(-1)
        return scalar_out, projected * gate


class LocalGVPEncoder(nn.Module):
    """Encode a fixed AlphaFold residue neighborhood into a site-level embedding."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int = 1,
        hidden_dim: int = 96,
        output_dim: int = 64,
    ):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.first = GVPBlock(scalar_dim, hidden_dim, vector_dim)
        self.second = GVPBlock(hidden_dim, hidden_dim, vector_dim)
        self.center_message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self, scalar: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if scalar.ndim != 3:
            raise ValueError("scalar local graphs must have shape (batch, nodes, features)")
        if scalar.shape[-1] != self.scalar_dim:
            raise ValueError("unexpected local graph scalar feature dimension")
        if vectors.ndim == 3:
            vectors = vectors.unsqueeze(-2)
        if vectors.shape[:-2] != scalar.shape[:-1] or vectors.shape[-2:] != (self.vector_dim, 3):
            raise ValueError("unexpected local graph vector feature dimensions")
        if mask.shape != scalar.shape[:2]:
            raise ValueError("local graph mask must have shape (batch, nodes)")
        center_indicator = scalar[:, :, -1:].clamp(0.0, 1.0)
        scalar, vectors = self.first(scalar, vectors)
        scalar, _ = self.second(scalar, vectors)
        weights = mask.float().unsqueeze(-1)
        center_weights = weights * center_indicator
        center_state = (scalar * center_weights).sum(dim=1) / center_weights.sum(dim=1).clamp_min(1.0)
        message = self.center_message(center_state).unsqueeze(1).expand_as(scalar)
        scalar = scalar + self.node_update(torch.cat([scalar, message], dim=-1))
        pooled = (scalar * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)
