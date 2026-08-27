"""Relation-aware fixed-neighbour graph attention for protein interactions."""

from __future__ import annotations

import math

import torch
from torch import nn


class RelationAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_key = nn.Embedding(relation_count, hidden_dim)
        self.relation_value = nn.Embedding(relation_count, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.first_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.second_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_state: torch.Tensor,
        neighbor_state: torch.Tensor,
        edge_weight: torch.Tensor,
        relation: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(node_state).unsqueeze(1)
        key = self.key(neighbor_state) + self.relation_key(relation)
        value = self.value(neighbor_state) + self.relation_value(relation)
        logits = (query * key).sum(dim=-1) / math.sqrt(node_state.shape[-1])
        logits = logits + torch.log(edge_weight.clamp_min(1e-8))
        valid = neighbor_mask.bool()
        logits = logits.masked_fill(~valid, -1e9)
        attention = torch.softmax(logits, dim=1) * valid.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
        aggregate = (attention.unsqueeze(-1) * value).sum(dim=1)
        state = self.first_norm(node_state + self.dropout(self.output(aggregate)))
        return self.second_norm(state + self.dropout(self.feed_forward(state)))


class RelationGraphEncoder(nn.Module):
    """Encode per-protein graph evidence sampled by ``relation_graph.py``."""

    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        relation_count: int,
        hidden_dim: int = 96,
        output_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("relation graph encoder requires at least one layer")
        self.node_dim = int(node_dim)
        self.global_dim = int(global_dim)
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.global_projection = nn.Linear(global_dim, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            RelationAttentionLayer(hidden_dim, relation_count, dropout) for _ in range(layers)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        node: torch.Tensor,
        neighbors: torch.Tensor,
        edge_weight: torch.Tensor,
        relation: torch.Tensor,
        neighbor_mask: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        if node.ndim != 2 or node.shape[-1] != self.node_dim:
            raise ValueError("node features have an unexpected shape")
        if neighbors.ndim != 3 or neighbors.shape[:1] != node.shape[:1] or neighbors.shape[-1] != self.node_dim:
            raise ValueError("neighbor node features have an unexpected shape")
        if edge_weight.shape != neighbors.shape[:2] or neighbor_mask.shape != neighbors.shape[:2]:
            raise ValueError("relation edge tensors must match neighbour dimensions")
        if relation.shape != neighbors.shape[:2]:
            raise ValueError("relation type tensor must match neighbour dimensions")
        if global_context.shape != (node.shape[0], self.global_dim):
            raise ValueError("global graph context has an unexpected shape")
        if relation.numel() and (relation.min() < 0 or relation.max() >= self.layers[0].relation_key.num_embeddings):
            raise ValueError("relation types are outside the configured range")

        state = self.input_norm(self.node_projection(node) + self.global_projection(global_context))
        neighbor_state = self.node_projection(neighbors)
        for layer in self.layers:
            state = layer(state, neighbor_state, edge_weight, relation.long(), neighbor_mask)
        return self.output(state)
