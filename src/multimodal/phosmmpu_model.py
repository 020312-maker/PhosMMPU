"""Enhanced multimodal phosphosite model with geometric and typed graph inputs."""

from __future__ import annotations

import torch
from torch import nn

from src.multimodal.torch_gvp import LocalGVPEncoder
from src.multimodal.torch_model import TabularEncoder
from src.multimodal.torch_relation_graph import RelationGraphEncoder


class ContextualSequenceEncoder(nn.Module):
    """Convolutional residue representation pooled globally and around the PTM site."""

    def __init__(self, alphabet_size: int, auxiliary_dim: int, embedding_dim: int):
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.auxiliary_dim = int(auxiliary_dim)
        self.convolution = nn.Sequential(
            nn.Conv1d(alphabet_size, 128, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 4 + auxiliary_dim, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, sequence: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.alphabet_size:
            raise ValueError("sequence has an unexpected shape")
        if auxiliary.shape != (sequence.shape[0], self.auxiliary_dim):
            raise ValueError("sequence auxiliary feature shape does not match the model")
        features = self.convolution(sequence.transpose(1, 2)).transpose(1, 2)
        center = features.shape[1] // 2
        global_max = features.max(dim=1).values
        center_value = features[:, center]
        near_mean = features[:, center - 2 : center + 3].mean(dim=1)
        wide_mean = features[:, center - 3 : center + 4].mean(dim=1)
        return self.projection(
            torch.cat([global_max, center_value, near_mean, wide_mean, auxiliary], dim=-1)
        )


class CrossModalTransformerFusion(nn.Module):
    """Fuse four modality tokens with missing-modality masking and gated residuals."""

    def __init__(self, embedding_dim: int, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        if layers < 0 or layers > 4:
            raise ValueError("cross-modal Transformer must have zero to four layers")
        if embedding_dim % 4:
            raise ValueError("embedding_dim must be divisible by four")
        if layers == 0:
            self.transformer = None
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(
                transformer_layer, num_layers=layers, enable_nested_tensor=False
            )
        self.modality_embedding = nn.Parameter(torch.zeros(1, 4, embedding_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.gate = nn.Linear(embedding_dim, 1)
        self.residual_gate = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor, modality_dropout: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[1] != 4:
            raise ValueError("four modality embeddings are required")
        if mask.shape != embeddings.shape[:2]:
            raise ValueError("modality mask must align with modality embeddings")
        active = mask.float()
        if self.training and modality_dropout > 0:
            keep = (torch.rand_like(active[:, 1:]) >= modality_dropout).float()
            dropped = torch.cat([active[:, :1], active[:, 1:] * keep], dim=1)
            active = torch.where(dropped.sum(dim=1, keepdim=True) > 0, dropped, active)
        if not torch.all(active.sum(dim=1) > 0):
            raise ValueError("at least one modality must remain available")
        tokens = embeddings * active.unsqueeze(-1) + self.modality_embedding
        transformed = (
            tokens
            if self.transformer is None
            else self.transformer(tokens, src_key_padding_mask=~active.bool())
        )
        logits = self.gate(transformed).squeeze(-1).masked_fill(active == 0, -torch.inf)
        gates = torch.softmax(logits, dim=1) * active
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = (transformed * gates.unsqueeze(-1)).sum(dim=1)
        anchor_index = active.argmax(dim=1, keepdim=True)
        anchor = transformed.gather(
            1, anchor_index.unsqueeze(-1).expand(-1, 1, transformed.shape[-1])
        ).squeeze(1)
        residual = torch.sigmoid(self.residual_gate(torch.cat([pooled, anchor], dim=-1)))
        return pooled + residual * anchor, gates


class PhosMMPUModel(nn.Module):
    """Four human-only modalities produce activity, interaction and proteostasis scores."""

    def __init__(
        self,
        alphabet_size: int,
        sequence_aux_dim: int,
        structure_scalar_dim: int,
        network_node_dim: int,
        network_global_dim: int,
        proteomics_dim: int,
        embedding_dim: int = 64,
        transformer_layers: int = 2,
        modality_dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.modality_dropout = float(modality_dropout)
        self.sequence_encoder = ContextualSequenceEncoder(
            alphabet_size, sequence_aux_dim, embedding_dim
        )
        self.structure_encoder = LocalGVPEncoder(
            structure_scalar_dim, vector_dim=1, hidden_dim=96, output_dim=embedding_dim
        )
        self.network_encoder = RelationGraphEncoder(
            network_node_dim,
            network_global_dim,
            relation_count=4,
            hidden_dim=96,
            output_dim=embedding_dim,
            layers=2,
        )
        self.proteomics_encoder = TabularEncoder(proteomics_dim, embedding_dim)
        self.fusion = CrossModalTransformerFusion(embedding_dim, transformer_layers)
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.25),
        )
        self.head = nn.Linear(64, 3)

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_aux: torch.Tensor,
        structure_scalars: torch.Tensor,
        structure_vectors: torch.Tensor,
        structure_mask: torch.Tensor,
        network_node: torch.Tensor,
        network_neighbors: torch.Tensor,
        network_weights: torch.Tensor,
        network_relations: torch.Tensor,
        network_neighbor_mask: torch.Tensor,
        network_global_context: torch.Tensor,
        proteomics: torch.Tensor,
        modality_mask: torch.Tensor,
        return_gates: bool = False,
    ):
        embeddings = torch.stack(
            (
                self.sequence_encoder(sequence, sequence_aux),
                self.structure_encoder(structure_scalars, structure_vectors, structure_mask),
                self.network_encoder(
                    network_node,
                    network_neighbors,
                    network_weights,
                    network_relations,
                    network_neighbor_mask,
                    network_global_context,
                ),
                self.proteomics_encoder(proteomics),
            ),
            dim=1,
        )
        fused, gates = self.fusion(embeddings, modality_mask, self.modality_dropout)
        scores = torch.sigmoid(self.head(self.shared(fused)))
        return (scores, gates) if return_gates else scores

    def dimensions(self) -> dict[str, int | float]:
        return {
            "embedding_dim": self.embedding_dim,
            "sequence_aux_dim": self.sequence_encoder.auxiliary_dim,
            "structure_scalar_dim": self.structure_encoder.scalar_dim,
            "network_node_dim": self.network_encoder.node_dim,
            "network_global_dim": self.network_encoder.global_dim,
            "proteomics_dim": self.proteomics_encoder.layers[0].in_features,
            "modality_dropout": self.modality_dropout,
            "transformer_layers": 0 if self.fusion.transformer is None else len(self.fusion.transformer.layers),
        }
