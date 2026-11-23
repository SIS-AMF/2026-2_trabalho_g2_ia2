"""Etapa 5 — Arquiteturas de Redes Neurais em PyTorch.

Implementa:
1. `FairMultiTaskRecommender`: MLP Feedforward Multi-Cabeça (3 cabeças:
   `produto_base`, `produto_com_multiplicador` [SKU] e `qtd_pacotes`).
2. `BasketDAE`: Denoising Autoencoder de cestas (comparativo da Aula 10, Slide 42).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FairMultiTaskRecommender(nn.Module):
    """Rede Neural MLP Multi-Task Genérica (3 cabeças) para recomendação de cestas."""

    def __init__(
        self,
        in_dim: int = 99,
        hidden_dims: Sequence[int] = (128, 64),
        n_bases: int = 20,
        n_skus: int = 34,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.n_bases = int(n_bases)
        self.n_skus = int(n_skus)
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        prev_dim = self.in_dim
        for idx, h_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            # Dropout ligeiramente mais suave na última camada oculta
            drop_rate = self.dropout if idx == 0 else max(0.10, self.dropout * 0.70)
            layers.append(nn.Dropout(p=drop_rate))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head_base = nn.Linear(prev_dim, self.n_bases)
        self.head_sku = nn.Linear(prev_dim, self.n_skus)
        self.head_qty = nn.Sequential(
            nn.Linear(prev_dim, self.n_skus),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (logits_base [B, 20], logits_sku [B, 34], pred_qty_log1p [B, 34])."""
        h = self.backbone(x)
        logits_base = self.head_base(h)
        logits_sku = self.head_sku(h)
        pred_qty = self.head_qty(h)
        return logits_base, logits_sku, pred_qty

    def get_config(self) -> dict[str, object]:
        return {
            "in_dim": self.in_dim,
            "hidden_dims": list(self.hidden_dims),
            "n_bases": self.n_bases,
            "n_skus": self.n_skus,
            "dropout": self.dropout,
        }


class BasketDAE(nn.Module):
    """Denoising Autoencoder de Cestas (Slide 42 — Aula 10).

    Recebe `[x_cart_sku_bin (34) || x_time (6)]`, aplica ruído de mascaramento
    durante o treino para corromper itens da cesta, comprime no bottleneck
    latente (24 neurônios) e reconstrói probabilidades para os 34 SKUs.
    """

    def __init__(
        self,
        n_skus: int = 34,
        n_time: int = 6,
        hidden_dim: int = 64,
        bottleneck_dim: int = 24,
        dropout: float = 0.25,
        masking_noise: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_skus = int(n_skus)
        self.n_time = int(n_time)
        self.masking_noise = float(masking_noise)

        in_total = self.n_skus + self.n_time
        self.encoder = nn.Sequential(
            nn.Linear(in_total, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=max(0.10, dropout * 0.80)),
            nn.Linear(hidden_dim, self.n_skus),
        )

    def forward(self, x_sku_bin: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        """Retorna logits reconstruídos dos 34 SKUs."""
        if self.training and self.masking_noise > 0.0:
            keep_prob = 1.0 - self.masking_noise
            noise_mask = (torch.rand_like(x_sku_bin) < keep_prob).float()
            x_corrupted = x_sku_bin * noise_mask
        else:
            x_corrupted = x_sku_bin
        z = self.encoder(torch.cat([x_corrupted, x_time], dim=1))
        return self.decoder(z)
