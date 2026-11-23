"""Etapa 5 — Função de Erro Composta Mascarada (MultiTaskRecommendationLoss)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def calcular_pos_weights(
    y_train: np.ndarray,
    max_weight: float = 6.0,
) -> torch.Tensor:
    """Calcula `pos_weight` suavizado (sqrt(N_neg / N_pos)) com teto para estabilidade."""
    n_samples = float(len(y_train))
    pos = np.maximum(y_train.sum(axis=0), 1.0)
    neg = np.maximum(n_samples - pos, 1.0)
    weights = np.sqrt(neg / pos)
    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


class MultiTaskRecommendationLoss(nn.Module):
    """Loss composta mascarada para as 3 cabeças do `FairMultiTaskRecommender`.

    L_total = w_base * L_BCE_masked(base) + w_sku * L_BCE_masked(sku) + w_qty * L_SmoothL1_pos(qty)
    """

    def __init__(
        self,
        w_base: float = 0.8,
        w_sku: float = 1.5,
        w_qty: float = 0.4,
        pos_weight_base: torch.Tensor | None = None,
        pos_weight_sku: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.w_base = float(w_base)
        self.w_sku = float(w_sku)
        self.w_qty = float(w_qty)

        self.register_buffer(
            "pos_weight_base",
            pos_weight_base if pos_weight_base is not None else torch.ones(1),
        )
        self.register_buffer(
            "pos_weight_sku",
            pos_weight_sku if pos_weight_sku is not None else torch.ones(1),
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        logits_base: torch.Tensor,
        logits_sku: torch.Tensor,
        pred_qty: torch.Tensor,
        y_base: torch.Tensor,
        y_sku: torch.Tensor,
        y_qty: torch.Tensor,
        mask_base: torch.Tensor,
        mask_sku: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. BCE mascarada para produto_base (mantém ativos alvos positivos mesmo se
        # o cliente comprou outra embalagem do mesmo produto_base)
        pw_base = self.pos_weight_base if self.pos_weight_base.numel() == logits_base.shape[1] else None
        bce_base_raw = nn.functional.binary_cross_entropy_with_logits(
            logits_base, y_base, pos_weight=pw_base, reduction="none"
        )
        valid_base = torch.clamp((1.0 - mask_base) + y_base, max=1.0)
        loss_base = (bce_base_raw * valid_base).sum() / valid_base.sum().clamp(min=1.0)

        # 2. BCE mascarada para SKU (exclui SKUs já presentes no carrinho)
        pw_sku = self.pos_weight_sku if self.pos_weight_sku.numel() == logits_sku.shape[1] else None
        bce_sku_raw = nn.functional.binary_cross_entropy_with_logits(
            logits_sku, y_sku, pos_weight=pw_sku, reduction="none"
        )
        valid_sku = torch.clamp(1.0 - mask_sku, min=0.0)
        loss_sku = (bce_sku_raw * valid_sku).sum() / valid_sku.sum().clamp(min=1.0)

        # 3. SmoothL1 (Huber) exclusivamente nos índices positivos (y_sku == 1)
        qty_err_raw = self.smooth_l1(pred_qty, y_qty)
        pos_mask = (y_sku > 0.5).float()
        loss_qty = (qty_err_raw * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)

        total = (
            self.w_base * loss_base
            + self.w_sku * loss_sku
            + self.w_qty * loss_qty
        )
        breakdown = {
            "loss_total": float(total.detach().item()),
            "loss_base": float(loss_base.detach().item()),
            "loss_sku": float(loss_sku.detach().item()),
            "loss_qty": float(loss_qty.detach().item()),
        }
        return total, breakdown
