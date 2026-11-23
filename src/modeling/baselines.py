"""Etapa 5 — Baselines Estatísticos Não-Neurais (Popularidade Global e Co-ocorrência/Lift).

Servem como referência científica obrigatória para medir o ganho real das redes
neurais (`FairMultiTaskRecommender` e `BasketDAE`) no mesmo split 80/20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityBaseline:
    """Baseline 1: Ranqueia SKUs pela frequência observada no conjunto de treino."""

    def __init__(self, n_skus: int) -> None:
        self.n_skus = int(n_skus)
        self.sku_scores_: np.ndarray = np.zeros(self.n_skus, dtype=np.float32)

    def fit(self, df_cestas: pd.DataFrame, train_pedido_ids: list[str]) -> "PopularityBaseline":
        train_set = set(train_pedido_ids)
        sub = df_cestas[df_cestas["pedido_id"].astype(str).isin(train_set)]
        counts = np.zeros(self.n_skus, dtype=np.float64)
        for sid, cnt in sub["sku_id"].value_counts().items():
            counts[int(sid)] = float(cnt)
        total = counts.sum()
        self.sku_scores_ = ((counts + 1.0) / (total + self.n_skus)).astype(np.float32)
        return self

    def predict_scores(self, X: np.ndarray, mask_sku: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (scores_sku [N, n_skus], pred_qty_log1p [N, n_skus])."""
        n_samples = len(X)
        scores = np.tile(self.sku_scores_, (n_samples, 1)).astype(np.float32)
        scores = np.where(mask_sku > 0.5, -1.0, scores)
        # Moda global no varejo: 1 pacote (80,7%) -> log1p(1)
        qty_pred = np.full((n_samples, self.n_skus), np.log1p(1.0), dtype=np.float32)
        return scores, qty_pred


class CooccurrenceBaseline:
    """Baseline 2: Matriz de Co-ocorrência condicional P(s_j | s_i) + suavização + mediana de qtd."""

    def __init__(self, n_skus: int, smoothing: float = 1.0) -> None:
        self.n_skus = int(n_skus)
        self.smoothing = float(smoothing)
        self.cond_prob_: np.ndarray = np.zeros((self.n_skus, self.n_skus), dtype=np.float32)
        self.prior_prob_: np.ndarray = np.zeros(self.n_skus, dtype=np.float32)
        self.median_qty_log1p_: np.ndarray = np.full(
            self.n_skus, np.log1p(1.0), dtype=np.float32
        )

    def fit(self, df_cestas: pd.DataFrame, train_pedido_ids: list[str]) -> "CooccurrenceBaseline":
        train_set = set(train_pedido_ids)
        sub = df_cestas[df_cestas["pedido_id"].astype(str).isin(train_set)]

        item_counts = np.zeros(self.n_skus, dtype=np.float64)
        cooc = np.zeros((self.n_skus, self.n_skus), dtype=np.float64)

        for _, grp in sub.groupby("pedido_id", sort=False):
            sids = sorted(set(int(s) for s in grp["sku_id"]))
            for s in sids:
                item_counts[s] += 1.0
            if len(sids) >= 2:
                for i in sids:
                    for j in sids:
                        if i != j:
                            cooc[i, j] += 1.0

        total_items = item_counts.sum()
        self.prior_prob_ = ((item_counts + 1.0) / (total_items + self.n_skus)).astype(np.float32)

        # P(j | i) suavizado em direção ao prior de popularidade
        cond = np.zeros((self.n_skus, self.n_skus), dtype=np.float64)
        for i in range(self.n_skus):
            denom = item_counts[i] + self.smoothing * self.n_skus
            cond[i, :] = (cooc[i, :] + self.smoothing * self.prior_prob_ * self.n_skus) / max(denom, 1e-8)
            cond[i, i] = 0.0
        self.cond_prob_ = cond.astype(np.float32)

        # Mediana histórica de qtd_pacotes por SKU no treino
        med_qty = np.full(self.n_skus, np.log1p(1.0), dtype=np.float32)
        for sid, grp in sub.groupby("sku_id"):
            med_qty[int(sid)] = float(np.log1p(max(1.0, float(grp["qtd_pacotes"].median()))))
        self.median_qty_log1p_ = med_qty
        return self

    def predict_scores(self, X: np.ndarray, mask_sku: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Soma P(s_j | s_i) sobre os SKUs no carrinho (com fallback de popularidade se vazio)."""
        n_samples = len(X)
        cart_bin = X[:, : self.n_skus]
        raw_scores = cart_bin @ self.cond_prob_

        # Normalizar pelo número de itens no carrinho; se carrinho vazio, usar prior
        cart_sizes = cart_bin.sum(axis=1, keepdims=True)
        scores = np.where(
            cart_sizes > 0,
            raw_scores / np.maximum(cart_sizes, 1.0),
            np.tile(self.prior_prob_, (n_samples, 1)),
        ).astype(np.float32)
        scores = np.where(mask_sku > 0.5, -1.0, scores)

        qty_pred = np.tile(self.median_qty_log1p_, (n_samples, 1)).astype(np.float32)
        return scores, qty_pred
