"""Funções puras de combinação hierárquica de scores, máscaras e métricas Top-K."""

from __future__ import annotations

import math

import numpy as np


def combinar_scores_hierarquicos(
    probs_sku: np.ndarray,
    probs_base: np.ndarray,
    sku_to_base_idx: np.ndarray,
    lambda_hier: float = 0.25,
) -> np.ndarray:
    """Combina p_sku^(1-lambda) * p_base(base(sku))^lambda de forma vetorizada."""
    eps = 1e-7
    p_s = np.clip(probs_sku, eps, 1.0)
    p_b = np.clip(probs_base[:, sku_to_base_idx], eps, 1.0)
    lam = float(np.clip(lambda_hier, 0.0, 1.0))
    return (np.power(p_s, 1.0 - lam) * np.power(p_b, lam)).astype(np.float32)


def aplicar_mascaras_ranking(
    scores: np.ndarray,
    mask_sku: np.ndarray,
    available_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Zera/invalida SKUs já presentes no carrinho e SKUs sem estoque disponível."""
    masked = scores.copy()
    masked = np.where(mask_sku > 0.5, -1.0, masked)
    if available_mask is not None:
        masked = np.where(available_mask < 0.5, -1.0, masked)
    return masked


def calcular_metricas_ranking(
    scores: np.ndarray,
    y_true: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Calcula Hit Rate@K, Recall@K, Precision@K, NDCG@K, MRR e Catalog Coverage@3."""
    n_samples, n_items = scores.shape
    order = np.argsort(-scores, axis=1)

    metrics: dict[str, float] = {}
    mrr_list: list[float] = []

    for i in range(n_samples):
        pos_set = set(np.flatnonzero(y_true[i] > 0.5).tolist())
        if not pos_set:
            continue
        rr = 0.0
        for rank_idx, item_idx in enumerate(order[i], start=1):
            if int(item_idx) in pos_set:
                rr = 1.0 / rank_idx
                break
        mrr_list.append(rr)
    metrics["mrr"] = round(float(np.mean(mrr_list)) if mrr_list else 0.0, 4)

    for k in ks:
        hits: list[float] = []
        recalls: list[float] = []
        precisions: list[float] = []
        ndcgs: list[float] = []
        recommended_unique: set[int] = set()

        topk = order[:, :k]
        for i in range(n_samples):
            pos_set = set(np.flatnonzero(y_true[i] > 0.5).tolist())
            if not pos_set:
                continue
            recs = [int(idx) for idx in topk[i] if scores[i, idx] >= 0.0]
            recommended_unique.update(recs)

            acertos = [1.0 if r in pos_set else 0.0 for r in recs]
            n_hit = sum(acertos)
            hits.append(1.0 if n_hit > 0 else 0.0)
            recalls.append(n_hit / len(pos_set))
            precisions.append(n_hit / k)

            dcg = sum(
                rel / math.log2(rank + 1) for rank, rel in enumerate(acertos, start=1)
            )
            ideal_len = min(len(pos_set), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"hit_rate@{k}"] = round(float(np.mean(hits)) if hits else 0.0, 4)
        metrics[f"recall@{k}"] = round(float(np.mean(recalls)) if recalls else 0.0, 4)
        metrics[f"precision@{k}"] = round(
            float(np.mean(precisions)) if precisions else 0.0, 4
        )
        metrics[f"ndcg@{k}"] = round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4)
        if k == 3:
            metrics["coverage@3"] = round(len(recommended_unique) / n_items, 4)

    return metrics


def calcular_metricas_quantidade(
    pred_qty_log1p: np.ndarray,
    y_qty_log1p: np.ndarray,
    y_sku: np.ndarray,
) -> dict[str, float]:
    """Avalia MAE e Acurácia Exata (%) de qtd_pacotes sobre os SKUs verdadeiros."""
    erros: list[float] = []
    exatos: list[float] = []
    n_samples = len(y_sku)

    for i in range(n_samples):
        pos_indices = np.flatnonzero(y_sku[i] > 0.5)
        for sid in pos_indices:
            true_q = max(1, int(round(math.expm1(float(y_qty_log1p[i, sid])))))
            pred_q = max(1, int(round(math.expm1(float(pred_qty_log1p[i, sid])))))
            erros.append(abs(pred_q - true_q))
            exatos.append(1.0 if pred_q == true_q else 0.0)

    return {
        "qty_mae": round(float(np.mean(erros)) if erros else 0.0, 4),
        "qty_exact_acc": round(float(np.mean(exatos)) if exatos else 0.0, 4),
    }


def projetar_scores_sku_para_base(
    scores_sku: np.ndarray,
    sku_to_base_idx: np.ndarray,
    n_bases: int,
) -> np.ndarray:
    """Agrega scores de SKU para nível de produto_base (máximo entre SKUs da mesma base)."""
    n_samples = len(scores_sku)
    scores_base = np.full((n_samples, n_bases), -1.0, dtype=np.float32)
    for b in range(n_bases):
        cols = np.flatnonzero(sku_to_base_idx == b)
        if len(cols) > 0:
            scores_base[:, b] = scores_sku[:, cols].max(axis=1)
    return scores_base
