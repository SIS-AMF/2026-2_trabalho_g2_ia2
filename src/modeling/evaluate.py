"""Etapa 6 — Orquestração de Avaliação Top-K, Gráficos Comparativos e Relatório Qualitativo.

Utiliza as funções puras de ranking e métricas de `src.utils.ranking` para avaliar
modelos no conjunto de Teste Hold-Out (20%) e gerar os artefatos em `reports/`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    EVALUATION_SUMMARY_PATH,
    FIGURES_DIR,
    QUALITATIVE_CASES_PATH,
)
from src.utils.ranking import (
    aplicar_mascaras_ranking,
    calcular_metricas_quantidade,
    calcular_metricas_ranking,
    projetar_scores_sku_para_base,
)


def avaliar_modelo_completo(
    nome_modelo: str,
    scores_sku: np.ndarray,
    pred_qty_log1p: np.ndarray,
    split_data: dict[str, np.ndarray],
    catalogo: dict[str, Any],
) -> dict[str, Any]:
    """Computa métricas SKU, Produto Base, Coverage@3 e Quantidade para um modelo."""
    sku_to_base_idx = np.array([s["base_id"] for s in catalogo["skus"]], dtype=np.int64)
    masked_sku = aplicar_mascaras_ranking(scores_sku, split_data["mask_sku"])
    sku_metrics = calcular_metricas_ranking(masked_sku, split_data["y_sku"], ks=(1, 3, 5))

    scores_base = projetar_scores_sku_para_base(masked_sku, sku_to_base_idx, catalogo["n_bases"])
    base_metrics = calcular_metricas_ranking(scores_base, split_data["y_base"], ks=(1, 3, 5))

    qty_metrics = calcular_metricas_quantidade(
        pred_qty_log1p=pred_qty_log1p,
        y_qty_log1p=split_data["y_qty"],
        y_sku=split_data["y_sku"],
    )

    row: dict[str, Any] = {"modelo": nome_modelo}
    for k, v in sku_metrics.items():
        row[f"sku_{k}"] = v
    row["base_hit_rate@3"] = base_metrics["hit_rate@3"]
    row["base_ndcg@3"] = base_metrics["ndcg@3"]
    row.update(qty_metrics)
    return row


def gerar_grafico_comparacao_modelos(
    df_eval: pd.DataFrame,
    caminho: Path = FIGURES_DIR / "08_comparacao_modelos.png",
) -> None:
    """08: Gráfico comparativo dos 4 modelos no Teste Hold-Out (20%)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    metricas_plot = [
        ("sku_hit_rate@3", "Hit Rate@3 (SKU)"),
        ("sku_ndcg@3", "NDCG@3 (SKU)"),
        ("base_hit_rate@3", "Hit Rate@3 (Produto Base)"),
        ("sku_coverage@3", "Catalog Coverage@3"),
    ]

    x = np.arange(len(df_eval))
    width = 0.19
    cores = ["#1f5a99", "#2a9d6e", "#e07a2f", "#7b4ea3"]

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    for idx, ((col, label), cor) in enumerate(zip(metricas_plot, cores)):
        offset = (idx - 1.5) * width
        vals = df_eval[col].values * 100.0
        bars = ax.bar(x + offset, vals, width=width, label=label, color=cor, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df_eval["modelo"].tolist(), fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Desempenho (%)")
    ax.set_ylim(0, max(float(df_eval[[m[0] for m in metricas_plot]].values.max()) * 118.0, 65.0))
    ax.set_title(
        "Comparação Final no Conjunto de Teste Hold-Out (20% dos Pedidos — Leave-One-Out)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper left", ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gerar_relatorio_qualitativo(
    scores_sku: np.ndarray,
    pred_qty_log1p: np.ndarray,
    test_meta: list[dict[str, Any]],
    mask_sku: np.ndarray,
    catalogo: dict[str, Any],
    caminho: Path = QUALITATIVE_CASES_PATH,
    n_exemplos: int = 8,
) -> None:
    """Gera `reports/qualitative_cases.md` com exemplos reais do Teste Hold-Out (20%)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sku_by_id = {s["sku_id"]: s for s in catalogo["skus"]}
    masked = aplicar_mascaras_ranking(scores_sku, mask_sku)

    escolhidos: list[int] = []
    pedidos_vistos: set[str] = set()
    acertos_top1: list[int] = []
    acertos_top3: list[int] = []
    outros: list[int] = []

    for idx, meta in enumerate(test_meta):
        pid = meta["pedido_id"]
        if pid in pedidos_vistos:
            continue
        pedidos_vistos.add(pid)
        top3 = np.argsort(-masked[idx])[:3].tolist()
        tgt = meta["target_skus"][0]
        if tgt == top3[0]:
            acertos_top1.append(idx)
        elif tgt in top3:
            acertos_top3.append(idx)
        else:
            outros.append(idx)

    escolhidos.extend(acertos_top1[:4])
    escolhidos.extend(acertos_top3[:2])
    escolhidos.extend(outros[:2])
    escolhidos = escolhidos[:n_exemplos]

    dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    linhas = [
        "# Análise Qualitativa — Exemplos Reais no Conjunto de Teste (20%)\n",
        "Amostra de cestas reais do conjunto de teste Hold-Out avaliadas pelo `FairMultiTaskRecommender` (Modelo Genérico).\n",
    ]

    for num, idx in enumerate(escolhidos, start=1):
        meta = test_meta[idx]
        entrada_txt = ", ".join(
            f"**{sku_by_id[s]['display_name']}** ({q} pct)"
            for s, q in zip(meta["input_skus"], meta["input_qtys"])
        )
        tgt_s = meta["target_skus"][0]
        tgt_q = meta["target_qtys"][0]
        tgt_info = sku_by_id[tgt_s]

        top3 = np.argsort(-masked[idx])[:3].tolist()
        acertou = "ACERTO NO TOP-3" if tgt_s in top3 else "FORA DO TOP-3"

        linhas.append(f"## Caso {num} — Pedido `{meta['pedido_id'][:8]}...` ({dias[meta['dia_semana']]}, {meta['hora']:02d}h) — {acertou}")
        linhas.append(f"- **Carrinho de Entrada:** {entrada_txt}")
        linhas.append(
            f"- **Item Real Comprado (Ground Truth):** `{tgt_info['display_name']}` "
            f"(mult `x{tgt_info['multiplicador']}`, `{tgt_q}` pct = `{tgt_q * tgt_info['multiplicador']}` un.)\n"
        )
        linhas.append("| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |")
        linhas.append("| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |")
        for rank, sid in enumerate(top3, start=1):
            info = sku_by_id[int(sid)]
            qp = max(1, int(round(math.expm1(float(pred_qty_log1p[idx, sid])))))
            qu = qp * int(info["multiplicador"])
            conf = float(masked[idx, sid]) * 100.0
            marcador = " **[REAL]**" if int(sid) == tgt_s else ""
            linhas.append(
                f"| {rank} | {info['display_name']}{marcador} | {info['produto_base']} | "
                f"x{info['multiplicador']} | {qp} | {qu} | R$ {info['preco_pacote_ref']:.2f} | {conf:.1f}% |"
            )
        linhas.append("")

    caminho.write_text("\n".join(linhas), encoding="utf-8")


def salvar_resumo_avaliacao(
    resultados: list[dict[str, Any]],
    csv_path: Path = EVALUATION_SUMMARY_PATH,
) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval = pd.DataFrame(resultados)
    df_eval.to_csv(csv_path, index=False)
    gerar_grafico_comparacao_modelos(df_eval)
    return df_eval
