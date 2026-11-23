"""Etapa 6 — Avaliação Comparativa no Teste Hold-Out (20%) e Relatórios.

Responsabilidade exclusiva deste módulo:
- Carregar o modelo principal já treinado (`models/best_checkpoint.pt`).
- Ajustar os 3 modelos comparativos (benchmarks de referência):
  1. `PopularityBaseline` (popularidade global no treino)
  2. `CooccurrenceBaseline` (matriz condicional P(s_j | s_i) no treino)
  3. `BasketDAE` (Denoising Autoencoder — comparativo neural, Slide 42)
- Avaliar os 4 modelos lado a lado sobre o conjunto de Teste Hold-Out (20%)
  utilizando as métricas puras de `src.utils.ranking`.
- Persistir `reports/evaluation_summary.csv`, `reports/evaluation_report.json`,
  `reports/figures/08_comparacao_modelos.png` e `reports/qualitative_cases.md`.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BEST_CHECKPOINT_PATH,
    CATALOG_METADATA_PATH,
    EVALUATION_REPORT_PATH,
    EVALUATION_SUMMARY_PATH,
    FIGURES_DIR,
    PEDIDOS_CESTAS_PATH,
    QUALITATIVE_CASES_PATH,
    SEED,
    SPLITS_DIR,
)
from src.data.dataset_builder import carregar_splits
from src.modeling.baselines import CooccurrenceBaseline, PopularityBaseline
from src.modeling.models import BasketDAE, FairMultiTaskRecommender
from src.modeling.train import fixar_seeds, prever_mlp_multitask
from src.utils.ranking import (
    aplicar_mascaras_ranking,
    calcular_metricas_quantidade,
    calcular_metricas_ranking,
    projetar_scores_sku_para_base,
)


# =============================================================================
# MODELO COMPARATIVO NEURAL (BENCHMARK): DENOISING AUTOENCODER (BasketDAE)
# =============================================================================


def treinar_basket_dae(
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    n_skus: int = 34,
    max_epochs: int = 100,
    patience: int = 16,
    seed: int = SEED,
) -> tuple[BasketDAE, dict[str, Any]]:
    """Treina o Denoising Autoencoder (`BasketDAE`) usado como benchmark comparativo."""
    fixar_seeds(seed)
    model = BasketDAE(
        n_skus=n_skus,
        n_time=6,
        hidden_dim=64,
        bottleneck_dim=24,
        dropout=0.25,
        masking_noise=0.35,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    # Reconstrói a cesta completa (entrada + alvo) a partir da entrada corrompida por ruído
    x_sku_train = train_arr["X"][:, :n_skus]
    x_time_train = train_arr["X"][:, -6:]
    y_full_train = np.clip(x_sku_train + train_arr["y_sku"], 0.0, 1.0).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(y_full_train),
        torch.from_numpy(x_time_train),
        torch.from_numpy(y_full_train),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    val_x_sku = torch.from_numpy(val_arr["X"][:, :n_skus])
    val_x_time = torch.from_numpy(val_arr["X"][:, -6:])

    best_ndcg = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx_full, bx_time, by_full in loader:
            if len(bx_full) < 2:
                continue
            optimizer.zero_grad()
            logits = model(bx_full, bx_time)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, by_full)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(val_x_sku, val_x_time)
            v_scores = torch.sigmoid(v_logits).numpy()
            v_masked = aplicar_mascaras_ranking(v_scores, val_arr["mask_sku"])
            v_ndcg = calcular_metricas_ranking(v_masked, val_arr["y_sku"], ks=(1, 3, 5))["ndcg@3"]

        if v_ndcg > best_ndcg + 1e-5:
            best_ndcg = v_ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_val_ndcg@3": best_ndcg, "best_epoch": best_epoch}


@torch.no_grad()
def prever_dae(
    model: BasketDAE,
    X: np.ndarray,
    n_skus: int = 34,
) -> tuple[np.ndarray, np.ndarray]:
    """Executa inferência em lote para o benchmark `BasketDAE`."""
    model.eval()
    logits = model(torch.from_numpy(X[:, :n_skus]), torch.from_numpy(X[:, -6:]))
    scores = torch.sigmoid(logits).numpy().astype(np.float32)
    qty_pred = np.full((len(X), n_skus), np.log1p(1.0), dtype=np.float32)
    return scores, qty_pred


# =============================================================================
# AVALIAÇÃO QUANTITATIVA, GRÁFICO COMPARATIVO E CASOS QUALITATIVOS
# =============================================================================


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

        linhas.append(
            f"## Caso {num} — Pedido `{meta['pedido_id'][:8]}...` "
            f"({dias[meta['dia_semana']]}, {meta['hora']:02d}h) — {acertou}"
        )
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


def executar_avaliacao_comparativa(
    checkpoint_path: Path = BEST_CHECKPOINT_PATH,
    evaluation_report_path: Path = EVALUATION_REPORT_PATH,
    seed: int = SEED,
) -> dict[str, Any]:
    """Avalia o modelo principal salvo contra os 3 benchmarks no Teste Hold-Out (20%)."""
    df_cestas = pd.read_parquet(PEDIDOS_CESTAS_PATH)
    with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as fp:
        catalogo = json.load(fp)
    bundle = carregar_splits(SPLITS_DIR)

    n_skus = int(catalogo["n_skus"])
    sku_to_base_idx = np.array([s["base_id"] for s in catalogo["skus"]], dtype=np.int64)

    # 1. Carregar modelo principal já treinado (FairMultiTaskRecommender)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]
    mlp_model = FairMultiTaskRecommender(
        in_dim=int(cfg["in_dim"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        n_bases=int(cfg["n_bases"]),
        n_skus=int(cfg["n_skus"]),
        dropout=float(cfg["dropout"]),
    )
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()
    lambda_hier = float(checkpoint["hparams"]["lambda_hier"])

    # 2. Ajustar os 3 modelos comparativos (usando apenas treino / validação)
    pop_model = PopularityBaseline(n_skus=n_skus).fit(df_cestas, bundle["splits"]["train_ids"])
    cooc_model = CooccurrenceBaseline(n_skus=n_skus, smoothing=1.0).fit(
        df_cestas, bundle["splits"]["train_ids"]
    )
    dae_model, dae_info = treinar_basket_dae(
        train_arr=bundle["train"],
        val_arr=bundle["val"],
        n_skus=n_skus,
        seed=seed,
    )

    # 3. Avaliação imparcial no conjunto de Teste Hold-Out (20%)
    test_arr = bundle["test"]
    pop_scores, pop_qty = pop_model.predict_scores(test_arr["X"], test_arr["mask_sku"])
    cooc_scores, cooc_qty = cooc_model.predict_scores(test_arr["X"], test_arr["mask_sku"])
    dae_scores, dae_qty = prever_dae(dae_model, test_arr["X"], n_skus=n_skus)
    mlp_scores, mlp_qty = prever_mlp_multitask(
        mlp_model, test_arr["X"], sku_to_base_idx, lambda_hier
    )

    resultados = [
        avaliar_modelo_completo("Baseline 1: Popularidade", pop_scores, pop_qty, test_arr, catalogo),
        avaliar_modelo_completo("Baseline 2: Co-ocorrência", cooc_scores, cooc_qty, test_arr, catalogo),
        avaliar_modelo_completo("BasketDAE (Autoencoder)", dae_scores, dae_qty, test_arr, catalogo),
        avaliar_modelo_completo("MLP Multi-Task (Principal)", mlp_scores, mlp_qty, test_arr, catalogo),
    ]
    df_eval = salvar_resumo_avaliacao(resultados)

    gerar_relatorio_qualitativo(
        scores_sku=mlp_scores,
        pred_qty_log1p=mlp_qty,
        test_meta=bundle["test_meta"],
        mask_sku=test_arr["mask_sku"],
        catalogo=catalogo,
    )

    mlp_eval = next(
        (r for r in resultados if "MLP Multi-Task" in r["modelo"]),
        resultados[-1],
    )
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dae_info": dae_info,
        "principal_test_metrics": mlp_eval,
        "all_models_test_evaluation": resultados,
    }
    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    return {"report": report, "df_eval": df_eval}


def main() -> None:
    """Ponto de entrada CLI (`python -m src.modeling.evaluate`)."""
    out = executar_avaliacao_comparativa(seed=SEED)
    print("=== ETAPA 6 — AVALIAÇÃO COMPARATIVA NO TESTE HOLD-OUT (20%) ===")
    cols = [
        "modelo",
        "sku_hit_rate@1",
        "sku_hit_rate@3",
        "sku_ndcg@3",
        "sku_mrr",
        "sku_coverage@3",
        "base_hit_rate@3",
        "qty_mae",
        "qty_exact_acc",
    ]
    print(out["df_eval"][cols].to_string(index=False))
    print(f"\nResumo CSV: {EVALUATION_SUMMARY_PATH}")
    print(f"Relatório JSON: {EVALUATION_REPORT_PATH}")
    print(f"Gráfico comparativo: {FIGURES_DIR / '08_comparacao_modelos.png'}")
    print(f"Casos qualitativos: {QUALITATIVE_CASES_PATH}")


if __name__ == "__main__":
    main()
