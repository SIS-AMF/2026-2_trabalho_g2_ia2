"""Etapa 3 — Análise Exploratória (EDA) e Geração de Evidências Visuais.

Gera os 6 gráficos de alta resolução (300 DPI) em `reports/figures/` e o resumo
estatístico estruturado em `reports/eda_summary.json`.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    EDA_SUMMARY_PATH,
    FIGURES_DIR,
    PEDIDOS_CESTAS_PATH,
    PREPROCESSING_AUDIT_PATH,
)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE_PRIMARY = "#1f5a99"
PALETTE_ACCENT = "#e07a2f"
PALETTE_SUCCESS = "#2a9d6e"
PALETTE_MUTED = "#6c757d"


def _salvar_figura(fig: plt.Figure, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gerar_grafico_01_distribuicao_itens(
    df_cestas: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    """01: Distribuição de itens/SKUs por pedido evidenciando a dor dos ~79% unitários."""
    varejo = df_cestas[df_cestas["is_pedido_varejo"]]
    skus_por_pedido = varejo.groupby("pedido_id")["produto_com_multiplicador"].nunique()
    contagem = skus_por_pedido.value_counts().sort_index()

    n_total = int(len(skus_por_pedido))
    n_unico = int((skus_por_pedido == 1).sum())
    n_multi = int((skus_por_pedido >= 2).sum())
    pct_unico = round(n_unico / n_total * 100.0, 1)
    pct_multi = round(n_multi / n_total * 100.0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})

    cores = [PALETTE_ACCENT if k == 1 else PALETTE_PRIMARY for k in contagem.index]
    bars = ax1.bar(contagem.index.astype(str), contagem.values, color=cores, edgecolor="white")
    for bar, val in zip(bars, contagem.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 35,
            f"{val}\n({val / n_total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_title("Quantidade de SKUs Distintos por Pedido (Varejo)", fontweight="bold")
    ax1.set_xlabel("Número de SKUs no Pedido")
    ax1.set_ylabel("Número de Pedidos")
    ax1.set_ylim(0, contagem.max() * 1.18)

    ax2.pie(
        [n_unico, n_multi],
        labels=[f"1 SKU\n({n_unico} | {pct_unico}%)", f"2+ SKUs\n({n_multi} | {pct_multi}%)"],
        colors=[PALETTE_ACCENT, PALETTE_PRIMARY],
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    ax2.set_title("Dor Real: Oportunidade de Cross-Sell", fontweight="bold")
    fig.suptitle(
        "Quase 4 em cada 5 clientes levam apenas 1 item nas feiras regionais (RS — 2026)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _salvar_figura(fig, figures_dir / "01_distribuicao_itens_por_pedido.png")

    return {
        "pedidos_varejo_total": n_total,
        "pedidos_varejo_1_sku": n_unico,
        "pedidos_varejo_multi_sku": n_multi,
        "pct_varejo_1_sku": pct_unico,
        "pct_varejo_multi_sku": pct_multi,
    }


def gerar_grafico_02_auditoria_expodireto(
    audit: dict[str, Any],
    figures_dir: Path,
) -> None:
    """02: Comprovação da limpeza contábil (139 erros ExpoDireto -> 0)."""
    rec = audit["reconciliacao_contabil"]
    casos = rec["casos_por_linha"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    antes_depois = [rec["pedidos_divergentes_antes"], rec["pedidos_divergentes_depois"]]
    bars1 = ax1.bar(
        ["Antes da Limpeza\n(CSV Bruto)", "Depois da Reconciliação\n(Base Auditada)"],
        antes_depois,
        color=["#c94a4a", PALETTE_SUCCESS],
        width=0.5,
    )
    for bar, val in zip(bars1, antes_depois):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val} pedidos",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.set_title("Pedidos com Erro Contábil (ExpoDireto)", fontweight="bold")
    ax1.set_ylabel("Número de Pedidos Divergentes")
    ax1.set_ylim(0, max(antes_depois) * 1.2)

    labels_casos = [
        "Caso A\n(Unitário x1)",
        "Caso D\n(Pacote Padrão)",
        "Caso B\n(ExpoDireto Diverg.)",
        "Caso C\n(Reg. em Pacotes)",
    ]
    vals_casos = [casos["A"], casos["D"], casos["B"], casos["C"]]
    bars2 = ax2.bar(
        labels_casos,
        vals_casos,
        color=[PALETTE_PRIMARY, PALETTE_SUCCESS, "#c94a4a", PALETTE_ACCENT],
    )
    for bar, val in zip(bars2, vals_casos):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 45,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_title("Distribuição das 3.813 Linhas por Caso de Reconciliação", fontweight="bold")
    ax2.set_ylabel("Número de Itens")
    ax2.set_ylim(0, max(vals_casos) * 1.16)

    fig.suptitle(
        "Auditoria Contábil: 139 pedidos divergentes corrigidos para 0 erros (erro máx = R$ 0,00)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _salvar_figura(fig, figures_dir / "02_auditoria_expodireto_antes_depois.png")


def gerar_grafico_03_top_produtos_e_skus(
    df_cestas: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    """03: Ranking dos Top 12 produtos base (ocorrências por embalagem e faturamento)."""
    varejo = df_cestas[df_cestas["is_pedido_varejo"]].copy()
    varejo["formato"] = np.where(varejo["multiplicador"] == 1, "Avulso (x1)", "Embalagem Múltipla (x2+)")

    top12 = varejo["produto_base"].astype(str).value_counts().head(12).index.tolist()
    sub = varejo[varejo["produto_base"].astype(str).isin(top12)]

    ct_vol = pd.crosstab(sub["produto_base"].astype(str), sub["formato"]).reindex(top12)[::-1]
    rev = (
        sub.groupby(sub["produto_base"].astype(str))["item_valor_total"]
        .sum()
        .reindex(top12)[::-1]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    ct_vol.plot(
        kind="barh",
        stacked=True,
        ax=ax1,
        color=[PALETTE_PRIMARY, PALETTE_ACCENT],
        edgecolor="white",
    )
    ax1.set_title("Top 12 Produtos Base — Ocorrências (Avulso vs. Pacotes)", fontweight="bold")
    ax1.set_xlabel("Número de Ocorrências em Pedidos (Varejo)")
    ax1.set_ylabel("")
    ax1.legend(title="Formato", loc="lower right")

    bars = ax2.barh(rev.index, rev.values, color=PALETTE_SUCCESS, edgecolor="white")
    for bar, val in zip(bars, rev.values):
        ax2.text(
            val + rev.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"R$ {val:,.0f}".replace(",", "."),
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_title("Faturamento no Varejo por Produto Base (R$)", fontweight="bold")
    ax2.set_xlabel("Faturamento Total (R$)")
    ax2.set_xlim(0, rev.max() * 1.18)

    fig.suptitle(
        "Demanda Concentrada e Forte Presença de Embalagens Promocionais (x3, x4, x6)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _salvar_figura(fig, figures_dir / "03_top_produtos_e_skus.png")

    return {
        "top_5_produtos_base": varejo["produto_base"].astype(str).value_counts().head(5).to_dict(),
        "faturamento_varejo_total": round(float(varejo["item_valor_total"].sum()), 2),
    }


def gerar_grafico_04_heatmap_coocorrencia_lift(
    df_cestas: pd.DataFrame,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    """04: Matriz de Lift e co-ocorrência entre os Top 12 produtos base no varejo."""
    varejo = df_cestas[df_cestas["is_pedido_varejo"]]
    top12 = varejo["produto_base"].astype(str).value_counts().head(12).index.tolist()

    baskets = varejo.groupby("pedido_id")["produto_base"].apply(
        lambda s: sorted(set(s.astype(str)))
    )
    n_orders = len(baskets)

    prod_counts = {p: 0 for p in top12}
    pair_counts = {(a, b): 0 for a in top12 for b in top12}

    for items in baskets:
        filtrados = [p for p in items if p in prod_counts]
        for p in filtrados:
            prod_counts[p] += 1
        for a, b in combinations(filtrados, 2):
            pair_counts[(a, b)] += 1
            pair_counts[(b, a)] += 1

    lift_mat = pd.DataFrame(np.nan, index=top12, columns=top12, dtype=float)
    annot_mat = pd.DataFrame("", index=top12, columns=top12, dtype=object)

    top_pairs: list[dict[str, Any]] = []
    for i, a in enumerate(top12):
        for j, b in enumerate(top12):
            if i == j:
                annot_mat.loc[a, b] = "—"
                continue
            c_ab = pair_counts[(a, b)]
            if c_ab > 0 and prod_counts[a] > 0 and prod_counts[b] > 0:
                lift = (c_ab / n_orders) / ((prod_counts[a] / n_orders) * (prod_counts[b] / n_orders))
                lift_mat.loc[a, b] = lift
                annot_mat.loc[a, b] = f"{lift:.1f}x\n({c_ab})"
                if i < j and c_ab >= 10:
                    top_pairs.append(
                        {"produto_a": a, "produto_b": b, "coocorrencias": int(c_ab), "lift": round(float(lift), 2)}
                    )
            else:
                lift_mat.loc[a, b] = 0.0
                annot_mat.loc[a, b] = "0"

    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(
        lift_mat,
        annot=annot_mat,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Lift de Associação"},
        ax=ax,
        annot_kws={"size": 7.5},
        vmin=0,
        vmax=min(4.5, float(np.nanmax(lift_mat.values))),
    )
    ax.set_title(
        "Matriz de Lift e Co-ocorrência (Top 12 Produtos Base — Pedidos de Varejo)\n"
        "Célula = Lift (nº de pedidos juntos)",
        fontweight="bold",
        pad=12,
    )
    plt.xticks(rotation=35, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    _salvar_figura(fig, figures_dir / "04_heatmap_coocorrencia_lift.png")

    top_pairs.sort(key=lambda x: (-x["coocorrencias"], -x["lift"]))
    return top_pairs[:10]


def gerar_grafico_05_vendas_feira_horario(
    df_cestas: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """05: Mix por feira (justifica modelo genérico + filtro de estoque) e curva horária."""
    varejo = df_cestas[df_cestas["is_pedido_varejo"]]

    top_events = (
        varejo.groupby("evento_nome", observed=True)["pedido_id"]
        .nunique()
        .nlargest(6)
        .index.tolist()
    )
    top_prods = varejo["produto_base"].astype(str).value_counts().head(8).index.tolist()
    sub = varejo[varejo["evento_nome"].isin(top_events)]

    ev_orders = sub.groupby("evento_nome", observed=True)["pedido_id"].nunique()
    ct = pd.crosstab(sub["evento_nome"].astype(str), sub["produto_base"].astype(str))
    pct_mix = ct[top_prods].div(ev_orders, axis=0).mul(100.0).reindex(top_events)

    pedidos_hora = varejo.groupby("pedido_id", observed=True).agg(
        hora=("hora", "first"),
        faixa=("faixa_horaria", "first"),
    )
    hora_counts = pedidos_hora["hora"].value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

    sns.heatmap(
        pct_mix,
        annot=True,
        fmt=".1f",
        cmap="OrRd",
        linewidths=0.5,
        cbar_kws={"label": "% dos Pedidos da Feira"},
        ax=ax1,
        annot_kws={"size": 8.5},
    )
    ax1.set_title("Mix por Feira (% pedidos) — Mostra Diferenças de Catálogo/Estoque", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

    cores_hora = [
        "#f4a261" if h < 12 else (PALETTE_PRIMARY if h < 18 else "#264653")
        for h in hora_counts.index
    ]
    ax2.bar(hora_counts.index, hora_counts.values, color=cores_hora, edgecolor="white")
    ax2.set_title("Distribuição de Pedidos por Hora do Dia (Varejo)", fontweight="bold")
    ax2.set_xlabel("Hora do Dia (Manhã=Laranja | Tarde=Azul | Noite=Escuro)")
    ax2.set_ylabel("Número de Pedidos")
    ax2.set_xticks(range(int(hora_counts.index.min()), int(hora_counts.index.max()) + 1))

    fig.suptitle(
        "Perfil das Feiras e Sazonalidade Horária (Base para Modelo Genérico + Filtro de Estoque)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _salvar_figura(fig, figures_dir / "05_vendas_por_feira_e_horario.png")


def gerar_grafico_06_qtd_pacotes_e_outliers(
    df_cestas: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    """06: Distribuição de qtd_pacotes no varejo vs. pedidos consolidados (outliers)."""
    varejo = df_cestas[df_cestas["is_pedido_varejo"]]

    bucket_qty = varejo["qtd_pacotes"].apply(lambda q: "1" if q == 1 else ("2" if q == 2 else "3+"))
    dist_qty = bucket_qty.value_counts().reindex(["1", "2", "3+"])
    n_items_varejo = len(varejo)

    pedidos_all = df_cestas.groupby("pedido_id", observed=True).agg(
        subtotal=("pedido_subtotal", "first"),
        is_consolidado=("is_pedido_consolidado", "first"),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    bars = ax1.bar(
        dist_qty.index,
        dist_qty.values,
        color=[PALETTE_PRIMARY, PALETTE_ACCENT, PALETTE_MUTED],
        width=0.55,
    )
    for bar, val in zip(bars, dist_qty.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val}\n({val / n_items_varejo * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_title("Distribuição de `qtd_pacotes` por Item (Varejo)", fontweight="bold")
    ax1.set_xlabel("Quantidade de Pacotes / Unidades Compradas")
    ax1.set_ylabel("Número de Itens")
    ax1.set_ylim(0, dist_qty.max() * 1.18)

    sub_varejo = pedidos_all[~pedidos_all["is_consolidado"]]["subtotal"]
    sub_cons = pedidos_all[pedidos_all["is_consolidado"]]["subtotal"]
    ax2.hist(sub_varejo, bins=30, color=PALETTE_PRIMARY, alpha=0.8, label=f"Varejo/Balcão (n={len(sub_varejo)})")
    ax2.hist(sub_cons, bins=15, color="#c94a4a", alpha=0.85, label=f"Consolidados/Fechamento (n={len(sub_cons)})")
    ax2.set_xscale("log")
    ax2.axvline(200, color="#c94a4a", linestyle="--", linewidth=1.2, label="Corte R$ 200")
    ax2.set_title("Subtotal dos Pedidos (Escala Log): Varejo vs. Consolidados", fontweight="bold")
    ax2.set_xlabel("Subtotal do Pedido em R$ (escala log)")
    ax2.set_ylabel("Número de Pedidos")
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        "Comportamento de Quantidade no Balcão (80,7% = 1 pacote) e Isolamento dos 18 Fechamentos",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _salvar_figura(fig, figures_dir / "06_distribuicao_qtd_pacotes_e_outliers.png")

    return {
        "pct_qtd_pacotes_1": round(float(dist_qty["1"] / n_items_varejo * 100.0), 1),
        "pct_qtd_pacotes_2": round(float(dist_qty["2"] / n_items_varejo * 100.0), 1),
        "pct_qtd_pacotes_3_plus": round(float(dist_qty["3+"] / n_items_varejo * 100.0), 1),
        "ticket_medio_varejo": round(float(sub_varejo.mean()), 2),
        "ticket_mediano_varejo": round(float(sub_varejo.median()), 2),
    }


def executar_eda(
    df_cestas: pd.DataFrame,
    audit: dict[str, Any],
    figures_dir: Path = FIGURES_DIR,
    summary_path: Path = EDA_SUMMARY_PATH,
) -> dict[str, Any]:
    """Executa toda a Etapa 3 e persiste os 6 gráficos + eda_summary.json."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    s1 = gerar_grafico_01_distribuicao_itens(df_cestas, figures_dir)
    gerar_grafico_02_auditoria_expodireto(audit, figures_dir)
    s3 = gerar_grafico_03_top_produtos_e_skus(df_cestas, figures_dir)
    top_pairs = gerar_grafico_04_heatmap_coocorrencia_lift(df_cestas, figures_dir)
    gerar_grafico_05_vendas_feira_horario(df_cestas, figures_dir)
    s6 = gerar_grafico_06_qtd_pacotes_e_outliers(df_cestas, figures_dir)

    summary = {
        "cestas_varejo": s1,
        "produtos_e_faturamento": s3,
        "top_10_pares_coocorrencia": top_pairs,
        "quantidades_e_ticket": s6,
        "figuras_geradas": sorted(p.name for p in figures_dir.glob("0[1-6]_*.png")),
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    """Ponto de entrada CLI (`python -m src.data.eda`)."""
    df_cestas = pd.read_parquet(PEDIDOS_CESTAS_PATH)
    with PREPROCESSING_AUDIT_PATH.open("r", encoding="utf-8") as fp:
        audit = json.load(fp)
    summary = executar_eda(df_cestas, audit)
    print("=== ETAPA 3 — EDA CONCLUÍDA ===")
    print(f"Figuras geradas ({len(summary['figuras_geradas'])}): {summary['figuras_geradas']}")
    print(f"Ticket médio varejo: R$ {summary['quantidades_e_ticket']['ticket_medio_varejo']:.2f}")
    print(f"Resumo salvo em: {EDA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
