"""Etapa 2 — Orquestração de Engenharia de Features e Catálogo Oficial.

Utiliza as funções puras de `src.utils.temporal` para construir a tabela
consolidada de cestas (`pedidos_cestas.parquet`) e o catálogo oficial
(`catalog_metadata.json`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    CATALOG_METADATA_PATH,
    ITENS_LIMPOS_PATH,
    PEDIDOS_CESTAS_CSV_PATH,
    PEDIDOS_CESTAS_PATH,
)
from src.utils.temporal import (
    calcular_economia_unitaria_pct,
    classificar_faixa_horaria,
    codificar_hora_ciclica,
    eh_fim_de_semana,
    montar_display_name,
)


def construir_catalogo_oficial(df_itens: pd.DataFrame) -> dict[str, Any]:
    """Constrói o catálogo determinístico com 20 produtos base e 34 SKUs canônicos."""
    df = df_itens.copy()
    df["produto_base_str"] = df["produto_base"].astype(str)
    df["sku_norm_str"] = df["produto_com_multiplicador"].astype(str)
    df["tipo_embalagem_str"] = df["tipo_embalagem"].astype(str)

    avulsos = df[df["multiplicador"] == 1]
    preco_avulso_por_base: dict[str, float] = {}
    for base, grp in df.groupby("produto_base_str"):
        sub_avulso = avulsos[avulsos["produto_base_str"] == base]
        if len(sub_avulso) > 0:
            preco_avulso_por_base[base] = round(
                float(sub_avulso["preco_unitario"].median()), 2
            )
        else:
            preco_avulso_por_base[base] = round(
                float(grp["preco_unitario"].median()), 2
            )

    bases_ordenadas = sorted(df["produto_base_str"].unique().tolist())
    base_to_id = {base: idx for idx, base in enumerate(bases_ordenadas)}
    bases_meta = [
        {
            "base_id": base_to_id[base],
            "produto_base": base,
            "preco_unitario_avulso_ref": preco_avulso_por_base[base],
        }
        for base in bases_ordenadas
    ]

    skus_ordenados = sorted(df["sku_norm_str"].unique().tolist())
    df_varejo = df[df["is_pedido_varejo"]]
    freq_varejo_map = df_varejo["sku_norm_str"].value_counts().to_dict()

    skus_meta: list[dict[str, Any]] = []
    for sku_id, sku_norm in enumerate(skus_ordenados):
        sub = df[df["sku_norm_str"] == sku_norm]
        produto_base = str(sub["produto_base_str"].iloc[0])
        multiplicador = int(sub["multiplicador"].iloc[0])
        tipo_embalagem = str(sub["tipo_embalagem_str"].mode().iloc[0])
        preco_pacote_ref = round(float(sub["preco_pacote"].median()), 2)
        preco_unitario_ref = round(preco_pacote_ref / multiplicador, 2)
        economia_pct = calcular_economia_unitaria_pct(
            preco_unitario=preco_unitario_ref,
            preco_avulso_ref=preco_avulso_por_base[produto_base],
        )
        display_name = montar_display_name(
            produto_base=produto_base,
            tipo_embalagem=tipo_embalagem,
            multiplicador=multiplicador,
        )
        skus_meta.append(
            {
                "sku_id": sku_id,
                "sku_norm": sku_norm,
                "produto_base": produto_base,
                "base_id": base_to_id[produto_base],
                "tipo_embalagem": tipo_embalagem,
                "display_name": display_name,
                "multiplicador": multiplicador,
                "preco_unitario_ref": preco_unitario_ref,
                "preco_pacote_ref": preco_pacote_ref,
                "economia_unitaria_pct": economia_pct,
                "frequencia_varejo": int(freq_varejo_map.get(sku_norm, 0)),
                "frequencia_total": int(len(sub)),
            }
        )

    return {
        "n_bases": len(bases_meta),
        "n_skus": len(skus_meta),
        "bases": bases_meta,
        "skus": skus_meta,
    }


def construir_tabela_cestas(
    df_itens: pd.DataFrame,
    catalogo: dict[str, Any],
) -> pd.DataFrame:
    """Consolida itens por (pedido_id, produto_com_multiplicador) e gera features temporais."""
    df = df_itens.copy()
    df["sku_norm_str"] = df["produto_com_multiplicador"].astype(str)
    df["produto_base_str"] = df["produto_base"].astype(str)

    preco_avulso_map = {
        b["produto_base"]: b["preco_unitario_avulso_ref"]
        for b in catalogo["bases"]
    }
    sku_info_map = {s["sku_norm"]: s for s in catalogo["skus"]}

    agrupado = df.groupby(["pedido_id", "sku_norm_str"], as_index=False).agg(
        pedido_data=("pedido_data", "first"),
        evento_nome=("evento_nome", "first"),
        evento_original=("evento_original", "first"),
        ponto_venda=("ponto_venda", "first"),
        pedido_metodo_pagamento=("pedido_metodo_pagamento", "first"),
        metodo_pagamento_grupo=("metodo_pagamento_grupo", "first"),
        pedido_subtotal=("pedido_subtotal", "first"),
        pedido_valor_desconto=("pedido_valor_desconto", "first"),
        pedido_valor_total=("pedido_valor_total", "first"),
        produto_base=("produto_base_str", "first"),
        multiplicador=("multiplicador", "first"),
        qtd_unidades_base=("qtd_unidades_base", "sum"),
        qtd_pacotes=("qtd_pacotes", "sum"),
        item_valor_total=("item_valor_total", "sum"),
        is_pedido_gratuito=("is_pedido_gratuito", "first"),
        is_pedido_consolidado=("is_pedido_consolidado", "first"),
        is_pedido_varejo=("is_pedido_varejo", "first"),
    )

    agrupado["preco_unitario"] = (
        agrupado["item_valor_total"] / agrupado["qtd_unidades_base"]
    )
    agrupado["preco_pacote"] = agrupado["preco_unitario"] * agrupado["multiplicador"]

    agrupado["sku_id"] = [sku_info_map[s]["sku_id"] for s in agrupado["sku_norm_str"]]
    agrupado["base_id"] = [sku_info_map[s]["base_id"] for s in agrupado["sku_norm_str"]]
    agrupado["tipo_embalagem"] = [
        sku_info_map[s]["tipo_embalagem"] for s in agrupado["sku_norm_str"]
    ]
    agrupado["display_name"] = [
        sku_info_map[s]["display_name"] for s in agrupado["sku_norm_str"]
    ]
    agrupado["economia_unitaria_pct"] = [
        calcular_economia_unitaria_pct(pu, preco_avulso_map[base])
        for pu, base in zip(agrupado["preco_unitario"], agrupado["produto_base"])
    ]

    horas = agrupado["pedido_data"].dt.hour.astype(int).tolist()
    dias_semana = agrupado["pedido_data"].dt.day_of_week.astype(int).tolist()
    meses = agrupado["pedido_data"].dt.month.astype(int).tolist()

    sin_cos = [codificar_hora_ciclica(h) for h in horas]
    agrupado["hora"] = pd.Series(horas, dtype="int64")
    agrupado["hora_sin"] = pd.Series([sc[0] for sc in sin_cos], dtype="float64")
    agrupado["hora_cos"] = pd.Series([sc[1] for sc in sin_cos], dtype="float64")
    agrupado["faixa_horaria"] = pd.Series(
        [classificar_faixa_horaria(h) for h in horas], dtype="category"
    )
    agrupado["dia_semana"] = pd.Series(dias_semana, dtype="int64")
    agrupado["fim_de_semana"] = pd.Series(
        [eh_fim_de_semana(d) for d in dias_semana], dtype="bool"
    )
    agrupado["mes"] = pd.Series(meses, dtype="int64")

    cesta_stats = agrupado.groupby("pedido_id", as_index=False).agg(
        n_skus_pedido=("sku_norm_str", "nunique"),
        n_bases_pedido=("produto_base", "nunique"),
    )
    agrupado = agrupado.merge(cesta_stats, on="pedido_id", how="left")
    agrupado["is_multi_sku"] = (agrupado["n_skus_pedido"] >= 2).astype("bool")
    agrupado["teve_desconto"] = (agrupado["pedido_valor_desconto"] > 0).astype("bool")

    agrupado = agrupado.rename(columns={"sku_norm_str": "produto_com_multiplicador"})
    agrupado["pedido_id"] = agrupado["pedido_id"].astype("string")
    for col_cat in (
        "evento_nome",
        "evento_original",
        "ponto_venda",
        "pedido_metodo_pagamento",
        "metodo_pagamento_grupo",
        "produto_base",
        "tipo_embalagem",
        "produto_com_multiplicador",
        "display_name",
        "faixa_horaria",
    ):
        agrupado[col_cat] = agrupado[col_cat].astype("category")

    for col_int in (
        "sku_id",
        "base_id",
        "multiplicador",
        "qtd_unidades_base",
        "qtd_pacotes",
        "hora",
        "dia_semana",
        "mes",
        "n_skus_pedido",
        "n_bases_pedido",
    ):
        agrupado[col_int] = agrupado[col_int].astype("int64")

    for col_float in (
        "pedido_subtotal",
        "pedido_valor_desconto",
        "pedido_valor_total",
        "preco_unitario",
        "preco_pacote",
        "item_valor_total",
        "economia_unitaria_pct",
        "hora_sin",
        "hora_cos",
    ):
        agrupado[col_float] = agrupado[col_float].astype("float64")

    assert (
        agrupado["qtd_pacotes"] * agrupado["multiplicador"]
        == agrupado["qtd_unidades_base"]
    ).all(), "Violação da invariante qtd_pacotes * multiplicador == qtd_unidades_base"
    assert (
        (agrupado["preco_unitario"] * agrupado["multiplicador"] - agrupado["preco_pacote"])
        .abs()
        .max()
        < 0.02
    ), "Violação da invariante preco_unitario * multiplicador == preco_pacote"
    assert agrupado.isna().sum().sum() == 0, "Valores nulos encontrados em pedidos_cestas"

    return agrupado


def executar_engenharia_features(
    df_itens: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Executa a Etapa 2 completa retornando (df_cestas, catalogo_oficial)."""
    catalogo = construir_catalogo_oficial(df_itens)
    df_cestas = construir_tabela_cestas(df_itens, catalogo)
    return df_cestas, catalogo


def salvar_artefatos_features(
    df_cestas: pd.DataFrame,
    catalogo: dict[str, Any],
    cestas_path: Path = PEDIDOS_CESTAS_PATH,
    cestas_csv_path: Path = PEDIDOS_CESTAS_CSV_PATH,
    catalogo_path: Path = CATALOG_METADATA_PATH,
) -> None:
    """Salva `pedidos_cestas.parquet`, espelho CSV (visualização) e `catalog_metadata.json`."""
    cestas_path.parent.mkdir(parents=True, exist_ok=True)
    cestas_csv_path.parent.mkdir(parents=True, exist_ok=True)
    catalogo_path.parent.mkdir(parents=True, exist_ok=True)
    df_cestas.to_parquet(cestas_path, engine="pyarrow", index=False)
    df_cestas.to_csv(cestas_csv_path, index=False, encoding="utf-8")
    with catalogo_path.open("w", encoding="utf-8") as fp:
        json.dump(catalogo, fp, ensure_ascii=False, indent=2)


def main() -> None:
    """Ponto de entrada CLI (`python -m src.data.features`)."""
    df_itens = pd.read_parquet(ITENS_LIMPOS_PATH)
    df_cestas, catalogo = executar_engenharia_features(df_itens)
    salvar_artefatos_features(df_cestas, catalogo)

    varejo = df_cestas[df_cestas["is_pedido_varejo"]]
    n_pedidos_varejo = int(varejo["pedido_id"].nunique())
    n_multi_sku_varejo = int(
        varejo[varejo["is_multi_sku"]]["pedido_id"].nunique()
    )
    print("=== ETAPA 2 — ENGENHARIA DE FEATURES CONCLUÍDA ===")
    print(f"Linhas consolidadas (pedido x SKU): {len(df_cestas)}")
    print(f"Catálogo oficial: {catalogo['n_bases']} produtos base | {catalogo['n_skus']} SKUs")
    print(
        f"Varejo: {n_pedidos_varejo} pedidos | {n_multi_sku_varejo} multi-SKU "
        f"({n_multi_sku_varejo / n_pedidos_varejo * 100:.1f}%)"
    )
    print(f"Salvo em: {PEDIDOS_CESTAS_PATH}")
    print(f"Espelho CSV: {PEDIDOS_CESTAS_CSV_PATH}")
    print(f"Catálogo salvo em: {CATALOG_METADATA_PATH}")


if __name__ == "__main__":
    main()
