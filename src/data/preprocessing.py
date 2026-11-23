"""Etapa 1 — Orquestração de Auditoria, Limpeza e Validação Contábil.

Aplica em sequência as funções puras de `src.utils.cleaning`, valida todas as
invariantes contábeis (0 erros em 3.003 pedidos) e persiste a base auditada em
`data/processed/itens_limpos.parquet` + `reports/preprocessing_audit.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    COLUNA_TIPO_DESCONTO,
    COLUNAS_ITENS_LIMPOS,
    COLUNAS_VENDEDOR_INVALIDAS,
    DATASET_RAW_PATH,
    EVENTOS_CONSOLIDADOS,
    ITENS_LIMPOS_CSV_PATH,
    ITENS_LIMPOS_PATH,
    LIMITE_SUBTOTAL_CONSOLIDADO,
    LIMITE_UNIDADES_CONSOLIDADO,
    MAPA_EMBALAGEM_SEM_MULTIPLICADOR,
    MAPA_EVENTOS,
    MAPA_SINONIMOS_PRODUTOS,
    PREPROCESSING_AUDIT_PATH,
    TIMEZONE,
)
from src.utils.cleaning import (
    calcular_valor_item,
    classificar_pedido_consolidado,
    classificar_pedido_gratuito,
    classificar_pedido_varejo,
    extrair_embalagem_e_multiplicador,
    extrair_produto_base,
    identificar_registro_em_pacotes,
    item_quantidade_valida,
    montar_sku_canonico,
    normalizar_evento,
    normalizar_metodo_pagamento,
    normalizar_texto_produto,
    reconciliar_quantidade_e_preco,
    validar_desconto_em_moeda,
    validar_fechamento_pedido,
)


def executar_preprocessamento(
    df_bruto: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Executa todas as subetapas da Etapa 1 usando as funções puras de `src.utils.cleaning`."""
    df = df_bruto.copy()
    linhas_brutas = int(len(df))
    pedidos_brutos = int(df["pedido_id"].nunique())

    # 1. Remover colunas inválidas de vendedor (login compartilhado)
    colunas_vendedor_presentes = [
        c for c in COLUNAS_VENDEDOR_INVALIDAS if c in df.columns
    ]
    df = df.drop(columns=colunas_vendedor_presentes)

    # 2. Validação automatizada de descontos (antes de remover pedido_tipo_desconto)
    descontos_validados: list[float] = []
    status_desconto_contagem: dict[str, int] = {
        "ja_em_moeda": 0,
        "convertido_de_percentual": 0,
        "inconsistente": 0,
    }
    pedidos_com_desconto_pct_convertido = 0
    pedidos_desconto_inconsistente: list[str] = []

    for row in df.itertuples(index=False):
        valido, desc_reais, status = validar_desconto_em_moeda(
            subtotal=row.pedido_subtotal,
            valor_desconto=row.pedido_valor_desconto,
            valor_total=row.pedido_valor_total,
        )
        status_desconto_contagem[status] += 1
        if status == "convertido_de_percentual":
            pedidos_com_desconto_pct_convertido += 1
        if not valido:
            pedidos_desconto_inconsistente.append(str(row.pedido_id))
        descontos_validados.append(desc_reais)

    if pedidos_desconto_inconsistente:
        raise AssertionError(
            "Falha na validação de descontos nos pedidos: "
            f"{sorted(set(pedidos_desconto_inconsistente))}"
        )

    df["pedido_valor_desconto"] = descontos_validados

    tipo_desconto_removido = False
    if COLUNA_TIPO_DESCONTO in df.columns:
        df = df.drop(columns=[COLUNA_TIPO_DESCONTO])
        tipo_desconto_removido = True

    # 3. Remover itens com quantidade <= 0
    mask_qtd_valida = [
        item_quantidade_valida(q) for q in df["item_pedido_quantidade"]
    ]
    df_removidos_qtd_zero = df.loc[[not m for m in mask_qtd_valida]]
    itens_removidos_qtd_zero = [
        {
            "pedido_id": str(r.pedido_id),
            "evento_nome": str(r.evento_nome),
            "item_pedido_nome_produto": str(r.item_pedido_nome_produto),
            "item_pedido_quantidade": int(r.item_pedido_quantidade),
        }
        for r in df_removidos_qtd_zero.itertuples(index=False)
    ]
    df = df.loc[mask_qtd_valida].copy().reset_index(drop=True)

    # Identificar pedidos divergentes no dado bruto (antes da reconciliação)
    soma_bruta_por_pedido = (
        (df["item_pedido_quantidade"] * df["item_pedido_preco_unitario"])
        .groupby(df["pedido_id"])
        .transform("sum")
    )
    divergente_bruto_flags = [
        not validar_fechamento_pedido(soma_b, sub)
        for soma_b, sub in zip(soma_bruta_por_pedido, df["pedido_subtotal"])
    ]
    pedidos_divergentes_antes = int(
        df.loc[divergente_bruto_flags, "pedido_id"].nunique()
    )

    # 4 & 5. Iterar registros aplicando saneamento, multiplicadores e reconciliação
    registros_processados: list[dict[str, Any]] = []
    casos_reconciliacao: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}

    for row, pedido_divergente in zip(
        df.itertuples(index=False), divergente_bruto_flags
    ):
        evento_original = normalizar_texto_produto(row.evento_nome)
        evento_nome, ponto_venda = normalizar_evento(evento_original, MAPA_EVENTOS)
        metodo_padrao, metodo_grupo = normalizar_metodo_pagamento(
            row.pedido_metodo_pagamento
        )

        nome_limpo = normalizar_texto_produto(row.item_pedido_nome_produto)
        produto_base = extrair_produto_base(nome_limpo, MAPA_SINONIMOS_PRODUTOS)
        tipo_embalagem, multiplicador = extrair_embalagem_e_multiplicador(
            nome_limpo=nome_limpo,
            produto_base=produto_base,
            preco_bruto=row.item_pedido_preco_unitario,
            mapa_sem_mult=MAPA_EMBALAGEM_SEM_MULTIPLICADOR,
        )
        registro_em_pacotes = identificar_registro_em_pacotes(
            nome_limpo=nome_limpo,
            evento_original=evento_original,
            multiplicador=multiplicador,
            pedido_divergente=pedido_divergente,
        )

        if multiplicador == 1:
            casos_reconciliacao["A"] += 1
        elif pedido_divergente:
            casos_reconciliacao["B"] += 1
        elif registro_em_pacotes:
            casos_reconciliacao["C"] += 1
        else:
            casos_reconciliacao["D"] += 1

        (
            qtd_unidades_base,
            qtd_pacotes,
            preco_unitario,
            preco_pacote,
        ) = reconciliar_quantidade_e_preco(
            qtd_bruta=row.item_pedido_quantidade,
            preco_bruto=row.item_pedido_preco_unitario,
            multiplicador=multiplicador,
            pedido_divergente=pedido_divergente,
            registro_em_pacotes=registro_em_pacotes,
        )

        sku_canonico = montar_sku_canonico(produto_base, multiplicador)
        item_valor_total = calcular_valor_item(qtd_unidades_base, preco_unitario)
        is_gratuito = classificar_pedido_gratuito(row.pedido_valor_total)

        registros_processados.append(
            {
                "pedido_id": str(row.pedido_id),
                "pedido_data": row.pedido_data,
                "evento_nome": evento_nome,
                "evento_original": evento_original,
                "ponto_venda": ponto_venda,
                "pedido_metodo_pagamento": metodo_padrao,
                "metodo_pagamento_grupo": metodo_grupo,
                "pedido_subtotal": float(row.pedido_subtotal),
                "pedido_valor_desconto": float(row.pedido_valor_desconto),
                "pedido_valor_total": float(row.pedido_valor_total),
                "produto_base": produto_base,
                "tipo_embalagem": tipo_embalagem,
                "produto_com_multiplicador": sku_canonico,
                "multiplicador": int(multiplicador),
                "qtd_unidades_base": int(qtd_unidades_base),
                "qtd_pacotes": int(qtd_pacotes),
                "preco_unitario": float(preco_unitario),
                "preco_pacote": float(preco_pacote),
                "item_valor_total": float(item_valor_total),
                "is_pedido_gratuito": bool(is_gratuito),
            }
        )

    df_limpo = pd.DataFrame(registros_processados)

    # Invariante contábil pós-reconciliação em 100% dos pedidos
    fechamento_pedidos = df_limpo.groupby("pedido_id", as_index=False).agg(
        soma_itens=("item_valor_total", "sum"),
        pedido_subtotal=("pedido_subtotal", "first"),
        qtd_total_unidades=("qtd_unidades_base", "sum"),
        evento_original=("evento_original", "first"),
        is_pedido_gratuito=("is_pedido_gratuito", "first"),
    )

    pedidos_divergentes_depois = [
        str(r.pedido_id)
        for r in fechamento_pedidos.itertuples(index=False)
        if not validar_fechamento_pedido(r.soma_itens, r.pedido_subtotal)
    ]
    if pedidos_divergentes_depois:
        raise AssertionError(
            "Falha na reconciliação contábil nos pedidos: "
            f"{pedidos_divergentes_depois[:10]}"
        )

    max_erro_fechamento = float(
        (fechamento_pedidos["soma_itens"] - fechamento_pedidos["pedido_subtotal"])
        .abs()
        .max()
    )

    # 6. Classificação de pedidos consolidados / fechamento de caixa e varejo
    mapa_consolidado: dict[str, bool] = {}
    mapa_varejo: dict[str, bool] = {}
    for r in fechamento_pedidos.itertuples(index=False):
        is_cons = classificar_pedido_consolidado(
            subtotal=r.pedido_subtotal,
            qtd_total_unidades=r.qtd_total_unidades,
            evento_original=r.evento_original,
            limite_subtotal=LIMITE_SUBTOTAL_CONSOLIDADO,
            limite_unidades=LIMITE_UNIDADES_CONSOLIDADO,
            eventos_consolidados=EVENTOS_CONSOLIDADOS,
        )
        mapa_consolidado[str(r.pedido_id)] = is_cons
        mapa_varejo[str(r.pedido_id)] = classificar_pedido_varejo(
            is_consolidado=is_cons,
            is_gratuito=r.is_pedido_gratuito,
        )

    df_limpo["is_pedido_consolidado"] = [
        mapa_consolidado[pid] for pid in df_limpo["pedido_id"]
    ]
    df_limpo["is_pedido_varejo"] = [
        mapa_varejo[pid] for pid in df_limpo["pedido_id"]
    ]

    # 7. Tipagem forte do schema final
    df_limpo["pedido_id"] = df_limpo["pedido_id"].astype("string")
    df_limpo["pedido_data"] = pd.to_datetime(
        df_limpo["pedido_data"],
        format="mixed",
        utc=True,
    ).dt.tz_convert(TIMEZONE)

    for col_cat in (
        "evento_nome",
        "evento_original",
        "ponto_venda",
        "pedido_metodo_pagamento",
        "metodo_pagamento_grupo",
        "produto_base",
        "tipo_embalagem",
        "produto_com_multiplicador",
    ):
        df_limpo[col_cat] = df_limpo[col_cat].astype("category")

    for col_int in ("multiplicador", "qtd_unidades_base", "qtd_pacotes"):
        df_limpo[col_int] = df_limpo[col_int].astype("int64")

    for col_float in (
        "pedido_subtotal",
        "pedido_valor_desconto",
        "pedido_valor_total",
        "preco_unitario",
        "preco_pacote",
        "item_valor_total",
    ):
        df_limpo[col_float] = df_limpo[col_float].astype("float64")

    for col_bool in (
        "is_pedido_gratuito",
        "is_pedido_consolidado",
        "is_pedido_varejo",
    ):
        df_limpo[col_bool] = df_limpo[col_bool].astype("bool")

    df_limpo = df_limpo[list(COLUNAS_ITENS_LIMPOS)]

    pedidos_finais = df_limpo.groupby("pedido_id").first()
    audit_report: dict[str, Any] = {
        "entrada": {
            "linhas_brutas": linhas_brutas,
            "pedidos_brutos": pedidos_brutos,
            "colunas_brutas": int(len(df_bruto.columns)),
        },
        "remocoes": {
            "colunas_vendedor_removidas": colunas_vendedor_presentes,
            "pedido_tipo_desconto_removido": tipo_desconto_removido,
            "itens_removidos_qtd_zero_count": len(itens_removidos_qtd_zero),
            "itens_removidos_qtd_zero": itens_removidos_qtd_zero,
        },
        "descontos": {
            "status_por_linha": status_desconto_contagem,
            "convertidos_de_percentual": pedidos_com_desconto_pct_convertido,
            "inconsistentes": len(pedidos_desconto_inconsistente),
        },
        "reconciliacao_contabil": {
            "pedidos_divergentes_antes": pedidos_divergentes_antes,
            "pedidos_divergentes_depois": len(pedidos_divergentes_depois),
            "max_erro_absoluto_reais": round(max_erro_fechamento, 8),
            "casos_por_linha": casos_reconciliacao,
        },
        "catalogo_normalizado": {
            "total_produtos_base": int(df_limpo["produto_base"].nunique()),
            "total_skus_canonicos": int(
                df_limpo["produto_com_multiplicador"].nunique()
            ),
            "produtos_base": sorted(df_limpo["produto_base"].unique().tolist()),
            "skus_canonicos": sorted(
                df_limpo["produto_com_multiplicador"].unique().tolist()
            ),
            "mapa_sinonimos_aplicado": MAPA_SINONIMOS_PRODUTOS,
        },
        "segmentacao_pedidos": {
            "linhas_finais": int(len(df_limpo)),
            "pedidos_finais": int(len(pedidos_finais)),
            "pedidos_gratuitos": int(pedidos_finais["is_pedido_gratuito"].sum()),
            "pedidos_consolidados": int(
                pedidos_finais["is_pedido_consolidado"].sum()
            ),
            "pedidos_varejo": int(pedidos_finais["is_pedido_varejo"].sum()),
            "linhas_varejo": int(df_limpo["is_pedido_varejo"].sum()),
        },
        "schema_final": {col: str(dtype) for col, dtype in df_limpo.dtypes.items()},
    }

    return df_limpo, audit_report


def salvar_artefatos_preprocessamento(
    df_limpo: pd.DataFrame,
    audit_report: dict[str, Any],
    parquet_path: Path = ITENS_LIMPOS_PATH,
    csv_path: Path = ITENS_LIMPOS_CSV_PATH,
    audit_path: Path = PREPROCESSING_AUDIT_PATH,
) -> None:
    """Persiste a base auditada em Parquet, espelho CSV (visualização) e relatório JSON."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    df_limpo.to_parquet(parquet_path, engine="pyarrow", index=False)
    df_limpo.to_csv(csv_path, index=False, encoding="utf-8")
    with audit_path.open("w", encoding="utf-8") as fp:
        json.dump(audit_report, fp, ensure_ascii=False, indent=2)


def main() -> None:
    """Ponto de entrada CLI (`python -m src.data.preprocessing`)."""
    df_bruto = pd.read_csv(DATASET_RAW_PATH)
    df_limpo, audit_report = executar_preprocessamento(df_bruto)
    salvar_artefatos_preprocessamento(df_limpo, audit_report)

    seg = audit_report["segmentacao_pedidos"]
    rec = audit_report["reconciliacao_contabil"]
    cat = audit_report["catalogo_normalizado"]
    print("=== ETAPA 1 — PRÉ-PROCESSAMENTO CONCLUÍDO ===")
    print(f"Linhas: {audit_report['entrada']['linhas_brutas']} -> {seg['linhas_finais']}")
    print(f"Pedidos: {audit_report['entrada']['pedidos_brutos']} -> {seg['pedidos_finais']}")
    print(
        f"Reconciliação contábil: {rec['pedidos_divergentes_antes']} -> "
        f"{rec['pedidos_divergentes_depois']} erros "
        f"(erro máx = R$ {rec['max_erro_absoluto_reais']:.6f})"
    )
    print(
        f"Catálogo normalizado: {cat['total_produtos_base']} produtos base | "
        f"{cat['total_skus_canonicos']} SKUs canônicos"
    )
    print(
        f"Segmentação de pedidos: {seg['pedidos_varejo']} varejo | "
        f"{seg['pedidos_consolidados']} consolidados | "
        f"{seg['pedidos_gratuitos']} gratuitos"
    )
    print(f"Salvo em: {ITENS_LIMPOS_PATH}")
    print(f"Espelho CSV: {ITENS_LIMPOS_CSV_PATH}")
    print(f"Auditoria salva em: {PREPROCESSING_AUDIT_PATH}")


if __name__ == "__main__":
    main()
