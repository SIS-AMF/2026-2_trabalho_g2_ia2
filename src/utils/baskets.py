"""Funções puras de combinatória de cestas (2^k - 2, LOO) e vetorização X/Y.

Compartilhadas entre o construtor de datasets (`src/data/dataset_builder.py`) e o
serviço de inferência em produção (`src/serving/inference.py`), garantindo zero
training-serving skew.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

from src.config import FAIXAS_HORARIAS
from src.utils.temporal import (
    classificar_faixa_horaria,
    codificar_hora_ciclica,
    eh_fim_de_semana,
)

# Layout fixo do vetor de entrada genérico X (dim = 99)
# [0:34]   x_cart_sku_bin (34)
# [34:68]  x_cart_sku_qty (34)  -> log1p(qtd_pacotes)
# [68:88]  x_cart_base_bin (20)
# [88:93]  x_cart_stats (5)     -> escalonado com StandardScaler ajustado só no treino
# [93:99]  x_time (6)           -> hora_sin, hora_cos, 3 faixas horárias, fim_de_semana
CART_STATS_SLICE = slice(88, 93)
INPUT_DIM_GENERICO = 99


def gerar_subconjuntos_proprios(
    itens_cesta: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Gera todos os 2^k - 2 subconjuntos próprios não vazios (entrada, alvo).

    Para k=2 -> 2; k=3 -> 6; k=4 -> 14; k=5 -> 30; k=6 -> 62.
    """
    k = len(itens_cesta)
    if k < 2:
        return []

    pares: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    indices = list(range(k))
    for r in range(1, k):
        for comb in combinations(indices, r):
            comb_set = set(comb)
            entrada = [itens_cesta[i] for i in comb]
            alvo = [itens_cesta[i] for i in indices if i not in comb_set]
            pares.append((entrada, alvo))
    return pares


def gerar_leave_one_out(
    itens_cesta: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Gera exatamente k amostras Leave-One-Out: (B \\ {i}, {i}) para validação/teste."""
    k = len(itens_cesta)
    if k < 2:
        return []
    return [
        ([itens_cesta[j] for j in range(k) if j != i], [itens_cesta[i]])
        for i in range(k)
    ]


def construir_vetor_entrada_raw(
    cart_items: list[dict[str, Any]],
    hora: int,
    dia_semana: int,
    catalogo: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrói o vetor genérico bruto X (99 floats) + máscaras (mask_sku, mask_base)."""
    n_skus = int(catalogo["n_skus"])
    n_bases = int(catalogo["n_bases"])
    sku_by_id = {s["sku_id"]: s for s in catalogo["skus"]}
    sku_by_norm = {s["sku_norm"]: s for s in catalogo["skus"]}

    x_sku_bin = np.zeros(n_skus, dtype=np.float32)
    x_sku_qty = np.zeros(n_skus, dtype=np.float32)
    x_base_bin = np.zeros(n_bases, dtype=np.float32)
    mask_sku = np.zeros(n_skus, dtype=np.float32)
    mask_base = np.zeros(n_bases, dtype=np.float32)

    qtd_pacotes_acum: dict[int, int] = {}
    for item in cart_items:
        if "sku_id" in item:
            info = sku_by_id[int(item["sku_id"])]
        else:
            info = sku_by_norm[str(item["sku_norm"])]
        sid = int(info["sku_id"])
        qp = max(1, int(item.get("qtd_pacotes", 1)))
        qtd_pacotes_acum[sid] = qtd_pacotes_acum.get(sid, 0) + qp

    total_pacotes = 0
    total_unidades = 0
    valor_parcial = 0.0
    tem_pacote = 0.0

    for sid, qp in qtd_pacotes_acum.items():
        info = sku_by_id[sid]
        bid = int(info["base_id"])
        mult = int(info["multiplicador"])
        preco_pkg = float(info["preco_pacote_ref"])

        x_sku_bin[sid] = 1.0
        x_sku_qty[sid] = float(math.log1p(qp))
        x_base_bin[bid] = 1.0
        mask_sku[sid] = 1.0
        mask_base[bid] = 1.0

        total_pacotes += qp
        total_unidades += qp * mult
        valor_parcial += qp * preco_pkg
        if mult > 1:
            tem_pacote = 1.0

    x_cart_stats = np.array(
        [
            float(len(qtd_pacotes_acum)),
            float(math.log1p(total_pacotes)),
            float(math.log1p(total_unidades)),
            float(math.log1p(valor_parcial)),
            tem_pacote,
        ],
        dtype=np.float32,
    )

    h_sin, h_cos = codificar_hora_ciclica(hora)
    faixa = classificar_faixa_horaria(hora)
    faixa_oh = [1.0 if faixa == f else 0.0 for f in FAIXAS_HORARIAS]
    fds = 1.0 if eh_fim_de_semana(dia_semana) else 0.0
    x_time = np.array([h_sin, h_cos, *faixa_oh, fds], dtype=np.float32)

    x_vec = np.concatenate([x_sku_bin, x_sku_qty, x_base_bin, x_cart_stats, x_time])
    assert x_vec.shape == (INPUT_DIM_GENERICO,), (
        f"Esperado ({INPUT_DIM_GENERICO},), obtido {x_vec.shape}"
    )
    return x_vec, mask_sku, mask_base


def construir_alvos_multitask(
    target_items: list[dict[str, Any]],
    catalogo: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrói os vetores alvo (y_base, y_sku, y_qty) para os itens mascarados."""
    n_skus = int(catalogo["n_skus"])
    n_bases = int(catalogo["n_bases"])
    sku_by_id = {s["sku_id"]: s for s in catalogo["skus"]}

    y_base = np.zeros(n_bases, dtype=np.float32)
    y_sku = np.zeros(n_skus, dtype=np.float32)
    y_qty = np.zeros(n_skus, dtype=np.float32)

    for item in target_items:
        sid = int(item["sku_id"])
        bid = int(sku_by_id[sid]["base_id"])
        qp = max(1, int(item["qtd_pacotes"]))
        y_base[bid] = 1.0
        y_sku[sid] = 1.0
        y_qty[sid] = float(math.log1p(qp))

    return y_base, y_sku, y_qty
