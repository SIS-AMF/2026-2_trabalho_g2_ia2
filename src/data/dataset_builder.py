"""Etapa 4 — Orquestração do Split 80/20 Anti-Leakage e Construção dos Tensores.

Utiliza as funções puras de `src.utils.baskets` para dividir pedidos por
`pedido_id` (80% Desenvolvimento / 20% Teste Hold-Out), expandir cestas via
2^k - 2 no treino, ajustar `StandardScaler` apenas no treino e persistir em
`data/processed/splits/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATALOG_METADATA_PATH,
    PEDIDOS_CESTAS_PATH,
    SEED,
    SPLITS_DIR,
    TEST_SIZE,
    VAL_SIZE_FROM_DEV,
)
from src.utils.baskets import (
    CART_STATS_SLICE,
    INPUT_DIM_GENERICO,
    construir_alvos_multitask,
    construir_vetor_entrada_raw,
    gerar_leave_one_out,
    gerar_subconjuntos_proprios,
)


def _agrupar_pedidos_em_dicionario(df_varejo: pd.DataFrame) -> dict[str, dict[str, Any]]:
    pedidos: dict[str, dict[str, Any]] = {}
    for pid, grp in df_varejo.groupby("pedido_id", sort=True):
        pid_str = str(pid)
        itens = [
            {
                "sku_id": int(r.sku_id),
                "base_id": int(r.base_id),
                "sku_norm": str(r.produto_com_multiplicador),
                "qtd_pacotes": int(r.qtd_pacotes),
            }
            for r in grp.itertuples(index=False)
        ]
        pedidos[pid_str] = {
            "pedido_id": pid_str,
            "hora": int(grp["hora"].iloc[0]),
            "dia_semana": int(grp["dia_semana"].iloc[0]),
            "evento_nome": str(grp["evento_nome"].iloc[0]),
            "n_skus": len(itens),
            "itens": itens,
        }
    return pedidos


def dividir_pedidos_80_20(
    pedidos_dict: dict[str, dict[str, Any]],
    test_size: float = TEST_SIZE,
    val_size_from_dev: float = VAL_SIZE_FROM_DEV,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Split estrito 80/20 por pedido_id + sub-split interno treino/validação nos 80%."""
    pids = sorted(pedidos_dict.keys())
    estratos = [
        0 if pedidos_dict[p]["n_skus"] == 1 else (1 if pedidos_dict[p]["n_skus"] == 2 else 2)
        for p in pids
    ]

    dev_ids, test_ids = train_test_split(
        pids,
        test_size=test_size,
        random_state=seed,
        stratify=estratos,
    )

    dev_estratos = [
        0 if pedidos_dict[p]["n_skus"] == 1 else (1 if pedidos_dict[p]["n_skus"] == 2 else 2)
        for p in dev_ids
    ]
    train_ids, val_ids = train_test_split(
        dev_ids,
        test_size=val_size_from_dev,
        random_state=seed,
        stratify=dev_estratos,
    )

    assert set(dev_ids).isdisjoint(set(test_ids)), "Vazamento entre dev e test!"
    assert set(train_ids).isdisjoint(set(val_ids)), "Vazamento entre train e val!"
    assert set(train_ids).isdisjoint(set(test_ids)), "Vazamento entre train e test!"
    assert set(val_ids).isdisjoint(set(test_ids)), "Vazamento entre val e test!"

    return {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "dev_ids": sorted(dev_ids),
        "test_ids": sorted(test_ids),
    }


def _montar_arrays_split(
    pedido_ids: list[str],
    pedidos_dict: dict[str, dict[str, Any]],
    catalogo: dict[str, Any],
    modo: str,
    incluir_cold_start: bool = False,
    cold_start_ratio: float = 0.12,
    seed: int = SEED,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Gera arrays (X, y_base, y_sku, y_qty, mask_sku, mask_base) + metadados."""
    X_list: list[np.ndarray] = []
    yb_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []
    yq_list: list[np.ndarray] = []
    ms_list: list[np.ndarray] = []
    mb_list: list[np.ndarray] = []
    meta_list: list[dict[str, Any]] = []

    for pid in pedido_ids:
        ped = pedidos_dict[pid]
        itens = ped["itens"]
        if len(itens) < 2:
            continue
        pares = (
            gerar_subconjuntos_proprios(itens)
            if modo == "subsets"
            else gerar_leave_one_out(itens)
        )
        for entrada, alvo in pares:
            x_vec, m_sku, m_base = construir_vetor_entrada_raw(
                cart_items=entrada,
                hora=ped["hora"],
                dia_semana=ped["dia_semana"],
                catalogo=catalogo,
            )
            y_base, y_sku, y_qty = construir_alvos_multitask(alvo, catalogo)
            assert np.all((x_vec[: catalogo["n_skus"]] * y_sku) == 0.0)

            X_list.append(x_vec)
            yb_list.append(y_base)
            ys_list.append(y_sku)
            yq_list.append(y_qty)
            ms_list.append(m_sku)
            mb_list.append(m_base)
            meta_list.append(
                {
                    "pedido_id": pid,
                    "hora": ped["hora"],
                    "dia_semana": ped["dia_semana"],
                    "is_cold_start": False,
                    "input_skus": [it["sku_id"] for it in entrada],
                    "input_qtys": [it["qtd_pacotes"] for it in entrada],
                    "target_skus": [it["sku_id"] for it in alvo],
                    "target_qtys": [it["qtd_pacotes"] for it in alvo],
                }
            )

    if incluir_cold_start and len(X_list) > 0:
        rng = np.random.default_rng(seed)
        n_cold = max(1, int(round(len(X_list) * cold_start_ratio)))
        candidatos_cs = [pid for pid in pedido_ids if len(pedidos_dict[pid]["itens"]) >= 1]
        escolhidos = rng.choice(
            candidatos_cs,
            size=min(n_cold, len(candidatos_cs)),
            replace=False,
        )
        for pid in escolhidos:
            ped = pedidos_dict[str(pid)]
            x_vec, m_sku, m_base = construir_vetor_entrada_raw(
                cart_items=[],
                hora=ped["hora"],
                dia_semana=ped["dia_semana"],
                catalogo=catalogo,
            )
            y_base, y_sku, y_qty = construir_alvos_multitask(ped["itens"], catalogo)
            X_list.append(x_vec)
            yb_list.append(y_base)
            ys_list.append(y_sku)
            yq_list.append(y_qty)
            ms_list.append(m_sku)
            mb_list.append(m_base)
            meta_list.append(
                {
                    "pedido_id": str(pid),
                    "hora": ped["hora"],
                    "dia_semana": ped["dia_semana"],
                    "is_cold_start": True,
                    "input_skus": [],
                    "input_qtys": [],
                    "target_skus": [it["sku_id"] for it in ped["itens"]],
                    "target_qtys": [it["qtd_pacotes"] for it in ped["itens"]],
                }
            )

    arrays = {
        "X": np.stack(X_list).astype(np.float32),
        "y_base": np.stack(yb_list).astype(np.float32),
        "y_sku": np.stack(ys_list).astype(np.float32),
        "y_qty": np.stack(yq_list).astype(np.float32),
        "mask_sku": np.stack(ms_list).astype(np.float32),
        "mask_base": np.stack(mb_list).astype(np.float32),
    }
    return arrays, meta_list


def construir_datasets_completos(
    df_cestas: pd.DataFrame,
    catalogo: dict[str, Any],
    seed: int = SEED,
) -> dict[str, Any]:
    """Executa a Etapa 4 completa sem data leakage."""
    df_varejo = df_cestas[df_cestas["is_pedido_varejo"]].copy()
    pedidos_dict = _agrupar_pedidos_em_dicionario(df_varejo)
    splits = dividir_pedidos_80_20(pedidos_dict, seed=seed)

    train_arr, train_meta = _montar_arrays_split(
        splits["train_ids"], pedidos_dict, catalogo, modo="subsets", incluir_cold_start=True, seed=seed
    )
    val_arr, val_meta = _montar_arrays_split(
        splits["val_ids"], pedidos_dict, catalogo, modo="loo", incluir_cold_start=False, seed=seed
    )
    test_arr, test_meta = _montar_arrays_split(
        splits["test_ids"], pedidos_dict, catalogo, modo="loo", incluir_cold_start=False, seed=seed
    )

    scaler = StandardScaler()
    train_arr["X"][:, CART_STATS_SLICE] = scaler.fit_transform(
        train_arr["X"][:, CART_STATS_SLICE]
    ).astype(np.float32)
    val_arr["X"][:, CART_STATS_SLICE] = scaler.transform(
        val_arr["X"][:, CART_STATS_SLICE]
    ).astype(np.float32)
    test_arr["X"][:, CART_STATS_SLICE] = scaler.transform(
        test_arr["X"][:, CART_STATS_SLICE]
    ).astype(np.float32)

    multi_counts = {
        nome: sum(1 for pid in ids if pedidos_dict[pid]["n_skus"] >= 2)
        for nome, ids in splits.items()
    }
    summary = {
        "seed": seed,
        "input_dim": INPUT_DIM_GENERICO,
        "pedidos_por_split": {k: len(v) for k, v in splits.items()},
        "pedidos_multi_sku_por_split": multi_counts,
        "amostras_geradas": {
            "train_total": int(len(train_arr["X"])),
            "train_cross_sell_subsets": int(sum(1 for m in train_meta if not m["is_cold_start"])),
            "train_cold_start": int(sum(1 for m in train_meta if m["is_cold_start"])),
            "val_loo": int(len(val_arr["X"])),
            "test_loo": int(len(test_arr["X"])),
        },
    }

    return {
        "splits": splits,
        "train": train_arr,
        "val": val_arr,
        "test": test_arr,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "test_meta": test_meta,
        "scaler": scaler,
        "summary": summary,
    }


def salvar_splits(
    bundle: dict[str, Any],
    splits_dir: Path = SPLITS_DIR,
) -> None:
    """Persiste índices, tensores `.npz`, metadados e `scaler.joblib` em `data/processed/splits/`."""
    splits_dir.mkdir(parents=True, exist_ok=True)

    with (splits_dir / "split_ids.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {"splits": bundle["splits"], "summary": bundle["summary"]},
            fp,
            ensure_ascii=False,
            indent=2,
        )

    with (splits_dir / "split_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "train_meta": bundle["train_meta"],
                "val_meta": bundle["val_meta"],
                "test_meta": bundle["test_meta"],
            },
            fp,
            ensure_ascii=False,
        )

    npz_payload: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        for key, arr in bundle[split_name].items():
            npz_payload[f"{split_name}_{key}"] = arr
    np.savez_compressed(splits_dir / "datasets.npz", **npz_payload)

    joblib.dump(bundle["scaler"], splits_dir / "scaler.joblib")


def carregar_splits(splits_dir: Path = SPLITS_DIR) -> dict[str, Any]:
    """Carrega os artefatos persistidos da Etapa 4."""
    with (splits_dir / "split_ids.json").open("r", encoding="utf-8") as fp:
        ids_doc = json.load(fp)
    with (splits_dir / "split_meta.json").open("r", encoding="utf-8") as fp:
        meta_doc = json.load(fp)

    npz = np.load(splits_dir / "datasets.npz")
    result: dict[str, Any] = {
        "splits": ids_doc["splits"],
        "summary": ids_doc["summary"],
        "train_meta": meta_doc["train_meta"],
        "val_meta": meta_doc["val_meta"],
        "test_meta": meta_doc["test_meta"],
        "scaler": joblib.load(splits_dir / "scaler.joblib"),
    }
    for split_name in ("train", "val", "test"):
        result[split_name] = {
            key: npz[f"{split_name}_{key}"]
            for key in ("X", "y_base", "y_sku", "y_qty", "mask_sku", "mask_base")
        }
    return result


def main() -> None:
    """Ponto de entrada CLI (`python -m src.data.dataset_builder`)."""
    df_cestas = pd.read_parquet(PEDIDOS_CESTAS_PATH)
    with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as fp:
        catalogo = json.load(fp)
    bundle = construir_datasets_completos(df_cestas, catalogo)
    salvar_splits(bundle)

    s = bundle["summary"]
    print("=== ETAPA 4 — DATASET E SPLIT 80/20 CONCLUÍDOS ===")
    print(f"Pedidos por split: {s['pedidos_por_split']}")
    print(f"Pedidos multi-SKU por split: {s['pedidos_multi_sku_por_split']}")
    print(f"Amostras geradas: {s['amostras_geradas']}")
    print(f"Dimensão do vetor X (genérico): {s['input_dim']}")
    print(f"Salvo em: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
