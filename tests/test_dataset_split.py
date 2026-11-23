"""Testes unitários e anti-data-leakage da Etapa 4 (Split 80/20 e Subconjuntos 2^k - 2)."""

from __future__ import annotations

import json

import numpy as np

from src.config import CATALOG_METADATA_PATH, SPLITS_DIR
from src.data.dataset_builder import carregar_splits
from src.utils.baskets import (
    INPUT_DIM_GENERICO,
    construir_vetor_entrada_raw,
    gerar_leave_one_out,
    gerar_subconjuntos_proprios,
)


class TestExpansaoSubconjuntosEVetores:
    def test_gerar_subconjuntos_proprios_exato(self) -> None:
        for k, esperado in [(1, 0), (2, 2), (3, 6), (4, 14), (5, 30)]:
            itens = [{"sku_id": i, "qtd_pacotes": 1} for i in range(k)]
            pares = gerar_subconjuntos_proprios(itens)
            assert len(pares) == esperado
            # Verificar unicidade dos conjuntos de entrada e ausência de sobreposição
            assinaturas = set()
            for entrada, alvo in pares:
                in_ids = frozenset(it["sku_id"] for it in entrada)
                tgt_ids = frozenset(it["sku_id"] for it in alvo)
                assert len(in_ids) >= 1 and len(tgt_ids) >= 1
                assert in_ids.isdisjoint(tgt_ids)
                assert len(in_ids | tgt_ids) == k
                assinaturas.add(in_ids)
            assert len(assinaturas) == esperado

    def test_gerar_leave_one_out(self) -> None:
        itens = [{"sku_id": i, "qtd_pacotes": 1} for i in range(4)]
        pares = gerar_leave_one_out(itens)
        assert len(pares) == 4
        for entrada, alvo in pares:
            assert len(entrada) == 3
            assert len(alvo) == 1
            assert alvo[0]["sku_id"] not in {it["sku_id"] for it in entrada}

    def test_vetor_entrada_generico_dim_99(self) -> None:
        with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as fp:
            catalogo = json.load(fp)
        x, m_sku, m_base = construir_vetor_entrada_raw(
            cart_items=[{"sku_id": 0, "qtd_pacotes": 2}],
            hora=15,
            dia_semana=5,
            catalogo=catalogo,
        )
        assert x.shape == (INPUT_DIM_GENERICO,)
        assert INPUT_DIM_GENERICO == 99
        assert x[0] == 1.0
        assert m_sku[0] == 1.0
        assert m_sku.sum() == 1.0
        assert m_base.sum() == 1.0
        assert not np.isnan(x).any()


class TestSplitAntiLeakagePersistido:
    def test_split_80_20_sem_vazamento_por_pedido_id(self) -> None:
        assert (SPLITS_DIR / "split_ids.json").exists(), "Execute `python -m src.data.dataset_builder`."
        bundle = carregar_splits(SPLITS_DIR)
        splits = bundle["splits"]

        train_set = set(splits["train_ids"])
        val_set = set(splits["val_ids"])
        dev_set = set(splits["dev_ids"])
        test_set = set(splits["test_ids"])

        # Interseções estritamente vazias
        assert dev_set.isdisjoint(test_set)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert train_set | val_set == dev_set

        # Proporção 80/20 verificada
        total = len(dev_set) + len(test_set)
        assert abs(len(test_set) / total - 0.20) < 0.01

        # Verificar pedido_ids presentes nos metadados de cada split
        train_meta_pids = {m["pedido_id"] for m in bundle["train_meta"]}
        val_meta_pids = {m["pedido_id"] for m in bundle["val_meta"]}
        test_meta_pids = {m["pedido_id"] for m in bundle["test_meta"]}
        assert train_meta_pids.isdisjoint(val_meta_pids)
        assert train_meta_pids.isdisjoint(test_meta_pids)
        assert val_meta_pids.isdisjoint(test_meta_pids)

        # Verificar dimensões e ausência de sobreposição entre carrinho e alvo
        for split_name in ("train", "val", "test"):
            arr = bundle[split_name]
            assert arr["X"].shape[1] == 99
            assert arr["y_base"].shape[1] == 20
            assert arr["y_sku"].shape[1] == 34
            assert arr["y_qty"].shape[1] == 34
            assert not np.isnan(arr["X"]).any()
            assert ((arr["X"][:, :34] * arr["y_sku"]) == 0.0).all()
