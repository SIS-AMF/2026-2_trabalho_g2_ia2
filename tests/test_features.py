"""Testes unitários e de integração da Etapa 2 (Engenharia de Features e Catálogo)."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.config import (
    CATALOG_METADATA_PATH,
    ITENS_LIMPOS_PATH,
    PEDIDOS_CESTAS_CSV_PATH,
    PEDIDOS_CESTAS_PATH,
)
from src.data.features import executar_engenharia_features
from src.utils.temporal import (
    calcular_economia_unitaria_pct,
    classificar_faixa_horaria,
    codificar_hora_ciclica,
    eh_fim_de_semana,
    montar_display_name,
)


class TestFuncoesPurasFeatures:
    def test_calcular_economia_unitaria_pct(self) -> None:
        # Rapadura Assada: avulso R$ 5,00 vs Pacote x6 R$ 4,17 -> ~16.6%
        assert calcular_economia_unitaria_pct(4.17, 5.00) == pytest.approx(16.6, abs=0.1)
        assert calcular_economia_unitaria_pct(5.00, 5.00) == 0.0
        assert calcular_economia_unitaria_pct(6.00, 5.00) == 0.0
        assert calcular_economia_unitaria_pct(5.00, 0.0) == 0.0

    def test_codificar_hora_ciclica(self) -> None:
        s0, c0 = codificar_hora_ciclica(0)
        assert (s0, c0) == pytest.approx((0.0, 1.0), abs=1e-5)
        s6, c6 = codificar_hora_ciclica(6)
        assert (s6, c6) == pytest.approx((1.0, 0.0), abs=1e-5)
        s12, c12 = codificar_hora_ciclica(12)
        assert (s12, c12) == pytest.approx((0.0, -1.0), abs=1e-5)
        # Norma unitária para qualquer hora
        for h in range(24):
            sh, ch = codificar_hora_ciclica(h)
            assert math.hypot(sh, ch) == pytest.approx(1.0, abs=1e-5)

    def test_classificar_faixa_horaria(self) -> None:
        assert classificar_faixa_horaria(8) == "Manhã"
        assert classificar_faixa_horaria(11) == "Manhã"
        assert classificar_faixa_horaria(12) == "Tarde"
        assert classificar_faixa_horaria(17) == "Tarde"
        assert classificar_faixa_horaria(18) == "Noite"
        assert classificar_faixa_horaria(23) == "Noite"

    def test_eh_fim_de_semana(self) -> None:
        for d in range(5):
            assert eh_fim_de_semana(d) is False
        assert eh_fim_de_semana(5) is True
        assert eh_fim_de_semana(6) is True

    def test_montar_display_name(self) -> None:
        assert montar_display_name("Cuca Alemã", "Unidade", 1) == "Cuca Alemã (Unidade)"
        assert (
            montar_display_name("Rapadura de Melado", "Pacote", 3)
            == "Rapadura de Melado (Pacote x3)"
        )


class TestIntegracaoFeaturesECatalogo:
    def test_executar_engenharia_features_e_invariantes(self) -> None:
        df_itens = pd.read_parquet(ITENS_LIMPOS_PATH)
        df_cestas, catalogo = executar_engenharia_features(df_itens)

        assert catalogo["n_bases"] == 20
        assert catalogo["n_skus"] == 34
        assert len(catalogo["bases"]) == 20
        assert len(catalogo["skus"]) == 34

        # IDs contíguos 0..N-1
        assert [b["base_id"] for b in catalogo["bases"]] == list(range(20))
        assert [s["sku_id"] for s in catalogo["skus"]] == list(range(34))

        # Unicidade de (pedido_id, produto_com_multiplicador)
        assert df_cestas.duplicated(subset=["pedido_id", "produto_com_multiplicador"]).sum() == 0

        # Invariantes obrigatórias
        assert (
            df_cestas["qtd_pacotes"] * df_cestas["multiplicador"]
            == df_cestas["qtd_unidades_base"]
        ).all()
        assert (
            (df_cestas["preco_unitario"] * df_cestas["multiplicador"] - df_cestas["preco_pacote"])
            .abs()
            .max()
            < 0.02
        )
        assert df_cestas.isna().sum().sum() == 0

        # Colunas temporais válidas
        assert df_cestas["hora"].between(0, 23).all()
        assert set(df_cestas["faixa_horaria"].unique()).issubset({"Manhã", "Tarde", "Noite"})

    def test_artefatos_persistidos_stage_2(self) -> None:
        assert PEDIDOS_CESTAS_PATH.exists(), "Execute `python -m src.data.features` antes dos testes."
        assert PEDIDOS_CESTAS_CSV_PATH.exists(), "Espelho CSV de pedidos_cestas ausente."
        assert CATALOG_METADATA_PATH.exists()

        df_cestas = pd.read_parquet(PEDIDOS_CESTAS_PATH)
        assert df_cestas.isna().sum().sum() == 0
        assert df_cestas["produto_base"].nunique() == 20
        assert df_cestas["produto_com_multiplicador"].nunique() == 34

        df_cestas_csv = pd.read_csv(PEDIDOS_CESTAS_CSV_PATH)
        assert list(df_cestas_csv.columns) == list(df_cestas.columns)
        assert len(df_cestas_csv) == len(df_cestas)
        assert df_cestas_csv.isna().sum().sum() == 0

        with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as fp:
            catalogo = json.load(fp)
        assert catalogo["n_bases"] == 20
        assert catalogo["n_skus"] == 34
        for sku in catalogo["skus"]:
            assert sku["multiplicador"] >= 1
            assert sku["preco_unitario_ref"] > 0
            assert sku["preco_pacote_ref"] > 0
