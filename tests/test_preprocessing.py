"""Testes unitários das funções puras e integração completa da Etapa 1."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import (
    COLUNA_TIPO_DESCONTO,
    COLUNAS_ITENS_LIMPOS,
    COLUNAS_VENDEDOR_INVALIDAS,
    DATASET_RAW_PATH,
    ITENS_LIMPOS_PATH,
    MAPA_EMBALAGEM_SEM_MULTIPLICADOR,
    MAPA_EVENTOS,
    MAPA_SINONIMOS_PRODUTOS,
    PREPROCESSING_AUDIT_PATH,
    TIMEZONE,
)
from src.data.preprocessing import executar_preprocessamento
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


# =============================================================================
# TESTES UNITÁRIOS DAS FUNÇÕES PURAS (SEM DEPENDÊNCIA DE DATAFRAME)
# =============================================================================


class TestValidarDescontoEmMoeda:
    def test_desconto_zero_em_moeda(self) -> None:
        valido, desc, status = validar_desconto_em_moeda(20.0, 0.0, 20.0)
        assert valido is True
        assert desc == 0.0
        assert status == "ja_em_moeda"

    def test_caso_real_expobento_tipo_pct_ja_em_reais(self) -> None:
        # Pedido ca605a42-...: subtotal=6, desconto=0.12 (marcado '%'), total=5.88
        valido, desc, status = validar_desconto_em_moeda(6.0, 0.12, 5.88)
        assert valido is True
        assert desc == pytest.approx(0.12)
        assert status == "ja_em_moeda"

    def test_conversao_automatica_quando_desconto_vier_em_percentual(self) -> None:
        # Subtotal=100, desconto=10 (%) -> total=90 -> R$ 10.00
        valido, desc, status = validar_desconto_em_moeda(100.0, 10.0, 90.0)
        # Aqui 100 - 10 == 90 coincide em moeda; testemos subtotal=200, 10% -> total=180
        valido2, desc2, status2 = validar_desconto_em_moeda(200.0, 10.0, 180.0)
        assert valido is True
        assert desc == pytest.approx(10.0)
        assert status == "ja_em_moeda"
        assert valido2 is True
        assert desc2 == pytest.approx(20.0)
        assert status2 == "convertido_de_percentual"

    def test_desconto_inconsistente_retorna_falso(self) -> None:
        valido, _, status = validar_desconto_em_moeda(50.0, 7.0, 30.0)
        assert valido is False
        assert status == "inconsistente"


class TestFuncoesSaneamentoEProtecaoProdutos:
    def test_item_quantidade_valida(self) -> None:
        assert item_quantidade_valida(1) is True
        assert item_quantidade_valida(6) is True
        assert item_quantidade_valida(0) is False
        assert item_quantidade_valida(-2) is False

    def test_classificar_pedido_gratuito(self) -> None:
        assert classificar_pedido_gratuito(0.0) is True
        assert classificar_pedido_gratuito(-1.0) is True
        assert classificar_pedido_gratuito(0.01) is False

    def test_normalizar_texto_produto(self) -> None:
        assert normalizar_texto_produto("  Bolacha de manteiga  (Un) ") == "Bolacha de manteiga (Un)"
        assert normalizar_texto_produto("Amendoim salgado e temperado ") == "Amendoim salgado e temperado"

    def test_extrair_produto_base_sinonimo_confirmado_e_protecao_distintos(self) -> None:
        # Único sinônimo confirmado (troca de embalagem)
        assert (
            extrair_produto_base("Amendoim salgado e temperado", MAPA_SINONIMOS_PRODUTOS)
            == "Amendoim Temperado"
        )
        assert (
            extrair_produto_base("Amendoim Temperado", MAPA_SINONIMOS_PRODUTOS)
            == "Amendoim Temperado"
        )
        # Produtos distintos NÃO podem ser fundidos
        assert (
            extrair_produto_base("Amendoim Temperado 100g", MAPA_SINONIMOS_PRODUTOS)
            == "Amendoim Temperado 100g"
        )
        assert (
            extrair_produto_base("Rapadura de Melado Grão Moído (Pacote x3)", MAPA_SINONIMOS_PRODUTOS)
            == "Rapadura de Melado Grão Moído"
        )
        assert (
            extrair_produto_base("Rapadura de Melado (Pacote x3)", MAPA_SINONIMOS_PRODUTOS)
            == "Rapadura de Melado"
        )
        assert (
            extrair_produto_base("Alfajor Preto (Pacote x4)", MAPA_SINONIMOS_PRODUTOS)
            == "Alfajor Preto"
        )
        assert (
            extrair_produto_base("Alfajor Branco (Pacote x4)", MAPA_SINONIMOS_PRODUTOS)
            == "Alfajor Branco"
        )

    def test_normalizar_evento(self) -> None:
        assert normalizar_evento("1 Moranguinho", MAPA_EVENTOS) == ("Festa do Moranguinho", "Caixa 1")
        assert normalizar_evento("2-Moranguinho", MAPA_EVENTOS) == ("Festa do Moranguinho", "Caixa 2")
        assert normalizar_evento("Expointer_2026", MAPA_EVENTOS) == ("Expointer_2026", "Padrão")

    def test_normalizar_metodo_pagamento(self) -> None:
        assert normalizar_metodo_pagamento("Pix") == ("PIX", "PIX")
        assert normalizar_metodo_pagamento("PIX") == ("PIX", "PIX")
        assert normalizar_metodo_pagamento("Cartão de Débito") == ("Cartão de Débito", "Cartão")
        assert normalizar_metodo_pagamento("Cartão de Crédito") == ("Cartão de Crédito", "Cartão")
        assert normalizar_metodo_pagamento("Dinheiro") == ("Dinheiro", "Dinheiro")


class TestEmbalagemMultiplicadorEReconciliacao:
    @pytest.mark.parametrize(
        "nome_limpo,base,preco,esperado",
        [
            ("Rapadura Assada (Pacote x6)", "Rapadura Assada", 4.166667, ("Pacote", 6)),
            ("Alfajor Preto (Caixa x16)", "Alfajor Preto", 5.0, ("Caixa", 16)),
            ("Alfajor Preto (Bandeja x5)", "Alfajor Preto", 4.0, ("Bandeja", 5)),
            ("Alfajor Preto (Display x1)", "Alfajor Preto", 5.0, ("Display", 1)),
            ("Alfajor Preto (Display x16)", "Alfajor Preto", 5.0, ("Display", 16)),
            # Exceção verificada: Pacote x1 a R$ 20 -> Pacote x3
            ("Rapadura de Melado Grão Moído (Pacote x1)", "Rapadura de Melado Grão Moído", 20.0, ("Pacote", 3)),
            # Todos os 6 casos da 1 Moranguinho sem xN
            ("Cuca Alemã (Fardo)", "Cuca Alemã", 60.0, ("Fardo", 3)),
            ("Rapadura Assada (Pacote)", "Rapadura Assada", 25.0, ("Pacote", 6)),
            ("Rapadura de Melado (Pacote)", "Rapadura de Melado", 20.0, ("Pacote", 3)),
            ("Alfajor Preto (Pacote)", "Alfajor Preto", 20.0, ("Pacote", 4)),
            ("Cuca Enrolada de Amendoim (Pacote)", "Cuca Enrolada de Amendoim", 25.0, ("Pacote", 2)),
            ("Cuca Enrolada de Chocolate (Pacote)", "Cuca Enrolada de Chocolate", 30.0, ("Pacote", 2)),
            # Unitários
            ("Cuca Alemã (Un)", "Cuca Alemã", 25.0, ("Unidade", 1)),
            ("Pé-de-Moleque", "Pé-de-Moleque", 5.0, ("Unidade", 1)),
        ],
    )
    def test_extrair_embalagem_e_multiplicador(
        self, nome_limpo: str, base: str, preco: float, esperado: tuple[str, int]
    ) -> None:
        assert (
            extrair_embalagem_e_multiplicador(
                nome_limpo, base, preco, MAPA_EMBALAGEM_SEM_MULTIPLICADOR
            )
            == esperado
        )

    def test_embalagem_sem_mult_desconhecida_falha_explicitamente(self) -> None:
        with pytest.raises(ValueError, match="não mapeada"):
            extrair_embalagem_e_multiplicador(
                "Produto Novo (Pacote)", "Produto Novo", 10.0, MAPA_EMBALAGEM_SEM_MULTIPLICADOR
            )

    def test_reconciliacao_todos_os_quatro_casos(self) -> None:
        # Caso A: unitário (mult == 1)
        assert reconciliar_quantidade_e_preco(2, 15.0, 1, False, False) == (2, 2, 15.0, 15.0)
        # Caso B: ExpoDireto divergente (qtd=12 un, preco=25 pacote x6 -> 2 pacotes, unit=4.1667)
        qb, qp, pu, pp = reconciliar_quantidade_e_preco(12, 25.0, 6, True, False)
        assert (qb, qp, pp) == (12, 2, 25.0)
        assert pu == pytest.approx(25.0 / 6.0)
        # Caso C: registro em pacotes (1 Moranguinho ou ExpoDireto 11 linhas: qtd=2 pacotes, preco=20 pacote x4)
        qb, qp, pu, pp = reconciliar_quantidade_e_preco(2, 20.0, 4, False, True)
        assert (qb, qp, pu, pp) == (8, 2, 5.0, 20.0)
        # Caso D: feira padrão (qtd=6 un, preco=6.666667 unit, mult=3 -> 2 pacotes, preco_pacote=20)
        qb, qp, pu, pp = reconciliar_quantidade_e_preco(6, 20.0 / 3.0, 3, False, False)
        assert (qb, qp) == (6, 2)
        assert pu == pytest.approx(20.0 / 3.0)
        assert pp == pytest.approx(20.0)

    def test_reconciliacao_rejeita_qtd_pacotes_fracionaria(self) -> None:
        with pytest.raises(ValueError, match="fracionária"):
            reconciliar_quantidade_e_preco(5, 4.166667, 6, False, False)

    def test_identificar_registro_em_pacotes(self) -> None:
        assert identificar_registro_em_pacotes("Alfajor Preto (Pacote x4)", "ExpoDireto", 4, False) is True
        assert identificar_registro_em_pacotes("Alfajor Preto (Pacote x4)", "ExpoDireto", 4, True) is False
        assert identificar_registro_em_pacotes("Cuca Alemã (Fardo)", "1 Moranguinho", 3, False) is True
        assert identificar_registro_em_pacotes("Rapadura Assada (Pacote x6)", "Expointer_2026", 6, False) is False

    def test_sku_valor_fechamento_e_classificacao(self) -> None:
        assert montar_sku_canonico("Rapadura Assada", 6) == "Rapadura Assada (x6)"
        assert calcular_valor_item(6, 25.0 / 6.0) == pytest.approx(25.0)
        assert validar_fechamento_pedido(25.0001, 25.0) is True
        assert validar_fechamento_pedido(25.05, 25.0) is False
        assert classificar_pedido_consolidado(250.0, 10, "Expointer_2026") is True
        assert classificar_pedido_consolidado(50.0, 45, "Expointer_2026") is True
        assert classificar_pedido_consolidado(50.0, 2, "Feira Polêsine") is True
        assert classificar_pedido_consolidado(50.0, 4, "Expointer_2026") is False
        assert classificar_pedido_varejo(False, False) is True
        assert classificar_pedido_varejo(True, False) is False
        assert classificar_pedido_varejo(False, True) is False


# =============================================================================
# TESTES DE INTEGRAÇÃO NO DATASET REAL E ARTEFATOS PERSISTIDOS
# =============================================================================


class TestIntegracaoDatasetReal:
    def test_executar_preprocessamento_no_csv_real(self) -> None:
        df_bruto = pd.read_csv(DATASET_RAW_PATH)
        df_limpo, audit = executar_preprocessamento(df_bruto)

        # Contagens exatas
        assert audit["entrada"]["linhas_brutas"] == 3815
        assert audit["entrada"]["pedidos_brutos"] == 3003
        assert len(df_limpo) == 3813
        assert df_limpo["pedido_id"].nunique() == 3003

        # Colunas descartadas não existem
        for col in (*COLUNAS_VENDEDOR_INVALIDAS, COLUNA_TIPO_DESCONTO):
            assert col not in df_limpo.columns

        # 0 erros de desconto e 0 erros contábeis pós-reconciliação
        assert audit["descontos"]["inconsistentes"] == 0
        assert audit["reconciliacao_contabil"]["pedidos_divergentes_antes"] == 139
        assert audit["reconciliacao_contabil"]["pedidos_divergentes_depois"] == 0
        assert audit["reconciliacao_contabil"]["max_erro_absoluto_reais"] < 0.01

        # 0 quantidades <= 0 e 0 violações das invariantes de pacote/preço
        assert (df_limpo["qtd_unidades_base"] <= 0).sum() == 0
        assert (df_limpo["qtd_pacotes"] <= 0).sum() == 0
        assert (df_limpo["qtd_pacotes"] * df_limpo["multiplicador"] == df_limpo["qtd_unidades_base"]).all()
        assert (df_limpo["preco_unitario"] * df_limpo["multiplicador"] - df_limpo["preco_pacote"]).abs().max() < 0.01

        # Fechamento por pedido em 100% dos 3.003 pedidos
        soma_por_pedido = df_limpo.groupby("pedido_id")["item_valor_total"].sum()
        subtotal_por_pedido = df_limpo.groupby("pedido_id")["pedido_subtotal"].first()
        assert (soma_por_pedido - subtotal_por_pedido).abs().max() < 0.01

        # Desconto em moeda em 100% dos 3.003 pedidos
        desconto_por_pedido = df_limpo.groupby("pedido_id")["pedido_valor_desconto"].first()
        total_por_pedido = df_limpo.groupby("pedido_id")["pedido_valor_total"].first()
        assert ((subtotal_por_pedido - desconto_por_pedido) - total_por_pedido).abs().max() < 0.01

        # Catálogo exato: 20 produtos base e 34 SKUs canônicos, sem fusão indevida
        produtos_base = set(df_limpo["produto_base"].unique())
        assert len(produtos_base) == 20
        assert df_limpo["produto_com_multiplicador"].nunique() == 34
        assert "Amendoim Temperado" in produtos_base
        assert "Amendoim Temperado 100g" in produtos_base
        assert "Amendoim salgado e temperado" not in produtos_base
        assert "Rapadura de Melado" in produtos_base
        assert "Rapadura de Melado Grão Moído" in produtos_base
        assert "Alfajor Preto" in produtos_base
        assert "Alfajor Branco" in produtos_base

        # Nenhum nulo em nenhuma coluna do schema final
        assert df_limpo.isna().sum().sum() == 0

    def test_artefatos_persistidos_parquet_e_audit_json(self) -> None:
        assert ITENS_LIMPOS_PATH.exists(), "Execute `python -m src.data.preprocessing` antes dos testes."
        assert PREPROCESSING_AUDIT_PATH.exists()

        df_parquet = pd.read_parquet(ITENS_LIMPOS_PATH)
        assert tuple(df_parquet.columns) == COLUNAS_ITENS_LIMPOS
        assert len(df_parquet) == 3813
        assert df_parquet.isna().sum().sum() == 0

        # Verificar tipagem forte preservada após round-trip Parquet
        assert str(df_parquet["pedido_id"].dtype) == "string"
        assert isinstance(df_parquet["pedido_data"].dtype, pd.DatetimeTZDtype)
        assert str(df_parquet["pedido_data"].dt.tz) == TIMEZONE
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
            assert isinstance(df_parquet[col_cat].dtype, pd.CategoricalDtype)
        for col_int in ("multiplicador", "qtd_unidades_base", "qtd_pacotes"):
            assert df_parquet[col_int].dtype == "int64"
        for col_float in (
            "pedido_subtotal",
            "pedido_valor_desconto",
            "pedido_valor_total",
            "preco_unitario",
            "preco_pacote",
            "item_valor_total",
        ):
            assert df_parquet[col_float].dtype == "float64"
        for col_bool in ("is_pedido_gratuito", "is_pedido_consolidado", "is_pedido_varejo"):
            assert df_parquet[col_bool].dtype == "bool"

        with PREPROCESSING_AUDIT_PATH.open("r", encoding="utf-8") as fp:
            audit = json.load(fp)
        assert audit["reconciliacao_contabil"]["pedidos_divergentes_depois"] == 0
        assert audit["descontos"]["inconsistentes"] == 0
        assert audit["catalogo_normalizado"]["total_produtos_base"] == 20
        assert audit["catalogo_normalizado"]["total_skus_canonicos"] == 34
