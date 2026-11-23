"""Testes das Etapas 5, 6 e 7: Rede Neural, Loss Mascarada, Exportação, Estoque e FastAPI."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app
from src.config import (
    CATALOG_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    PREPROCESSING_ARTIFACTS_PATH,
    RECOMMENDER_ARTIFACT_DIR,
    TRAINING_REPORT_PATH,
)
from src.modeling.losses import MultiTaskRecommendationLoss
from src.modeling.models import BasketDAE, FairMultiTaskRecommender
from src.serving.inference import RecommenderService


class TestArquiteturaELoss:
    def test_forward_pass_dimensoes_tres_cabecas(self) -> None:
        model = FairMultiTaskRecommender(in_dim=99, hidden_dims=(128, 64), n_bases=20, n_skus=34)
        model.eval()
        x = torch.randn(4, 99)
        lb, ls, pq = model(x)
        assert lb.shape == (4, 20)
        assert ls.shape == (4, 34)
        assert pq.shape == (4, 34)
        assert (pq >= 0.0).all()

    def test_loss_mascarada_ignora_itens_ja_no_carrinho_e_nao_positivos_na_qty(self) -> None:
        criterion = MultiTaskRecommendationLoss(w_base=1.0, w_sku=1.5, w_qty=0.5)
        lb = torch.zeros(2, 20, requires_grad=True)
        ls = torch.zeros(2, 34, requires_grad=True)
        pq = torch.ones(2, 34, requires_grad=True)

        yb = torch.zeros(2, 20)
        ys = torch.zeros(2, 34)
        yq = torch.zeros(2, 34)
        mb = torch.zeros(2, 20)
        ms = torch.zeros(2, 34)

        # Item 0 está no carrinho; item 5 é o alvo real
        ms[:, 0] = 1.0
        ys[:, 5] = 1.0
        yb[:, 2] = 1.0
        yq[:, 5] = np.log1p(2.0)

        loss, breakdown = criterion(lb, ls, pq, yb, ys, yq, mb, ms)
        loss.backward()

        assert np.isfinite(breakdown["loss_total"])
        # Gradiente na posição 0 do SKU (já no carrinho) deve ser EXATAMENTE zero
        assert float(ls.grad[:, 0].abs().max().item()) == 0.0
        # Gradiente da cabeça de quantidade deve ser não-nulo apenas na posição positiva 5
        assert float(pq.grad[:, 5].abs().min().item()) > 0.0
        assert float(pq.grad[:, 0].abs().max().item()) == 0.0
        assert float(pq.grad[:, 10].abs().max().item()) == 0.0

    def test_basket_dae_forward(self) -> None:
        dae = BasketDAE(n_skus=34, n_time=6)
        dae.train()
        out = dae(torch.zeros(4, 34), torch.zeros(4, 6))
        assert out.shape == (4, 34)


class TestExportacaoInferenciaEEstoque:
    def test_artefatos_versionados_existem(self) -> None:
        for p in (
            MODEL_WEIGHTS_PATH,
            PREPROCESSING_ARTIFACTS_PATH,
            CATALOG_METADATA_PATH,
            TRAINING_REPORT_PATH,
        ):
            assert p.exists(), f"Artefato ausente: {p}"

    def test_round_trip_exportado_reproduz_predicoes_identicas(self) -> None:
        service_a = RecommenderService.from_artifact(RECOMMENDER_ARTIFACT_DIR)
        service_b = RecommenderService.from_artifact(RECOMMENDER_ARTIFACT_DIR)

        cart = [{"sku_norm": "Alfajor Preto (x1)", "qtd_pacotes": 2}]
        recs_a = service_a.recommend(cart, hora=15, dia_semana=5, top_k=3)
        recs_b = service_b.recommend(cart, hora=15, dia_semana=5, top_k=3)

        assert len(recs_a) == 3
        for a, b in zip(recs_a, recs_b):
            assert a["sku_norm"] == b["sku_norm"]
            assert a["confianca"] == pytest.approx(b["confianca"], abs=1e-6)
            assert a["qtd_unidades_base"] == a["qtd_pacotes"] * a["multiplicador"]
            assert a["valor_estimado"] == pytest.approx(a["qtd_pacotes"] * a["preco_pacote"], abs=0.02)

    def test_nunca_recomenda_item_ja_no_carrinho_e_pula_item_sem_estoque(self) -> None:
        service = RecommenderService.from_artifact(RECOMMENDER_ARTIFACT_DIR)
        cart = [
            {"sku_norm": "Alfajor Preto (x1)", "qtd_pacotes": 1},
            {"sku_norm": "Cuca Alemã (x1)", "qtd_pacotes": 1},
        ]
        recs_all = service.recommend(cart, hora=14, dia_semana=6, top_k=5)
        cart_skus = {c["sku_norm"] for c in cart}
        for r in recs_all:
            assert r["sku_norm"] not in cart_skus

        # Simular falta de estoque do Top-1 atual -> ranking deve pular para o próximo!
        top1_sem_estoque = recs_all[0]["sku_norm"]
        top2_esperado = recs_all[1]["sku_norm"]
        available = [s["sku_norm"] for s in service.catalogo["skus"] if s["sku_norm"] != top1_sem_estoque]

        recs_filtradas = service.recommend(
            cart, hora=14, dia_semana=6, top_k=3, available_skus=available
        )
        assert len(recs_filtradas) == 3
        assert top1_sem_estoque not in {r["sku_norm"] for r in recs_filtradas}
        assert recs_filtradas[0]["sku_norm"] == top2_esperado

    def test_cold_start_carrinho_vazio_e_upsell_embalagem(self) -> None:
        service = RecommenderService.from_artifact(RECOMMENDER_ARTIFACT_DIR)
        recs_cs = service.recommend([], hora=11, dia_semana=6, top_k=3)
        assert len(recs_cs) == 3
        assert all(0.0 <= r["confianca"] <= 1.0 for r in recs_cs)

        # 2 unidades avulsas de Rapadura de Melado devem sugerir Pacote x3
        upsells = service.suggest_package_upsells(
            [{"sku_norm": "Rapadura de Melado (x1)", "qtd_pacotes": 2}]
        )
        assert len(upsells) == 1
        assert upsells[0]["sugestao_sku_norm"] == "Rapadura de Melado (x3)"


class TestFastAPIEndpoints:
    def test_health_catalog_metrics_e_recommend(self) -> None:
        client = TestClient(app)

        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_cat = client.get("/api/catalog")
        assert r_cat.status_code == 200
        cat = r_cat.json()
        assert cat["n_bases"] == 20
        assert cat["n_skus"] == 34

        r_met = client.get("/api/metrics")
        assert r_met.status_code == 200
        assert "hit_rate_at_3" in r_met.json()["summary"]

        r_rec = client.post(
            "/api/recommend",
            json={
                "cart_items": [{"sku_norm": "Rapadura de Melado (x1)", "qtd_pacotes": 2}],
                "hora": 15,
                "dia_semana": 5,
                "top_k": 3,
            },
        )
        assert r_rec.status_code == 200
        body = r_rec.json()
        assert len(body["recommendations"]) == 3
        assert len(body["upsells"]) >= 1
        assert "hit_rate_at_3" in body["model_metrics"]
