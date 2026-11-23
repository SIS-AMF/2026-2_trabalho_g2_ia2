"""API REST leve em FastAPI (`api/main.py`) servindo o modelo exportado `models/recommender_v1/`."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import RECOMMENDER_ARTIFACT_DIR
from src.serving.inference import RecommenderService

app = FastAPI(
    title="Feira PDV — API de Recomendação com Rede Neural",
    version="1.0.0",
    description=(
        "Serve o modelo exportado FairMultiTaskRecommender (PyTorch) para "
        "recomendações Top-K em tempo real com filtro pós-processamento de estoque."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_service() -> RecommenderService:
    return RecommenderService.from_artifact(RECOMMENDER_ARTIFACT_DIR)


class CartItemRequest(BaseModel):
    sku_norm: str = Field(..., description="Chave canônica do SKU (ex.: 'Alfajor Preto (x1)')")
    qtd_pacotes: int = Field(default=1, ge=1, description="Quantidade de pacotes/unidades no carrinho")


class RecommendRequest(BaseModel):
    cart_items: list[CartItemRequest] = Field(default_factory=list)
    hora: int | None = Field(default=None, ge=0, le=23)
    dia_semana: int | None = Field(default=None, ge=0, le=6)
    top_k: int = Field(default=3, ge=1, le=10)
    available_skus: list[str] | None = Field(
        default=None,
        description="Lista opcional de SKUs em estoque; itens fora dela são pulados no ranking Top-K.",
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    service = get_service()
    return {
        "status": "ok",
        "model_version": str(service.training_report.get("version", "recommender_v1")),
    }


@app.get("/api/catalog")
def get_catalog() -> dict[str, Any]:
    service = get_service()
    return service.catalogo


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    service = get_service()
    return {
        "summary": service.get_public_metrics(),
        "all_models_test_evaluation": service.training_report.get(
            "all_models_test_evaluation", []
        ),
        "best_epoch": service.training_report.get("best_epoch"),
        "hparams": service.training_report.get("hparams"),
    }


@app.post("/api/recommend")
def recommend(payload: RecommendRequest) -> dict[str, Any]:
    service = get_service()
    now = datetime.now()
    hora = payload.hora if payload.hora is not None else now.hour
    dia_semana = payload.dia_semana if payload.dia_semana is not None else now.weekday()

    cart_dicts = [item.model_dump() for item in payload.cart_items]
    recs = service.recommend(
        cart_items=cart_dicts,
        hora=hora,
        dia_semana=dia_semana,
        top_k=payload.top_k,
        available_skus=payload.available_skus,
    )
    upsells = service.suggest_package_upsells(
        cart_items=cart_dicts,
        available_skus=payload.available_skus,
    )
    return {
        "recommendations": recs,
        "upsells": upsells,
        "model_metrics": service.get_public_metrics(),
        "context_used": {
            "hora": hora,
            "dia_semana": dia_semana,
            "stock_filter_active": payload.available_skus is not None,
        },
    }
