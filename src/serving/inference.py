"""Etapa 7 — Serviço de Inferência (`RecommenderService`) com Filtro de Estoque Top-K."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from src.config import (
    CATALOG_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    PREPROCESSING_ARTIFACTS_PATH,
    RECOMMENDER_ARTIFACT_DIR,
    TRAINING_REPORT_PATH,
)
from src.modeling.models import FairMultiTaskRecommender
from src.utils.baskets import CART_STATS_SLICE, construir_vetor_entrada_raw
from src.utils.ranking import aplicar_mascaras_ranking, combinar_scores_hierarquicos


class RecommenderService:
    """Carrega o pacote versionado `models/recommender_v1/` e serve recomendações completas."""

    def __init__(
        self,
        model: FairMultiTaskRecommender,
        scaler: Any,
        catalogo: dict[str, Any],
        lambda_hier: float,
        training_report: dict[str, Any],
    ) -> None:
        self.model = model
        self.model.eval()
        self.scaler = scaler
        self.catalogo = catalogo
        self.lambda_hier = float(lambda_hier)
        self.training_report = training_report

        self.sku_by_id: dict[int, dict[str, Any]] = {
            int(s["sku_id"]): s for s in catalogo["skus"]
        }
        self.sku_by_norm: dict[str, dict[str, Any]] = {
            str(s["sku_norm"]): s for s in catalogo["skus"]
        }
        self.sku_to_base_idx = np.array(
            [int(s["base_id"]) for s in catalogo["skus"]], dtype=np.int64
        )
        # Mapa produto_base -> lista de embalagens múltiplas (mult > 1) ordenadas por multiplicador
        self.pacotes_por_base: dict[str, list[dict[str, Any]]] = {}
        for s in catalogo["skus"]:
            if int(s["multiplicador"]) > 1:
                self.pacotes_por_base.setdefault(str(s["produto_base"]), []).append(s)
        for lst in self.pacotes_por_base.values():
            lst.sort(key=lambda item: int(item["multiplicador"]))

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path = RECOMMENDER_ARTIFACT_DIR,
    ) -> "RecommenderService":
        art = Path(artifact_dir)
        weights_payload = torch.load(
            art / MODEL_WEIGHTS_PATH.name, map_location="cpu", weights_only=False
        )
        cfg = weights_payload["model_config"]
        model = FairMultiTaskRecommender(
            in_dim=int(cfg["in_dim"]),
            hidden_dims=tuple(cfg["hidden_dims"]),
            n_bases=int(cfg["n_bases"]),
            n_skus=int(cfg["n_skus"]),
            dropout=float(cfg["dropout"]),
        )
        model.load_state_dict(weights_payload["model_state_dict"])
        model.eval()

        prep = joblib.load(art / PREPROCESSING_ARTIFACTS_PATH.name)
        with (art / CATALOG_METADATA_PATH.name).open("r", encoding="utf-8") as fp:
            catalogo = json.load(fp)
        with (art / TRAINING_REPORT_PATH.name).open("r", encoding="utf-8") as fp:
            report = json.load(fp)

        return cls(
            model=model,
            scaler=prep["scaler"],
            catalogo=catalogo,
            lambda_hier=float(weights_payload["lambda_hier"]),
            training_report=report,
        )

    def _construir_available_mask(
        self,
        available_skus: list[str] | None,
    ) -> np.ndarray | None:
        if available_skus is None:
            return None
        allowed = set(available_skus)
        mask = np.zeros(int(self.catalogo["n_skus"]), dtype=np.float32)
        for sid, info in self.sku_by_id.items():
            if info["sku_norm"] in allowed or info["display_name"] in allowed:
                mask[sid] = 1.0
        return mask

    @torch.no_grad()
    def recommend(
        self,
        cart_items: list[dict[str, Any]],
        hora: int = 15,
        dia_semana: int = 5,
        top_k: int = 3,
        available_skus: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retorna Top-K recomendações completas, coerentes e disponíveis em estoque."""
        # Filtrar itens desconhecidos sem quebrar inferência
        valid_cart = [
            it
            for it in cart_items
            if (
                ("sku_id" in it and int(it["sku_id"]) in self.sku_by_id)
                or ("sku_norm" in it and str(it["sku_norm"]) in self.sku_by_norm)
            )
        ]

        x_raw, mask_sku, _ = construir_vetor_entrada_raw(
            cart_items=valid_cart,
            hora=hora,
            dia_semana=dia_semana,
            catalogo=self.catalogo,
        )
        x_scaled = x_raw.copy()
        x_scaled[CART_STATS_SLICE] = self.scaler.transform(
            x_raw[CART_STATS_SLICE].reshape(1, -1)
        ).astype(np.float32)[0]

        x_t = torch.from_numpy(x_scaled).unsqueeze(0)
        logits_b, logits_s, pred_q = self.model(x_t)

        probs_b = torch.sigmoid(logits_b).numpy()
        probs_s = torch.sigmoid(logits_s).numpy()
        scores = combinar_scores_hierarquicos(
            probs_sku=probs_s,
            probs_base=probs_b,
            sku_to_base_idx=self.sku_to_base_idx,
            lambda_hier=self.lambda_hier,
        )

        avail_mask = self._construir_available_mask(available_skus)
        masked_scores = aplicar_mascaras_ranking(
            scores=scores,
            mask_sku=mask_sku.reshape(1, -1),
            available_mask=avail_mask.reshape(1, -1) if avail_mask is not None else None,
        )[0]

        pred_qty_vec = pred_q.numpy()[0]
        order = np.argsort(-masked_scores)

        recomendacoes: list[dict[str, Any]] = []
        for sid in order:
            score_val = float(masked_scores[int(sid)])
            if score_val < 0.0:
                continue
            info = self.sku_by_id[int(sid)]
            mult = int(info["multiplicador"])
            qtd_pacotes = max(1, int(round(math.expm1(float(pred_qty_vec[int(sid)])))))
            qtd_unidades = qtd_pacotes * mult
            preco_pkg = float(info["preco_pacote_ref"])
            preco_unit = float(info["preco_unitario_ref"])

            recomendacoes.append(
                {
                    "rank": len(recomendacoes) + 1,
                    "sku_id": int(sid),
                    "sku_norm": str(info["sku_norm"]),
                    "produto_base": str(info["produto_base"]),
                    "produto_com_multiplicador": str(info["display_name"]),
                    "tipo_embalagem": str(info["tipo_embalagem"]),
                    "multiplicador": mult,
                    "qtd_pacotes": qtd_pacotes,
                    "qtd_unidades_base": qtd_unidades,
                    "preco_unitario": round(preco_unit, 2),
                    "preco_pacote": round(preco_pkg, 2),
                    "valor_estimado": round(qtd_pacotes * preco_pkg, 2),
                    "confianca": round(score_val, 4),
                }
            )
            if len(recomendacoes) >= int(top_k):
                break

        return recomendacoes

    def suggest_package_upsells(
        self,
        cart_items: list[dict[str, Any]],
        available_skus: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Camada auxiliar determinística: sugere migração de unidades avulsas para pacote."""
        allowed = set(available_skus) if available_skus is not None else None
        upsells: list[dict[str, Any]] = []

        for it in cart_items:
            info = (
                self.sku_by_id.get(int(it["sku_id"]))
                if "sku_id" in it
                else self.sku_by_norm.get(str(it.get("sku_norm", "")))
            )
            if info is None or int(info["multiplicador"]) != 1:
                continue

            qtd_un = max(1, int(it.get("qtd_pacotes", 1)))
            base = str(info["produto_base"])
            pacotes = self.pacotes_por_base.get(base, [])
            for pkg in pacotes:
                mult = int(pkg["multiplicador"])
                if allowed is not None and pkg["sku_norm"] not in allowed:
                    continue
                # Sugerir se cliente já está perto do pacote (>= 50% do multiplicador, mín 2 un)
                # e o pacote oferece economia unitária real
                if qtd_un >= max(2, math.ceil(mult * 0.5)) and float(pkg["economia_unitaria_pct"]) > 0.0:
                    custo_atual = round(qtd_un * float(info["preco_unitario_ref"]), 2)
                    custo_pacote = float(pkg["preco_pacote_ref"])
                    upsells.append(
                        {
                            "origem_sku_norm": str(info["sku_norm"]),
                            "origem_display_name": str(info["display_name"]),
                            "qtd_unidades_no_carrinho": qtd_un,
                            "custo_atual": custo_atual,
                            "sugestao_sku_norm": str(pkg["sku_norm"]),
                            "sugestao_display_name": str(pkg["display_name"]),
                            "multiplicador": mult,
                            "preco_pacote": custo_pacote,
                            "preco_unitario_pacote": float(pkg["preco_unitario_ref"]),
                            "economia_unitaria_pct": float(pkg["economia_unitaria_pct"]),
                            "mensagem": (
                                f"Cliente levando {qtd_un} un. de {base} (R$ {custo_atual:.2f}). "
                                f"Sugira trocar pelo {pkg['display_name']} por R$ {custo_pacote:.2f} "
                                f"({pkg['economia_unitaria_pct']:.1f}% de economia por unidade)!"
                            ),
                        }
                    )
                    break

        return upsells

    def get_public_metrics(self) -> dict[str, Any]:
        m = self.training_report.get("principal_test_metrics", {})
        return {
            "version": self.training_report.get("version", "recommender_v1"),
            "model_name": self.training_report.get("model_name", "FairMultiTaskRecommender"),
            "hit_rate_at_1": m.get("sku_hit_rate@1", 0.0),
            "hit_rate_at_3": m.get("sku_hit_rate@3", 0.0),
            "hit_rate_at_5": m.get("sku_hit_rate@5", 0.0),
            "ndcg_at_3": m.get("sku_ndcg@3", 0.0),
            "base_hit_rate_at_3": m.get("base_hit_rate@3", 0.0),
            "coverage_at_3": m.get("sku_coverage@3", 0.0),
            "qty_mae": m.get("qty_mae", 0.0),
            "qty_exact_acc": m.get("qty_exact_acc", 0.0),
        }
