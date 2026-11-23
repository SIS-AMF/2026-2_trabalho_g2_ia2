"""Etapa 7 — Exportação Versionada do Modelo e Artefatos (`models/recommender_v1/`)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import torch

from src.config import (
    BEST_CHECKPOINT_PATH,
    CATALOG_METADATA_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_WEIGHTS_PATH,
    PREPROCESSING_ARTIFACTS_PATH,
    RECOMMENDER_ARTIFACT_DIR,
    SPLITS_DIR,
    TRAINING_REPORT_PATH,
)
from src.modeling.models import FairMultiTaskRecommender


def exportar_pacote_modelo(
    checkpoint_path: Path = BEST_CHECKPOINT_PATH,
    evaluation_report_path: Path = EVALUATION_REPORT_PATH,
    splits_dir: Path = SPLITS_DIR,
    catalogo_path: Path = CATALOG_METADATA_PATH,
    artifact_dir: Path = RECOMMENDER_ARTIFACT_DIR,
) -> dict[str, Path]:
    """Empacota pesos, pré-processadores, catálogo e relatório em `models/recommender_v1/`."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["model_config"]
    hparams = checkpoint["hparams"]

    # 1. Reconstruir modelo e validar state_dict antes de exportar
    model = FairMultiTaskRecommender(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        n_bases=int(model_cfg["n_bases"]),
        n_skus=int(model_cfg["n_skus"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Salvar pesos + configuração arquitetural (`model_weights.pt`)
    weights_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg,
        "lambda_hier": float(hparams["lambda_hier"]),
        "version": "recommender_v1",
    }
    weights_out = artifact_dir / MODEL_WEIGHTS_PATH.name
    torch.save(weights_payload, weights_out)

    # 2. Artefatos de pré-processamento (`preprocessing_artifacts.joblib`)
    with catalogo_path.open("r", encoding="utf-8") as fp:
        catalogo = json.load(fp)
    scaler = joblib.load(splits_dir / "scaler.joblib")

    prep_artifacts = {
        "scaler": scaler,
        "base_to_idx": {b["produto_base"]: int(b["base_id"]) for b in catalogo["bases"]},
        "sku_to_idx": {s["sku_norm"]: int(s["sku_id"]) for s in catalogo["skus"]},
        "idx_to_sku": {int(s["sku_id"]): s["sku_norm"] for s in catalogo["skus"]},
        "input_dim": int(model_cfg["in_dim"]),
    }
    prep_out = artifact_dir / PREPROCESSING_ARTIFACTS_PATH.name
    joblib.dump(prep_artifacts, prep_out)

    # 4. Garantir `catalog_metadata.json` presente no diretório versionado
    cat_out = artifact_dir / CATALOG_METADATA_PATH.name
    if catalogo_path.resolve() != cat_out.resolve():
        shutil.copy2(catalogo_path, cat_out)

    # 4. Relatório completo de treino e métricas (`training_report.json`)
    if evaluation_report_path.exists():
        with evaluation_report_path.open("r", encoding="utf-8") as fp:
            eval_doc = json.load(fp)
        test_evaluation = eval_doc["all_models_test_evaluation"]
        mlp_eval = eval_doc["principal_test_metrics"]
    else:
        test_evaluation = checkpoint["test_evaluation"]
        mlp_eval = next(
            (r for r in test_evaluation if "MLP Multi-Task" in r["modelo"]),
            test_evaluation[-1],
        )

    training_report: dict[str, Any] = {
        "version": "recommender_v1",
        "model_name": "FairMultiTaskRecommender",
        "trained_at_utc": checkpoint["trained_at_utc"],
        "seed": checkpoint["seed"],
        "model_config": model_cfg,
        "hparams": hparams,
        "best_epoch": checkpoint["best_epoch"],
        "best_val_ndcg@3": checkpoint["best_val_ndcg@3"],
        "split_summary": checkpoint["split_summary"],
        "principal_test_metrics": mlp_eval,
        "all_models_test_evaluation": test_evaluation,
        "history": checkpoint["history"],
    }
    report_out = artifact_dir / TRAINING_REPORT_PATH.name
    with report_out.open("w", encoding="utf-8") as fp:
        json.dump(training_report, fp, ensure_ascii=False, indent=2)

    return {
        "weights": weights_out,
        "preprocessing_artifacts": prep_out,
        "catalog": cat_out,
        "training_report": report_out,
    }


def main() -> None:
    """Ponto de entrada CLI (`python -m src.serving.export`)."""
    paths = exportar_pacote_modelo()
    print("=== ETAPA 7 — MODELO EXPORTADO COM SUCESSO ===")
    for nome, p in paths.items():
        print(f"  - {nome}: {p} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
