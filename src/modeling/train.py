"""Etapa 5 & 6 — Treinamento, Ajuste de Hiperparâmetros (Optuna) e Avaliação no Teste 20%.

Treina e compara:
1. Baseline 1 — Popularidade Global
2. Baseline 2 — Co-ocorrência / Lift
3. BasketDAE — Denoising Autoencoder (Slide 42)
4. FairMultiTaskRecommender — MLP Multi-Task Genérico (3 cabeças) otimizado com Optuna
"""

from __future__ import annotations

import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CATALOG_METADATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PEDIDOS_CESTAS_PATH,
    SEED,
    SPLITS_DIR,
)
from src.data.dataset_builder import carregar_splits
from src.modeling.baselines import CooccurrenceBaseline, PopularityBaseline
from src.modeling.evaluate import (
    avaliar_modelo_completo,
    gerar_relatorio_qualitativo,
    salvar_resumo_avaliacao,
)
from src.modeling.losses import MultiTaskRecommendationLoss, calcular_pos_weights
from src.modeling.models import BasketDAE, FairMultiTaskRecommender
from src.utils.ranking import (
    aplicar_mascaras_ranking,
    calcular_metricas_ranking,
    combinar_scores_hierarquicos,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ARCH_OPTIONS: dict[str, tuple[int, ...]] = {
    "64_32": (64, 32),
    "128_64": (128, 64),
    "128_64_32": (128, 64, 32),
}


def fixar_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fazer_dataloader(
    split_arr: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(split_arr["X"]),
        torch.from_numpy(split_arr["y_base"]),
        torch.from_numpy(split_arr["y_sku"]),
        torch.from_numpy(split_arr["y_qty"]),
        torch.from_numpy(split_arr["mask_base"]),
        torch.from_numpy(split_arr["mask_sku"]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def prever_mlp_multitask(
    model: FairMultiTaskRecommender,
    X: np.ndarray,
    sku_to_base_idx: np.ndarray,
    lambda_hier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Executa inferência em lote retornando (scores_sku, pred_qty_log1p)."""
    model.eval()
    x_t = torch.from_numpy(X)
    logits_b, logits_s, pred_q = model(x_t)
    probs_b = torch.sigmoid(logits_b).numpy()
    probs_s = torch.sigmoid(logits_s).numpy()
    scores = combinar_scores_hierarquicos(
        probs_sku=probs_s,
        probs_base=probs_b,
        sku_to_base_idx=sku_to_base_idx,
        lambda_hier=lambda_hier,
    )
    return scores, pred_q.numpy()


def treinar_mlp_multitask(
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    catalogo: dict[str, Any],
    hparams: dict[str, Any],
    max_epochs: int = 120,
    patience: int = 18,
    seed: int = SEED,
) -> dict[str, Any]:
    """Treina um `FairMultiTaskRecommender` com AdamW, ReduceLROnPlateau e EarlyStopping."""
    fixar_seeds(seed)
    sku_to_base_idx = np.array([s["base_id"] for s in catalogo["skus"]], dtype=np.int64)

    model = FairMultiTaskRecommender(
        in_dim=int(train_arr["X"].shape[1]),
        hidden_dims=hparams["hidden_dims"],
        n_bases=int(catalogo["n_bases"]),
        n_skus=int(catalogo["n_skus"]),
        dropout=float(hparams["dropout"]),
    )

    pw_cap = float(hparams.get("pos_weight_cap", 2.5))
    pw_base = calcular_pos_weights(train_arr["y_base"], max_weight=pw_cap)
    pw_sku = calcular_pos_weights(train_arr["y_sku"], max_weight=pw_cap)

    criterion = MultiTaskRecommendationLoss(
        w_base=float(hparams["w_base"]),
        w_sku=float(hparams["w_sku"]),
        w_qty=float(hparams["w_qty"]),
        pos_weight_base=pw_base,
        pos_weight_sku=pw_sku,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-5
    )

    train_loader = _fazer_dataloader(train_arr, batch_size=int(hparams["batch_size"]), shuffle=True)

    # Tensores de validação para cálculo rápido da val_loss por época
    val_x = torch.from_numpy(val_arr["X"])
    val_yb = torch.from_numpy(val_arr["y_base"])
    val_ys = torch.from_numpy(val_arr["y_sku"])
    val_yq = torch.from_numpy(val_arr["y_qty"])
    val_mb = torch.from_numpy(val_arr["mask_base"])
    val_ms = torch.from_numpy(val_arr["mask_sku"])

    best_val_ndcg = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history: list[dict[str, float]] = []

    lambda_hier = float(hparams["lambda_hier"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for bx, byb, bys, byq, bmb, bms in train_loader:
            if len(bx) < 2:
                continue  # BatchNorm1d requer batch >= 2 durante treino
            optimizer.zero_grad()
            lb, ls, pq = model(bx)
            loss, _ = criterion(lb, ls, pq, byb, bys, byq, bmb, bms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        avg_train_loss = train_loss_sum / max(1, n_batches)

        # Avaliação na validação
        model.eval()
        with torch.no_grad():
            vlb, vls, vpq = model(val_x)
            val_loss_t, val_breakdown = criterion(vlb, vls, vpq, val_yb, val_ys, val_yq, val_mb, val_ms)
            v_probs_b = torch.sigmoid(vlb).numpy()
            v_probs_s = torch.sigmoid(vls).numpy()
            v_scores = combinar_scores_hierarquicos(v_probs_s, v_probs_b, sku_to_base_idx, lambda_hier)
            v_masked = aplicar_mascaras_ranking(v_scores, val_arr["mask_sku"])
            v_metrics = calcular_metricas_ranking(v_masked, val_arr["y_sku"], ks=(1, 3, 5))

        val_ndcg3 = v_metrics["ndcg@3"]
        val_hit3 = v_metrics["hit_rate@3"]
        scheduler.step(val_ndcg3)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 5),
                "val_loss": round(float(val_loss_t.item()), 5),
                "val_loss_base": round(val_breakdown["loss_base"], 5),
                "val_loss_sku": round(val_breakdown["loss_sku"], 5),
                "val_loss_qty": round(val_breakdown["loss_qty"], 5),
                "val_ndcg@3": val_ndcg3,
                "val_hit_rate@3": val_hit3,
                "val_coverage@3": v_metrics["coverage@3"],
            }
        )

        if val_ndcg3 > best_val_ndcg + 1e-5:
            best_val_ndcg = val_ndcg3
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_val_ndcg@3": best_val_ndcg,
        "best_epoch": best_epoch,
        "history": history,
        "hparams": hparams,
    }


def treinar_basket_dae(
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    n_skus: int = 34,
    max_epochs: int = 100,
    patience: int = 16,
    seed: int = SEED,
) -> tuple[BasketDAE, dict[str, Any]]:
    """Treina o Denoising Autoencoder (BasketDAE) comparativo."""
    fixar_seeds(seed)
    model = BasketDAE(n_skus=n_skus, n_time=6, hidden_dim=64, bottleneck_dim=24, dropout=0.25, masking_noise=0.35)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    # No DAE, reconstruímos a cesta completa (entrada + alvo) a partir da entrada corrompida
    x_sku_train = train_arr["X"][:, :n_skus]
    x_time_train = train_arr["X"][:, -6:]
    y_full_train = np.clip(x_sku_train + train_arr["y_sku"], 0.0, 1.0).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(y_full_train),
        torch.from_numpy(x_time_train),
        torch.from_numpy(y_full_train),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    val_x_sku = torch.from_numpy(val_arr["X"][:, :n_skus])
    val_x_time = torch.from_numpy(val_arr["X"][:, -6:])

    best_ndcg = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx_full, bx_time, by_full in loader:
            if len(bx_full) < 2:
                continue
            optimizer.zero_grad()
            logits = model(bx_full, bx_time)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, by_full)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(val_x_sku, val_x_time)
            v_scores = torch.sigmoid(v_logits).numpy()
            v_masked = aplicar_mascaras_ranking(v_scores, val_arr["mask_sku"])
            v_ndcg = calcular_metricas_ranking(v_masked, val_arr["y_sku"], ks=(1, 3, 5))["ndcg@3"]

        if v_ndcg > best_ndcg + 1e-5:
            best_ndcg = v_ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_val_ndcg@3": best_ndcg, "best_epoch": best_epoch}


@torch.no_grad()
def prever_dae(
    model: BasketDAE,
    X: np.ndarray,
    n_skus: int = 34,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(torch.from_numpy(X[:, :n_skus]), torch.from_numpy(X[:, -6:]))
    scores = torch.sigmoid(logits).numpy().astype(np.float32)
    qty_pred = np.full((len(X), n_skus), np.log1p(1.0), dtype=np.float32)
    return scores, qty_pred


def executar_busca_optuna(
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    catalogo: dict[str, Any],
    n_trials: int = 30,
    seed: int = SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Busca de hiperparâmetros com Optuna otimizando `val_ndcg@3` (SKU)."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        arch_key = trial.suggest_categorical("arch", list(ARCH_OPTIONS.keys()))
        hparams = {
            "hidden_dims": ARCH_OPTIONS[arch_key],
            "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.15, 0.45),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "w_base": trial.suggest_float("w_base", 0.3, 1.2),
            "w_sku": 1.5,
            "w_qty": trial.suggest_float("w_qty", 0.2, 0.8),
            "pos_weight_cap": trial.suggest_float("pos_weight_cap", 1.0, 4.5),
            "lambda_hier": trial.suggest_float("lambda_hier", 0.0, 0.45),
        }
        res = treinar_mlp_multitask(
            train_arr=train_arr,
            val_arr=val_arr,
            catalogo=catalogo,
            hparams=hparams,
            max_epochs=90,
            patience=14,
            seed=seed,
        )
        trial.set_user_attr("best_epoch", res["best_epoch"])
        return float(res["best_val_ndcg@3"])

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_p = study.best_params
    best_hparams = {
        "hidden_dims": ARCH_OPTIONS[best_p["arch"]],
        "lr": float(best_p["lr"]),
        "dropout": float(best_p["dropout"]),
        "weight_decay": float(best_p["weight_decay"]),
        "batch_size": int(best_p["batch_size"]),
        "w_base": float(best_p["w_base"]),
        "w_sku": 1.5,
        "w_qty": float(best_p["w_qty"]),
        "pos_weight_cap": float(best_p["pos_weight_cap"]),
        "lambda_hier": float(best_p["lambda_hier"]),
    }

    rows = [
        {
            "trial": t.number,
            "val_ndcg@3": round(float(t.value), 4) if t.value is not None else None,
            "best_epoch": t.user_attrs.get("best_epoch"),
            **t.params,
        }
        for t in study.trials
    ]
    df_trials = pd.DataFrame(rows).sort_values("val_ndcg@3", ascending=False)
    return best_hparams, df_trials


def gerar_grafico_curvas_treinamento(
    history: list[dict[str, float]],
    best_epoch: int,
    caminho: Path = FIGURES_DIR / "07_curvas_treinamento.png",
) -> None:
    """07: Curvas de Loss (treino vs. validação) e NDCG@3 / Hit Rate@3 por época."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df_h = pd.DataFrame(history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(df_h["epoch"], df_h["train_loss"], label="Loss Treino", color="#1f5a99", linewidth=2)
    ax1.plot(df_h["epoch"], df_h["val_loss"], label="Loss Validação", color="#e07a2f", linewidth=2)
    ax1.axvline(best_epoch, color="#2a9d6e", linestyle="--", linewidth=1.4, label=f"Melhor Época ({best_epoch})")
    ax1.set_title("Convergência da Função de Erro Composta (Multi-Task Loss)", fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df_h["epoch"], df_h["val_hit_rate@3"] * 100.0, label="Val Hit Rate@3 (%)", color="#1f5a99", linewidth=2)
    ax2.plot(df_h["epoch"], df_h["val_ndcg@3"] * 100.0, label="Val NDCG@3 (%)", color="#2a9d6e", linewidth=2)
    ax2.plot(df_h["epoch"], df_h["val_coverage@3"] * 100.0, label="Val Coverage@3 (%)", color="#7b4ea3", linestyle=":", linewidth=2)
    ax2.axvline(best_epoch, color="#2a9d6e", linestyle="--", linewidth=1.4)
    ax2.set_title("Métricas de Ranking na Validação (Early Stopping)", fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Desempenho (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Treinamento do FairMultiTaskRecommender (AdamW + Dropout + L2 + Early Stopping)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def executar_pipeline_treino_e_avaliacao(
    n_trials: int = 30,
    seed: int = SEED,
) -> dict[str, Any]:
    """Executa tuning Optuna, treino dos 4 modelos, avaliação Hold-Out 20% e salva checkpoint."""
    df_cestas = pd.read_parquet(PEDIDOS_CESTAS_PATH)
    with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as fp:
        catalogo = json.load(fp)
    bundle = carregar_splits(SPLITS_DIR)

    n_skus = int(catalogo["n_skus"])
    sku_to_base_idx = np.array([s["base_id"] for s in catalogo["skus"]], dtype=np.int64)

    # 1. Baselines (ajustados apenas no conjunto de treino)
    pop_model = PopularityBaseline(n_skus=n_skus).fit(df_cestas, bundle["splits"]["train_ids"])
    cooc_model = CooccurrenceBaseline(n_skus=n_skus, smoothing=1.0).fit(
        df_cestas, bundle["splits"]["train_ids"]
    )

    # 2. Denoising Autoencoder comparativo (BasketDAE)
    dae_model, dae_info = treinar_basket_dae(
        train_arr=bundle["train"],
        val_arr=bundle["val"],
        n_skus=n_skus,
        seed=seed,
    )

    # 3. Busca de hiperparâmetros com Optuna para o FairMultiTaskRecommender
    best_hparams, df_trials = executar_busca_optuna(
        train_arr=bundle["train"],
        val_arr=bundle["val"],
        catalogo=catalogo,
        n_trials=n_trials,
        seed=seed,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_trials.to_csv(MODELS_DIR / "optuna_trials.csv", index=False)

    # 4. Treino final do melhor MLP Multi-Task com registro completo de curvas
    mlp_res = treinar_mlp_multitask(
        train_arr=bundle["train"],
        val_arr=bundle["val"],
        catalogo=catalogo,
        hparams=best_hparams,
        max_epochs=150,
        patience=20,
        seed=seed,
    )
    mlp_model: FairMultiTaskRecommender = mlp_res["model"]
    gerar_grafico_curvas_treinamento(mlp_res["history"], mlp_res["best_epoch"])

    # 5. Avaliação imparcial no conjunto de Teste Hold-Out (20%)
    test_arr = bundle["test"]

    pop_scores, pop_qty = pop_model.predict_scores(test_arr["X"], test_arr["mask_sku"])
    cooc_scores, cooc_qty = cooc_model.predict_scores(test_arr["X"], test_arr["mask_sku"])
    dae_scores, dae_qty = prever_dae(dae_model, test_arr["X"], n_skus=n_skus)
    mlp_scores, mlp_qty = prever_mlp_multitask(
        mlp_model, test_arr["X"], sku_to_base_idx, best_hparams["lambda_hier"]
    )

    resultados = [
        avaliar_modelo_completo("Baseline 1: Popularidade", pop_scores, pop_qty, test_arr, catalogo),
        avaliar_modelo_completo("Baseline 2: Co-ocorrência", cooc_scores, cooc_qty, test_arr, catalogo),
        avaliar_modelo_completo("BasketDAE (Autoencoder)", dae_scores, dae_qty, test_arr, catalogo),
        avaliar_modelo_completo("MLP Multi-Task (Principal)", mlp_scores, mlp_qty, test_arr, catalogo),
    ]
    df_eval = salvar_resumo_avaliacao(resultados)

    # Relatório qualitativo no conjunto de teste (8 casos reais)
    gerar_relatorio_qualitativo(
        scores_sku=mlp_scores,
        pred_qty_log1p=mlp_qty,
        test_meta=bundle["test_meta"],
        mask_sku=test_arr["mask_sku"],
        catalogo=catalogo,
    )

    # Salvar checkpoint completo intermediário para a Etapa 7 (Exportação)
    checkpoint = {
        "model_state_dict": mlp_model.state_dict(),
        "model_config": mlp_model.get_config(),
        "hparams": {
            **best_hparams,
            "hidden_dims": list(best_hparams["hidden_dims"]),
        },
        "best_epoch": mlp_res["best_epoch"],
        "best_val_ndcg@3": mlp_res["best_val_ndcg@3"],
        "history": mlp_res["history"],
        "dae_info": dae_info,
        "test_evaluation": resultados,
        "split_summary": bundle["summary"],
        "seed": seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, MODELS_DIR / "best_checkpoint.pt")

    return {"checkpoint": checkpoint, "df_eval": df_eval, "df_trials": df_trials}


def main() -> None:
    """Ponto de entrada CLI (`python -m src.modeling.train`)."""
    out = executar_pipeline_treino_e_avaliacao(n_trials=30, seed=SEED)
    print("=== ETAPAS 5 & 6 — TREINO, TUNING E AVALIAÇÃO CONCLUÍDOS ===")
    print("\nMelhores hiperparâmetros (Optuna):")
    print(json.dumps(out["checkpoint"]["hparams"], indent=2))
    print(f"\nMelhor época (Early Stopping): {out['checkpoint']['best_epoch']}")
    print("\n=== TABELA COMPARATIVA NO TESTE HOLD-OUT (20%) ===")
    cols = [
        "modelo",
        "sku_hit_rate@1",
        "sku_hit_rate@3",
        "sku_ndcg@3",
        "sku_mrr",
        "sku_coverage@3",
        "base_hit_rate@3",
        "qty_mae",
        "qty_exact_acc",
    ]
    print(out["df_eval"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
