"""Etapa 5 — Treinamento e Ajuste de Hiperparâmetros do Modelo Principal.

Responsabilidade exclusiva deste módulo:
- Treinar a rede neural principal `FairMultiTaskRecommender` (MLP Multi-Task de 3 cabeças)
  utilizando APENAS os conjuntos de Treino e Validação (nunca toca no Teste 20%).
- Executar a busca de hiperparâmetros com Optuna (`val_ndcg@3`).
- Salvar o gráfico de convergência (`07_curvas_treinamento.png`) e o checkpoint
  vencedor (`models/best_checkpoint.pt`).

Componentes explícitos do treinamento:
- Arquitetura + Regularização estrutural (`BatchNorm1d`, `Dropout`): `src/modeling/models.py`
- Função de Erro composta mascarada (`MultiTaskRecommendationLoss`): `src/modeling/losses.py`
- Otimizador + Regularizador L2: `torch.optim.AdamW(..., weight_decay=...)`
- Controle de convergência: `ReduceLROnPlateau` + `clip_grad_norm_` + `EarlyStopping`

A comparação com modelos alternativos (Baselines + BasketDAE) e a avaliação no
Teste Hold-Out (20%) ficam isoladas em `src/modeling/evaluate.py`.
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
    BEST_CHECKPOINT_PATH,
    CATALOG_METADATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    SEED,
    SPLITS_DIR,
)
from src.data.dataset_builder import carregar_splits
from src.modeling.losses import MultiTaskRecommendationLoss, calcular_pos_weights
from src.modeling.models import FairMultiTaskRecommender
from src.utils.ranking import (
    aplicar_mascaras_ranking,
    calcular_metricas_ranking,
    combinar_scores_hierarquicos,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Arquiteturas candidatas do backbone MLP avaliadas pelo Optuna
ARCH_OPTIONS: dict[str, tuple[int, ...]] = {
    "64_32": (64, 32),
    "128_64": (128, 64),
    "128_64_32": (128, 64, 32),
}


def fixar_seeds(seed: int = SEED) -> None:
    """Fixa as seeds de `random`, `numpy` e `torch` para reprodutibilidade."""
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
    """Executa inferência em lote retornando `(scores_sku, pred_qty_log1p)`."""
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
    """Treina o `FairMultiTaskRecommender` sobre Treino/Validação com Early Stopping."""
    fixar_seeds(seed)
    sku_to_base_idx = np.array([s["base_id"] for s in catalogo["skus"]], dtype=np.int64)

    # -------------------------------------------------------------------------
    # 1. ARQUITETURA DA REDE NEURAL (inclui regularização BatchNorm1d + Dropout)
    # -------------------------------------------------------------------------
    model = FairMultiTaskRecommender(
        in_dim=int(train_arr["X"].shape[1]),
        hidden_dims=hparams["hidden_dims"],
        n_bases=int(catalogo["n_bases"]),
        n_skus=int(catalogo["n_skus"]),
        dropout=float(hparams["dropout"]),
    )

    # -------------------------------------------------------------------------
    # 2. FUNÇÃO DE ERRO COMPOSTA MASCARADA (Loss das 3 cabeças)
    # -------------------------------------------------------------------------
    pw_cap = float(hparams.get("pos_weight_cap", 2.5))
    pw_base = calcular_pos_weights(train_arr["y_base"], max_weight=pw_cap)
    pw_sku = calcular_pos_weights(train_arr["y_sku"], max_weight=pw_cap)

    loss_fn = MultiTaskRecommendationLoss(
        w_base=float(hparams["w_base"]),
        w_sku=float(hparams["w_sku"]),
        w_qty=float(hparams["w_qty"]),
        pos_weight_base=pw_base,
        pos_weight_sku=pw_sku,
    )

    # -------------------------------------------------------------------------
    # 3. OTIMIZADOR (AdamW) + REGULARIZADOR L2 (weight_decay) + SCHEDULER
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),  # Regularização L2 nos pesos
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-5
    )

    # -------------------------------------------------------------------------
    # 4. LOOP DE TREINAMENTO (Backpropagation + Gradient Clipping + Early Stopping)
    # -------------------------------------------------------------------------
    train_loader = _fazer_dataloader(
        train_arr, batch_size=int(hparams["batch_size"]), shuffle=True
    )

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
        # 4.1 Fase de treino (Dropout e BatchNorm ativos)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for bx, byb, bys, byq, bmb, bms in train_loader:
            if len(bx) < 2:
                continue  # BatchNorm1d requer batch >= 2 durante o treino
            optimizer.zero_grad()
            lb, ls, pq = model(bx)
            loss, _ = loss_fn(lb, ls, pq, byb, bys, byq, bmb, bms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        avg_train_loss = train_loss_sum / max(1, n_batches)

        # 4.2 Avaliação na Validação (sem gradiente e sem tocar no Teste 20%)
        model.eval()
        with torch.no_grad():
            vlb, vls, vpq = model(val_x)
            val_loss_t, val_breakdown = loss_fn(
                vlb, vls, vpq, val_yb, val_ys, val_yq, val_mb, val_ms
            )
            v_probs_b = torch.sigmoid(vlb).numpy()
            v_probs_s = torch.sigmoid(vls).numpy()
            v_scores = combinar_scores_hierarquicos(
                v_probs_s, v_probs_b, sku_to_base_idx, lambda_hier
            )
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

        # 4.3 Early Stopping (regularização temporal — guarda pesos da melhor época)
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


def executar_busca_optuna(
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    catalogo: dict[str, Any],
    n_trials: int = 30,
    seed: int = SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Busca de hiperparâmetros com Optuna otimizando `val_ndcg@3` (validação)."""
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


def treinar_modelo_principal(
    n_trials: int = 30,
    seed: int = SEED,
    checkpoint_path: Path = BEST_CHECKPOINT_PATH,
) -> dict[str, Any]:
    """Executa tuning Optuna e treino final do `FairMultiTaskRecommender` (sem tocar no Teste)."""
    with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as fp:
        catalogo = json.load(fp)
    bundle = carregar_splits(SPLITS_DIR)

    # 1. Busca de hiperparâmetros na validação (Optuna)
    best_hparams, df_trials = executar_busca_optuna(
        train_arr=bundle["train"],
        val_arr=bundle["val"],
        catalogo=catalogo,
        n_trials=n_trials,
        seed=seed,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_trials.to_csv(MODELS_DIR / "optuna_trials.csv", index=False)

    # 2. Treino final da melhor configuração com registro completo de curvas
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

    # 3. Salvar checkpoint do modelo principal (lido por evaluate.py e export.py)
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
        "split_summary": bundle["summary"],
        "seed": seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    return {"checkpoint": checkpoint, "df_trials": df_trials}


def main() -> None:
    """Ponto de entrada CLI (`python -m src.modeling.train`)."""
    out = treinar_modelo_principal(n_trials=30, seed=SEED)
    ckpt = out["checkpoint"]
    print("=== ETAPA 5 — TREINAMENTO DO MODELO PRINCIPAL CONCLUÍDO ===")
    print("\nMelhores hiperparâmetros (Optuna — Validação):")
    print(json.dumps(ckpt["hparams"], indent=2))
    print(
        f"\nMelhor época (Early Stopping): {ckpt['best_epoch']} "
        f"| Val NDCG@3: {ckpt['best_val_ndcg@3']:.4f}"
    )
    print(f"Checkpoint salvo em: {BEST_CHECKPOINT_PATH}")
    print(f"Curvas salvas em: {FIGURES_DIR / '07_curvas_treinamento.png'}")


if __name__ == "__main__":
    main()
