"""Orquestrador ponta-a-ponta do pipeline de dados, treinamento, avaliação e exportação.

Exemplos de uso:
    python -m src.pipeline                # Executa todas as etapas (1 -> 7)
    python -m src.pipeline --skip-train   # Reprocessa dados, avalia benchmarks e reexporta checkpoint atual
    python -m src.pipeline --only-data    # Executa apenas a camada de dados (Etapas 1 -> 4)
"""

from __future__ import annotations

import argparse

from src.data import dataset_builder, eda, features, preprocessing
from src.modeling import evaluate, train
from src.serving import export


def executar_pipeline(
    skip_train: bool = False,
    only_data: bool = False,
) -> None:
    """Executa as etapas do pipeline na ordem exata do DAG de dependências."""
    print(">>> [1/7] Pré-processamento e auditoria contábil (src.data.preprocessing)")
    preprocessing.main()

    print("\n>>> [2/7] Engenharia de features e catálogo oficial (src.data.features)")
    features.main()

    print("\n>>> [3/7] Análise exploratória e geração de gráficos (src.data.eda)")
    eda.main()

    print("\n>>> [4/7] Split 80/20 anti-leakage e tensores (src.data.dataset_builder)")
    dataset_builder.main()

    if only_data:
        print("\n=== PIPELINE DE DADOS (ETAPAS 1 -> 4) CONCLUÍDO ===")
        return

    if not skip_train:
        print("\n>>> [5/7] Treinamento da rede principal + Optuna (src.modeling.train)")
        train.main()
    else:
        print("\n>>> [5/7] Treinamento pulado (--skip-train): reutilizando models/best_checkpoint.pt.")

    print("\n>>> [6/7] Avaliação comparativa no Teste 20%: Baselines + DAE vs. MLP (src.modeling.evaluate)")
    evaluate.main()

    print("\n>>> [7/7] Exportação versionada do pacote do modelo (src.serving.export)")
    export.main()
    print("\n=== PIPELINE COMPLETO CONCLUÍDO COM SUCESSO ===")


def main() -> None:
    """Ponto de entrada CLI (`python -m src.pipeline`)."""
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo do sistema de recomendação de feiras."
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Pula o treinamento/Optuna da rede principal e reavalia/reexporta usando models/best_checkpoint.pt.",
    )
    parser.add_argument(
        "--only-data",
        action="store_true",
        help="Executa apenas a camada de dados (pré-processamento, features, EDA e splits).",
    )
    args = parser.parse_args()
    executar_pipeline(skip_train=args.skip_train, only_data=args.only_data)


if __name__ == "__main__":
    main()
