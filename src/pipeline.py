"""Orquestrador ponta-a-ponta do pipeline de dados, treinamento e exportação.

Exemplos de uso:
    python -m src.pipeline                # Executa todas as etapas (1 -> 7)
    python -m src.pipeline --skip-train   # Reprocessa dados + EDA + splits + exporta checkpoint atual
    python -m src.pipeline --only-data    # Executa apenas a camada de dados (Etapas 1 -> 4)
"""

from __future__ import annotations

import argparse

from src.data import dataset_builder, eda, features, preprocessing
from src.modeling import train
from src.serving import export


def executar_pipeline(
    skip_train: bool = False,
    only_data: bool = False,
) -> None:
    """Executa as etapas do pipeline na ordem exata do DAG de dependências."""
    print(">>> [1/6] Pré-processamento e auditoria contábil (src.data.preprocessing)")
    preprocessing.main()

    print("\n>>> [2/6] Engenharia de features e catálogo oficial (src.data.features)")
    features.main()

    print("\n>>> [3/6] Análise exploratória e geração de gráficos (src.data.eda)")
    eda.main()

    print("\n>>> [4/6] Split 80/20 anti-leakage e tensores (src.data.dataset_builder)")
    dataset_builder.main()

    if only_data:
        print("\n=== PIPELINE DE DADOS (ETAPAS 1 -> 4) CONCLUÍDO ===")
        return

    if not skip_train:
        print("\n>>> [5/6] Treinamento, tuning Optuna e avaliação Hold-Out (src.modeling.train)")
        train.main()
    else:
        print("\n>>> [5/6] Treinamento pulado (--skip-train): reutilizando checkpoint existente.")

    print("\n>>> [6/6] Exportação versionada do pacote do modelo (src.serving.export)")
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
        help="Pula a etapa de treinamento/Optuna e reexporta usando models/best_checkpoint.pt.",
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
