"""Configurações centrais, caminhos e mapas canônicos do projeto."""

from __future__ import annotations

from pathlib import Path

# Diretórios e arquivos
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_RAW_PATH: Path = PROJECT_ROOT / "dataset.csv"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
ITENS_LIMPOS_PATH: Path = DATA_PROCESSED_DIR / "itens_limpos.parquet"
PEDIDOS_CESTAS_PATH: Path = DATA_PROCESSED_DIR / "pedidos_cestas.parquet"
SPLITS_DIR: Path = DATA_PROCESSED_DIR / "splits"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RECOMMENDER_ARTIFACT_DIR: Path = MODELS_DIR / "recommender_v1"
CATALOG_METADATA_PATH: Path = RECOMMENDER_ARTIFACT_DIR / "catalog_metadata.json"
MODEL_WEIGHTS_PATH: Path = RECOMMENDER_ARTIFACT_DIR / "model_weights.pt"
PREPROCESSING_ARTIFACTS_PATH: Path = (
    RECOMMENDER_ARTIFACT_DIR / "preprocessing_artifacts.joblib"
)
TRAINING_REPORT_PATH: Path = RECOMMENDER_ARTIFACT_DIR / "training_report.json"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
PREPROCESSING_AUDIT_PATH: Path = REPORTS_DIR / "preprocessing_audit.json"
EDA_SUMMARY_PATH: Path = REPORTS_DIR / "eda_summary.json"
EVALUATION_SUMMARY_PATH: Path = REPORTS_DIR / "evaluation_summary.csv"
QUALITATIVE_CASES_PATH: Path = REPORTS_DIR / "qualitative_cases.md"

# Reprodutibilidade e tolerâncias
SEED: int = 42
TIMEZONE: str = "America/Sao_Paulo"
TOLERANCIA_MONETARIA: float = 0.01
TOLERANCIA_INTEIRO: float = 1e-6

# Split 80/20 (Desenvolvimento / Teste Hold-Out) e validação interna
TEST_SIZE: float = 0.20
VAL_SIZE_FROM_DEV: float = 0.15

# Faixas horárias canônicas (ordem fixa para one-hot)
FAIXAS_HORARIAS: tuple[str, ...] = ("Manhã", "Tarde", "Noite")

# Colunas descartadas (Etapa 1)
COLUNAS_VENDEDOR_INVALIDAS: tuple[str, ...] = (
    "pedido_vendedor",
    "item_pedido_user_id",
)
COLUNA_TIPO_DESCONTO: str = "pedido_tipo_desconto"

# Mapa explícito de sinônimos de produtos (Etapa 1 - Seção 4)
# REGRA ESTRITA: nenhuma fusão automática por similaridade textual.
# Único sinônimo confirmado pelo negócio (mesmo produto com troca de embalagem).
MAPA_SINONIMOS_PRODUTOS: dict[str, str] = {
    "Amendoim salgado e temperado": "Amendoim Temperado",
}

# Mapa de unificação de eventos (apenas para auditoria e EDA; não entra na NN)
MAPA_EVENTOS: dict[str, tuple[str, str]] = {
    "1 Moranguinho": ("Festa do Moranguinho", "Caixa 1"),
    "2-Moranguinho": ("Festa do Moranguinho", "Caixa 2"),
}
PONTO_VENDA_PADRAO: str = "Padrão"

# Mapa de padronização de métodos de pagamento
MAPA_PAGAMENTO_PADRAO: dict[str, str] = {
    "Dinheiro": "Dinheiro",
    "Pix": "PIX",
    "PIX": "PIX",
    "Cartão": "Cartão",
    "Cartão de Crédito": "Cartão de Crédito",
    "Cartão de Débito": "Cartão de Débito",
}
MAPA_PAGAMENTO_GRUPO: dict[str, str] = {
    "Dinheiro": "Dinheiro",
    "PIX": "PIX",
    "Cartão": "Cartão",
    "Cartão de Crédito": "Cartão",
    "Cartão de Débito": "Cartão",
}

# Mapa determinístico de embalagens sem "xN" (exclusivo da 1 Moranguinho)
MAPA_EMBALAGEM_SEM_MULTIPLICADOR: dict[tuple[str, str], int] = {
    ("Cuca Alemã", "Fardo"): 3,
    ("Rapadura Assada", "Pacote"): 6,
    ("Rapadura de Melado", "Pacote"): 3,
    ("Alfajor Preto", "Pacote"): 4,
    ("Cuca Enrolada de Amendoim", "Pacote"): 2,
    ("Cuca Enrolada de Chocolate", "Pacote"): 2,
}

# Critérios de classificação de pedidos consolidados / fechamento de caixa
EVENTOS_CONSOLIDADOS: frozenset[str] = frozenset({"Feira Polêsine", "ExpoSerra"})
LIMITE_SUBTOTAL_CONSOLIDADO: float = 200.0
LIMITE_UNIDADES_CONSOLIDADO: int = 40

# Ordem canônica da tabela auditada (Etapa 1 - Seção 7)
COLUNAS_ITENS_LIMPOS: tuple[str, ...] = (
    "pedido_id",
    "pedido_data",
    "evento_nome",
    "evento_original",
    "ponto_venda",
    "pedido_metodo_pagamento",
    "metodo_pagamento_grupo",
    "pedido_subtotal",
    "pedido_valor_desconto",
    "pedido_valor_total",
    "produto_base",
    "tipo_embalagem",
    "produto_com_multiplicador",
    "multiplicador",
    "qtd_unidades_base",
    "qtd_pacotes",
    "preco_unitario",
    "preco_pacote",
    "item_valor_total",
    "is_pedido_gratuito",
    "is_pedido_consolidado",
    "is_pedido_varejo",
)
