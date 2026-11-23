[← Etapa Anterior: Engenharia de Features](2_features.md) | [Índice](README.md) | [Próxima Etapa: Dataset e Split →](4_dataset_e_split.md)

# Etapa 3 — Análise Exploratória (EDA) e Evidências Visuais

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Gerar evidências quantitativas e 6 gráficos reproduzíveis que comprovam a dor de negócio, validam a limpeza e sustentam o `README.md`, os notebooks `.ipynb` e o *pitch* |
| **Módulos & Notebooks** | `src/data/eda.py`, `notebooks/02_eda_e_cestas.ipynb` |
| **Testes** | Validação de geração dos 6 arquivos de imagem e resumo estatístico |
| **Entradas** | `data/processed/itens_limpos.parquet`, `data/processed/pedidos_cestas.parquet`, `reports/preprocessing_audit.json` |
| **Saídas** | `reports/figures/01_*.png` .. `06_*.png`, `reports/eda_summary.json` |
| **Status** | `[x] Concluído` |

---

## 1. Gráficos Obrigatórios (`reports/figures/`)

| # | Arquivo de Saída | Pergunta de Negócio / Evidência Comprovada |
| :-: | :--- | :--- |
| **1** | `01_distribuicao_itens_por_pedido.png` | Evidencia a dor principal: **79,0% dos pedidos** possuem apenas 1 produto base (`78,9%` 1 SKU no varejo) vs. **21,1%** multi-SKU |
| **2** | `02_auditoria_expodireto_antes_depois.png` | Comprova o rigor da limpeza contábil: **139 pedidos divergentes na `ExpoDireto` $\rightarrow$ 0 erros** |
| **3** | `03_top_produtos_e_skus.png` | Ranking dos 20 produtos base (volume vs. faturamento) e participação de embalagens (unitário vs. pacotes `x3`, `x4`, `x6`) |
| **4** | `04_heatmap_coocorrencia_lift.png` | Matriz de co-ocorrência e **Lift** entre produtos base (ex.: *Cuca Alemã $\leftrightarrow$ Cuca Enrolada de Amendoim*, *Alfajor Preto $\leftrightarrow$ Rapadura Assada*, *Alfajor Branco $\leftrightarrow$ Alfajor Preto*) |
| **5** | `05_vendas_por_feira_e_horario.png` | Perfil de demanda por feira e faixa horária — evidencia também diferenças de catálogo disponível entre feiras, justificando o modelo genérico + filtro de estoque |
| **6** | `06_distribuicao_qtd_pacotes_e_outliers.png` | Mostra a distribuição `80,7% (1) / 15,6% (2) / 3,7% (3+)` no balcão vs. os ~16 pedidos consolidados filtrados |

## 2. Padrão Visual e Reuso
- Salvar todos os gráficos em alta resolução (`300 DPI`, fundo limpo, rótulos em português e títulos autoexplicativos) para incorporação direta no `README.md` raiz, em `notebooks/02_eda_e_cestas.ipynb` e nos slides do *pitch*.
- Salvar estatísticas-chave (números exatos citados no texto e no *pitch*) em `reports/eda_summary.json` para evitar divergências numéricas entre documentação, API e apresentação.

---

## Critérios de Pronto (*Definition of Done*)
- [x] Execução limpa via comando único (`python -m src.data.eda`) e exibição verificada em `notebooks/02_eda_e_cestas.ipynb`.
- [x] Todos os 6 arquivos PNG gerados em `reports/figures/` sem texto cortado ou sobreposto.
- [x] `reports/eda_summary.json` consistente com os achados documentados na Etapa 1 e Etapa 2.
