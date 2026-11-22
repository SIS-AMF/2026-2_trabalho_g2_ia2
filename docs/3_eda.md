[← Etapa Anterior: Engenharia de Features](2_features.md) | [Índice](README.md) | [Próxima Etapa: Dataset e Split →](4_dataset_e_split.md)

# Etapa 3 — Análise Exploratória (EDA) e Evidências Visuais

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Gerar evidências quantitativas e 6 gráficos reproduzíveis que comprovam a dor de negócio, validam a limpeza e sustentam o `README.md` e o *pitch* |
| **Módulos** | `src/eda.py` |
| **Testes** | Validação de geração dos 6 arquivos de imagem e resumo estatístico |
| **Entradas** | `data/processed/itens_limpos.parquet`, `data/processed/pedidos_cestas.parquet`, `reports/preprocessing_audit.json` |
| **Saídas** | `reports/figures/01_*.png` .. `06_*.png`, `reports/eda_summary.json` |
| **Status** | `[ ] Planejado` |

---

## 1. Gráficos Obrigatórios (`reports/figures/`)

| # | Arquivo de Saída | Pergunta de Negócio / Evidência Comprovada |
| :-: | :--- | :--- |
| **1** | `01_distribuicao_itens_por_pedido.png` | Evidencia a dor principal: **79,0% dos pedidos** possuem apenas 1 produto base vs. **21,0%** multi-item |
| **2** | `02_auditoria_expodireto_antes_depois.png` | Comprova o rigor da limpeza contábil: **139 pedidos divergentes na `ExpoDireto` $\rightarrow$ 0 erros** |
| **3** | `03_top_produtos_e_skus.png` | Ranking dos 20 produtos base (volume vs. faturamento) e participação de embalagens (unitário vs. pacotes `x3`, `x4`, `x6`) |
| **4** | `04_heatmap_coocorrencia_lift.png` | Matriz de co-ocorrência e **Lift** entre produtos base (ex.: *Cuca Alemã $\leftrightarrow$ Cuca Enrolada de Amendoim*, *Alfajor Preto $\leftrightarrow$ Rapadura Assada*, *Alfajor Branco $\leftrightarrow$ Alfajor Preto*) |
| **5** | `05_vendas_por_feira_e_horario.png` | Perfil de demanda por feira (`evento_nome`) e por faixa horária (justifica o uso de features contextuais na rede neural) |
| **6** | `06_distribuicao_qtd_pacotes_e_outliers.png` | Mostra a distribuição `80,7% (1) / 15,6% (2) / 3,7% (3+)` no balcão vs. os ~16 pedidos consolidados filtrados |

## 2. Padrão Visual e Reuso
- Salvar todos os gráficos em alta resolução (`300 DPI`, fundo limpo, rótulos em português e títulos autoexplicativos) para incorporação direta no `README.md` raiz, na Aba 2 do app Streamlit e nos slides do *pitch*.
- Salvar estatísticas-chave (números exatos citados no texto e no *pitch*) em `reports/eda_summary.json` para evitar números divergentes entre relatórios e interface.

---

## Critérios de Pronto (*Definition of Done*)
- [ ] Execução limpa via comando único (`python -m src.eda`).
- [ ] Todos os 6 arquivos PNG gerados em `reports/figures/` sem texto cortado ou sobreposto.
- [ ] `reports/eda_summary.json` consistente com os achados documentados na Etapa 1 e Etapa 2.
