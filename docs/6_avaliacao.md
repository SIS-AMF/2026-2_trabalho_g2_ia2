[← Etapa Anterior: Modelagem e Treino](5_modelagem_e_treino.md) | [Índice](README.md) | [Próxima Etapa: Exportação e Inferência →](7_exportacao_e_inferencia.md)

# Etapa 6 — Avaliação Quantitativa e Qualitativa

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Medir de forma imparcial no conjunto de **Teste (Hold-out)** a qualidade do ranking de produtos/embalagens, a diversidade das recomendações e a precisão da previsão de quantidades |
| **Módulos** | `src/evaluate.py` |
| **Testes** | Validação determinística do cálculo de métricas Top-$K$ e comparação completa |
| **Entradas** | Split de Teste (`data/processed/splits/`), baselines, `BasketDAE` e melhor checkpoint de `FairMultiTaskRecommender` |
| **Saídas** | `reports/evaluation_summary.csv`, `reports/figures/08_comparacao_modelos.png`, `reports/qualitative_cases.md` |
| **Status** | `[ ] Planejado` |

---

## 1. Métricas Avaliadas ($K \in \{1, 3, 5\}$)

| Grupo | Métrica | O Que Mede |
| :--- | :--- | :--- |
| **Ranking (Produto Base e SKU)** | `Hit Rate@K` | % de cestas em que pelo menos 1 item correto aparece no Top-$K$ |
| **Ranking (Produto Base e SKU)** | `Recall@K` e `Precision@K` | Cobertura e precisão dos itens corretos no Top-$K$ |
| **Ranking (Produto Base e SKU)** | `NDCG@K` e `MRR` | Qualidade da ordenação (premia colocar o item certo em 1º lugar) |
| **Diversidade / Personalização** | `Catalog Coverage@3` | % do catálogo efetivamente recomendado (detecta viés de popularidade) |
| **Quantidade (`qtd_pacotes`)** | `MAE` e `Acurácia Exata (%)` | Erro médio e acerto exato da quantidade de pacotes prevista nos acertos |

## 2. Comparação Consolidada de Modelos
Gerar tabela comparativa única (`reports/evaluation_summary.csv`) e gráfico de barras agrupadas (`reports/figures/08_comparacao_modelos.png`) comparando os 4 modelos sobre o mesmo conjunto de teste:
1. `Baseline 1 — Popularidade (Global / por Feira)`
2. `Baseline 2 — Co-ocorrência / Lift`
3. `BasketDAE (Denoising Autoencoder)`
4. `FairMultiTaskRecommender (MLP Multi-Task)`

Incluir ainda o resultado do experimento complementar **Leave-Last-Events-Out** (generalização para *Expo Silveira Martins* e *Festa do Moranguinho*).

## 3. Análise Qualitativa de Casos Reais
Gerar relatório (`reports/qualitative_cases.md`) com **5 a 8 exemplos reais de cestas do conjunto de teste**, exibindo lado a lado:
- **Entrada:** Carrinho parcial + Feira + Horário;
- **Saída da Rede Neural:** Top-3 Recomendações Completas (`produto_base`, `produto_com_multiplicador`, `multiplicador`, `qtd_pacotes`, `qtd_unidades_base`, `preco_unitario`, `preco_pacote`, `confianca`);
- **Realidade (Ground Truth):** item e quantidade efetivamente comprados pelo cliente.

---

## Critérios de Pronto (*Definition of Done*)
- [ ] Avaliação executada **uma única vez no conjunto de Teste final** (sem vazamento para escolha de hiperparâmetros).
- [ ] `reports/evaluation_summary.csv`, `reports/figures/08_comparacao_modelos.png` e `reports/qualitative_cases.md` gerados e coerentes.
- [ ] Evidência clara de comparação entre baselines estatísticos e redes neurais (incluindo análise de `Catalog Coverage@3` para confirmar personalização real).
