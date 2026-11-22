[← Etapa Anterior: Exportação e Inferência](7_exportacao_e_inferencia.md) | [Índice](README.md)

# Etapa 8 — Interface PDV, Testes, Repositório GitHub, Pitch e Cronograma

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Construir a interface funcional de demonstração consumindo o modelo exportado, consolidar testes/documentação no GitHub e preparar o pitch de 7 minutos dentro do prazo |
| **Módulos** | `app/streamlit_app.py`, `README.md` (raiz), `tests/` |
| **Entradas** | Pacote `models/recommender_v1/`, `reports/figures/`, `reports/evaluation_summary.csv` |
| **Saídas** | Aplicação Streamlit interativa, repositório GitHub reprodutível e roteiro ensaiado do pitch |
| **Referência** | [descricao_trabalho.pdf](../descricao_trabalho.pdf) |
| **Status** | `[ ] Planejado` |

---

## 1. Interface Funcional de Demonstração (`app/streamlit_app.py`)
Aplicação web interativa em **Streamlit**, consumindo **exclusivamente o modelo exportado** via `RecommenderService.from_artifact("models/recommender_v1/")`:
- **Aba 1 — Simulador de PDV de Feira (Demo ao Vivo):**
  - *Sidebar* com contexto da venda (Feira, Dia da Semana, Horário) e seletor de itens para montar o carrinho (SKU + quantidade de pacotes, exibindo subtotal atual em tempo real).
  - Painel principal exibindo **Top-3 Recomendações Completas** em cartões claros (`produto_base`, `embalagem/multiplicador`, `qtd_pacotes`, `qtd_unidades_base`, `preço`, barra de confiança e botão *"+ Adicionar ao Carrinho"* que recalcula dinamicamente as próximas recomendações).
  - Banner opcional de **Upsell de Embalagem** quando aplicável e indicador de impacto comercial simulado (*Novo Ticket Estimado* e *+R$ de acréscimo*).
- **Aba 2 — Auditoria e Inteligência das Feiras (EDA):**
  - Exibe a dor dos 79,0% de compra unitária, a reconciliação dos 139 pedidos da `ExpoDireto`, o mapa de co-ocorrência/Lift e o perfil por feira/horário.
- **Aba 3 — Laboratório da Rede Neural:**
  - Apresenta a arquitetura Multi-Task, as curvas de *Loss* (treino vs. validação), a tabela comparativa com baselines e Autoencoder, e os metadados do artefato exportado (`training_report.json`).

## 2. Suite de Testes Automatizados (`tests/`)
Execução unificada via `pytest -v`:
- `tests/test_preprocessing.py`: valida 0 erros de desconto, 0 erros de fechamento contábil nos 3.003 pedidos, 0 `item_pedido_quantidade <= 0`, remoção das 3 colunas inválidas e 0 `qtd_pacotes` fracionárias.
- `tests/test_features.py`: valida `qtd_pacotes * multiplicador == qtd_unidades_base`, `abs(preco_unitario * multiplicador - preco_pacote) < 0.02`, 20 produtos base e 34 SKUs normalizados.
- `tests/test_dataset_split.py`: garante interseção vazia de `pedido_id` entre `train`, `val` e `test` (zero *data leakage*).
- `tests/test_model_and_export.py`: valida dimensões das 3 cabeças, mascaramento da loss, salvamento $\rightarrow$ recarregamento (*round-trip*) com predições numericamente idênticas e garantia de que itens já no carrinho nunca aparecem nas recomendações.

## 3. Checklist de Entregáveis no GitHub ([descricao_trabalho.pdf](../descricao_trabalho.pdf))
- [ ] Repositório GitHub organizado e versionado (em caso de grupo de 2–3 alunos, garantir commits rastreáveis de **todos** os integrantes distribuídos por módulo: ex. *Integrante A: ETL/Features/EDA*, *Integrante B: Rede Neural/Tuning/Avaliação/Exportação*, *Integrante C: Interface Streamlit/Testes/README/Pitch*).
- [ ] `README.md` na raiz completo: problema/dor, abordagem técnica, diagrama da arquitetura, link para `docs/README.md`, instruções passo a passo de reprodução (`venv`, `pip install -r requirements.txt`, pipeline completo, `pytest`, `streamlit run`) e resultados/prints incorporados.
- [ ] Código-fonte modular e limpo em `src/`, `app/` e `tests/`.
- [ ] Artefato do modelo exportado em `models/recommender_v1/` (com comando reproduzível documentado para regenerar tudo em <2 min).
- [ ] Visualizações, logs e prints da interface em `reports/figures/`.

## 4. Roteiro do Pitch Final (Até 7 Minutos)

| Tempo | Bloco | Pontos-Chave |
| :---: | :--- | :--- |
| **0:00 – 1:00** | **1. Problema / Dor Real** | Dados reais de 16 feiras do RS (3.003 pedidos): **79% dos clientes levam apenas 1 produto**. Oportunidade clara de cross-sell inteligente no balcão. |
| **1:00 – 2:30** | **2. Dados e Engenharia** | Desafios reais do dataset (139 erros de preço de pacote na `ExpoDireto` corrigidos para 0, 58 nomes brutos normalizados para 34 SKUs / 20 bases, tratamento de pedidos consolidados). |
| **2:30 – 4:00** | **3. Solução com Rede Neural** | Por que **MLP Multi-Task (3 cabeças)** em PyTorch: prevê produto base + SKU/multiplicador + quantidade com loss composta mascarada, `AdamW`, `Dropout`/L2 e derivação 100% coerente; comparação contra Popularidade, Co-ocorrência e Denoising Autoencoder; exportação do modelo. |
| **4:00 – 5:45** | **4. Demo ao Vivo (PDV)** | Simulador Streamlit rodando o modelo exportado: montar carrinho real (ex.: *Alfajor Preto* na *Expointer*) $\rightarrow$ ver Top-3 recomendações completas e adicionar ao carrinho em tempo real. |
| **5:45 – 7:00** | **5. Resultados, Impacto e Próximos Passos** | Ganhos nas métricas Top-$K$ e cobertura sobre baselines, potencial de aumento no ticket médio e próximos passos (integração via API no PDV das feiras, recomendação por estoque disponível e retreinamento incremental após cada feira). |

## 5. Cronograma de Execução (22/11 até 23/11 às 12h00)

| Bloco | Período Sugerido | Entregas do Bloco | Critério de Pronto |
| :--- | :--- | :--- | :--- |
| **Bloco 1 — Setup + ETL + Features** | 22/11 (Manhã, ~2h) | `venv`, `requirements.txt`, `src/config.py`, `src/preprocessing.py`, `src/features.py`, `tests/test_preprocessing.py`, `tests/test_features.py` | `pytest` verde; 0 erros contábeis em 3.003 pedidos |
| **Bloco 2 — EDA + Cestas + Baselines** | 22/11 (Início Tarde, ~2h) | `src/eda.py`, gráficos em `reports/figures/`, `src/dataset_builder.py`, `src/baselines.py`, `tests/test_dataset_split.py` | 6 gráficos salvos; split sem leakage verificado; baselines registrados |
| **Bloco 3 — Redes Neurais + Tuning + Avaliação** | 22/11 (Fim Tarde, ~3h) | `src/models.py`, `src/losses.py`, `src/train.py`, `src/evaluate.py` (MLP Multi-Task + DAE + Optuna) | Curvas de treino sem overfitting severo; tabela comparativa final gerada |
| **Bloco 4 — Exportação + Interface Streamlit** | 22/11 (Noite, ~2,5h) | `src/export.py`, `src/inference.py`, `models/recommender_v1/`, `app/streamlit_app.py`, `tests/test_model_and_export.py` | `pytest` 100% verde; app Streamlit rodando fluído do modelo exportado |
| **Bloco 5 — README, Prints, Revisão e Pitch** | 23/11 (Manhã, até 11h00) | `README.md` raiz com prints/gráficos, teste limpo de reprodução, push final no GitHub e ensaio cronometrado do pitch (≤ 7 min) | Repositório validado e pitch ensaiado com folga antes das 12h00 |

---

## Critérios de Pronto (*Definition of Done*)
- [ ] `streamlit run app/streamlit_app.py` abre sem erros e executa recomendações reais a partir de `models/recommender_v1/`.
- [ ] `pytest -v` passa 100% em todos os 4 arquivos de teste.
- [ ] `README.md` na raiz validado passo a passo em ambiente limpo.
- [ ] Repositório GitHub atualizado e pitch ensaiado em até 7 minutos antes das 12h00 de 23/11/2026.
