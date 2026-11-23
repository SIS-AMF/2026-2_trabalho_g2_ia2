[← Etapa Anterior: Exportação e Inferência](7_exportacao_e_inferencia.md) | [Índice](README.md)

# Etapa 8 — Interface PDV (React + Vite), Testes, Repositório GitHub, Pitch e Cronograma

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Construir a interface funcional de PDV em `front/` (React + Vite + TypeScript) conectada à API FastAPI do modelo exportado, consolidar a suite de testes/documentação no GitHub e preparar o pitch de 7 minutos |
| **Módulos** | `front/` (React + Vite + TS), `api/main.py`, `README.md` (raiz), `tests/` |
| **Entradas** | API FastAPI (`models/recommender_v1/`), `reports/figures/`, `reports/evaluation_summary.csv` |
| **Saídas** | PDV React + Vite funcional com **Venda Simulada + Modal de Recomendação IA (Top-3)**, repositório GitHub reprodutível e pitch ensaiado |
| **Referência** | [descricao_trabalho.pdf](../descricao_trabalho.pdf) |
| **Status** | `[x] Concluído` |

---

## 1. Interface Funcional em React + Vite (`front/src/`)

A interface em `front/src/` segue o layout operacional de PDV de feira (paleta artesanal cacau/creme, catálogo à esquerda, carrinho à direita e modais de apresentação, estoque, recomendação e sucesso):

### 1.1 Componentes Implementados em `front/`
1. **Aplicação Standalone React + Vite + TypeScript:** `front/package.json`, `vite.config.ts`, `index.html`, `src/main.tsx` e `src/App.tsx` sem dependências externas de Supabase ou pagamento real.
2. **Catálogo Mocado Dinâmico (34 SKUs):** carregar os produtos a partir de `GET /api/catalog` (com *fallback* em JSON estático sincronizado com `catalog_metadata.json`), incluindo opção simples de marcar itens como **"Sem estoque"** para demonstrar ao vivo o pós-processamento de estoque pulando para o próximo item do ranking!
3. **Fluxo de Venda Simulada:** remover a integração real de maquininha/pagamento (Point/Mercado Pago), mantendo uma finalização rápida e simulada para demonstração no pitch.
4. **Novo Modal `PDVRecommendationModal` (disparado ao clicar em "Finalizar Venda"):**
   - Envia o carrinho atual (+ hora/dia + lista `available_skus` em estoque) para `POST /api/recommend` (FastAPI $\rightarrow$ modelo PyTorch exportado).
   - Abre o modal exibindo o **Top-3 de Recomendações** em cartões claros contendo:
     - **`produto_com_multiplicador`** (`display_name`) e **`produto_base`**;
     - **Embalagem / `multiplicador`**, **`qtd_pacotes`** e **`qtd_unidades_base`**;
     - **`preco_unitario`**, **`preco_pacote`** e acréscimo no subtotal;
     - **Confiança da IA (%)** (probabilidade estimada pelo modelo para aquele carrinho) + **selo discreto da Taxa de Acerto real do modelo (`Hit Rate@3`)** medida no conjunto de teste 20%;
     - Botão **"+ Adicionar ao Carrinho"** em cada cartão (insere o produto no carrinho e permite atualizar o total/sugestões);
     - Botão **"Concluir Venda Simulada"** $\rightarrow$ encerra a venda exibindo o `PDVSuccessModal` com o resumo final e o ganho de ticket gerado pela recomendação!

## 2. Suite de Testes Automatizados (`tests/`)
Execução unificada via `pytest -v`:
- `tests/test_preprocessing.py`: testa cada **função pura** isoladamente + integridade global (`0` erros de desconto, `0` erros contábeis nos `3.003` pedidos, `0` `item_pedido_quantidade <= 0`, remoção das 3 colunas inválidas, `0` `qtd_pacotes` fracionárias e zero fusão indevida de produtos).
- `tests/test_features.py`: valida `qtd_pacotes * multiplicador == qtd_unidades_base`, `abs(preco_unitario * multiplicador - preco_pacote) < 0.02`, `20` produtos base e `34` SKUs normalizados.
- `tests/test_dataset_split.py`: garante interseção vazia de `pedido_id` entre Desenvolvimento (80%) e Teste (20%), expansão $2^k - 2$ sem duplicatas e `dim = 99`.
- `tests/test_model_and_export.py`: valida dimensões das 3 cabeças, mascaramento da loss, salvamento $\rightarrow$ recarregamento (*round-trip*), garantia de exclusão de itens já no carrinho, **pós-processamento de estoque (`available_skus`)** e endpoints FastAPI.

## 3. Checklist de Entregáveis no GitHub ([descricao_trabalho.pdf](../descricao_trabalho.pdf))
- [ ] Repositório GitHub organizado e versionado (em caso de grupo de até 3 alunos, garantir commits rastreáveis de **todos** os integrantes).
- [ ] `README.md` na raiz completo: problema/dor, abordagem técnica, diagrama da arquitetura, link para `docs/README.md`, instruções passo a passo de reprodução (Python `venv` + pipeline + `pytest` + `uvicorn api.main:app` + `cd front && npm install && npm run dev`) e resultados/prints incorporados.
- [ ] Código-fonte modular e limpo em `src/`, `api/`, `front/`, `notebooks/` e `tests/`.
- [ ] Artefato do modelo exportado em `models/recommender_v1/` (com comando reproduzível para regenerar).
- [ ] Visualizações, notebooks executados e prints do PDV + Modal Top-3 em `reports/figures/`.

## 4. Roteiro do Pitch Final (Até 7 Minutos)

| Tempo | Bloco | Pontos-Chave |
| :---: | :--- | :--- |
| **0:00 – 1:00** | **1. Problema / Dor Real** | Dados reais de 16 feiras do RS (3.003 pedidos): **79% dos clientes levam apenas 1 produto**. Oportunidade clara de cross-sell inteligente no balcão para qualquer feira. |
| **1:00 – 2:15** | **2. Dados e Engenharia** | Funções puras de auditoria: `139` erros de preço de pacote na `ExpoDireto` corrigidos para `0`, `58` nomes brutos normalizados para `34` SKUs / `20` produtos base e tratamento de pedidos consolidados. |
| **2:15 – 3:45** | **3. Solução com Rede Neural** | **MLP Multi-Task Genérico (3 cabeças)** em PyTorch: split 80/20 + expansão $2^k - 2$, prevê produto base + SKU/multiplicador + quantidade com loss composta mascarada, `AdamW`, `Dropout`/L2 e derivação 100% coerente; filtro pós-processamento de estoque no Top-3; comparação contra Popularidade, Co-ocorrência e Denoising Autoencoder; exportação do modelo + API FastAPI. |
| **3:45 – 5:45** | **4. Demo ao Vivo (PDV React + Vite)** | Montar um carrinho no PDV real adaptado $\rightarrow$ clicar em **"Finalizar Venda"** $\rightarrow$ abrir o **Modal Top-3 da IA** com confiança (%) e detalhes completos $\rightarrow$ adicionar item recomendado e concluir venda simulada (e mostrar rapidamente o filtro pulando item sem estoque!). |
| **5:45 – 7:00** | **5. Resultados, Impacto e Próximos Passos** | Métricas reais no Hold-Out 20% (`Hit Rate@3`, `NDCG@3`, `Catalog Coverage@3`, `MAE` de quantidade), ganho potencial de ticket médio e próximos passos (teste comparativo com/sem evento, integração direta ao estoque real do Supabase no PDV e retreinamento incremental após cada feira). |

## 5. Cronograma de Execução (22/11 12h00 até 23/11 às 12h00)

| Bloco | Período Sugerido | Entregas do Bloco | Critério de Pronto |
| :--- | :--- | :--- | :--- |
| **Bloco 1 — Setup + ETL (Funções Puras) + Features** | 22/11 (12h00–14h00) | `venv`, `requirements.txt`, `src/config.py`, `src/utils/cleaning.py`, `src/utils/temporal.py`, `src/data/preprocessing.py`, `src/data/features.py`, `notebooks/01_*.ipynb`, `tests/test_preprocessing.py`, `tests/test_features.py` | `pytest` verde; 0 erros contábeis em 3.003 pedidos |
| **Bloco 2 — EDA + Cestas ($2^k-2$) + Split 80/20 + Baselines** | 22/11 (14h00–16h00) | `src/data/eda.py`, 6 gráficos em `reports/figures/`, `notebooks/02_*.ipynb`, `src/utils/baskets.py`, `src/data/dataset_builder.py`, `src/modeling/baselines.py`, `tests/test_dataset_split.py` | 6 gráficos salvos; split 80/20 sem leakage; baselines registrados |
| **Bloco 3 — Redes Neurais + Tuning + Avaliação (20%)** | 22/11 (16h00–18h30) | `src/modeling/models.py`, `src/modeling/losses.py`, `src/modeling/train.py`, `src/utils/ranking.py`, `src/modeling/evaluate.py`, `notebooks/03_*.ipynb` (MLP Multi-Task Genérico + DAE + Optuna) | Curvas de treino sem overfitting severo; tabela comparativa final gerada |
| **Bloco 4 — Exportação + API FastAPI + Frontend React/Vite** | 22/11 (19h00–22h30) | `src/serving/export.py`, `src/serving/inference.py`, `src/pipeline.py`, `api/main.py`, `models/recommender_v1/`, `front/` adaptado com **Modal Top-3** e venda simulada, `tests/test_model_and_export.py` | `pytest` 100% verde; PDV React consumindo FastAPI com Modal Top-3 funcionando ao vivo |
| **Bloco 5 — README, Prints, Revisão e Pitch** | 23/11 (08h00–11h00) | `README.md` raiz completo com prints/gráficos, teste limpo de reprodução, push final no GitHub e ensaio cronometrado do pitch (≤ 7 min) *(+ ablação com/sem evento apenas se sobrar tempo!)* | Repositório validado e pitch ensaiado com folga antes das 12h00 |

---

## Critérios de Pronto (*Definition of Done*)
- [x] `uvicorn api.main:app --port 8000` + `cd front && npm run dev` rodando juntos sem erros, com o **Modal Top-3** respondendo ao vivo ao clicar em *"Finalizar Venda"*.
- [x] `pytest -v` passando 100% em todos os 4 arquivos de teste.
- [x] `README.md` na raiz + notebooks `.ipynb` validados passo a passo.
- [ ] Repositório GitHub atualizado e pitch ensaiado em até 7 minutos antes das 12h00 de 23/11/2026.
