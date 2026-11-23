[← Etapa Anterior: EDA](3_eda.md) | [Índice](README.md) | [Próxima Etapa: Modelagem e Treino →](5_modelagem_e_treino.md)

# Etapa 4 — Formulação do Problema, Cestas e Split 80/20 Anti-Leakage

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Transformar as cestas limpas de varejo em exemplos supervisionados multi-label `(X_carrinho_parcial, X_tempo) -> (Y_base, Y_sku, Y_qtd_pacotes)` usando **split 80/20** por `pedido_id` e expansão por **subconjuntos únicos ($2^k - 2$)** sem *data leakage* |
| **Módulos & Notebooks** | `src/utils/baskets.py`, `src/data/dataset_builder.py`, `notebooks/02_eda_e_cestas.ipynb` |
| **Testes** | `tests/test_dataset_split.py` |
| **Entradas** | `data/processed/pedidos_cestas.parquet`, `models/recommender_v1/catalog_metadata.json` |
| **Saídas** | `data/processed/splits/` (índices fixos 80/20 por `pedido_id`, sub-split interno de validação, tensores e `StandardScaler` ajustado no treino) |
| **Status** | `[x] Concluído` |

---

## 1. Política Rigorosa Anti-Leakage (Split 80/20 por `pedido_id`)
1. **Split primeiro, expansão depois:** a divisão dos `629` pedidos multi-SKU de varejo (e demais pedidos para cold-start/popularidade) ocorre **estritamente no nível de `pedido_id`**, antes de gerar qualquer subconjunto mascarado:
   - **Desenvolvimento (80%):** `~503` pedidos multi-SKU.
   - **Teste Hold-Out Final (20%):** `~126` pedidos multi-SKU — mantido **intocado** até a Etapa 6.
   - Estratificação pelo tamanho da cesta (`k = 2` vs. `k >= 3`).
2. **Validação Interna (dentro dos 80% de Desenvolvimento):**
   - Para *Early Stopping* e busca de hiperparâmetros (Optuna) **sem vazar o Teste (20%)**, aplicamos um sub-split estratificado interno (ex.: `85% treino / 15% validação` dentro dos 80%, equivalente a `68% / 12% / 20%`, ou `5-Fold CV` por `pedido_id` no Optuna).
   - O `StandardScaler` das 5 estatísticas numéricas do carrinho parcial é ajustado **exclusivamente** no conjunto de treino.

## 2. Expansão por Subconjuntos Próprios Únicos ($2^k - 2$)
No carrinho de feira, **a ordem dos produtos não importa** (`{A, B}` gera exatamente o mesmo vetor de entrada que `{B, A}`). Por isso, expandimos cada pedido multi-SKU de tamanho $k \ge 2$ usando **todos os subconjuntos próprios não vazios ($2^k - 2$)** em vez de permutações ($k!$), evitando duplicatas idênticas em pedidos com $k \ge 4$:
- Pedido com **2 itens `{A, B}`** $\rightarrow 2^2 - 2 = \mathbf{2}$ amostras: `{A}→{B}`, `{B}→{A}`.
- Pedido com **3 itens `{A, B, C}`** $\rightarrow 2^3 - 2 = \mathbf{6}$ amostras: `{A}→{B,C}`, `{B}→{A,C}`, `{C}→{A,B}`, `{A,B}→{C}`, `{A,C}→{B}`, `{B,C}→{A}`.
- Pedido com **4 itens** $\rightarrow 2^4 - 2 = \mathbf{14}$ amostras (todas únicas).
- **Volume total nos 629 pedidos multi-SKU de varejo:** `1.998` amostras únicas (`~1.600` no bloco de 80% Desenvolvimento).
- **Protocolo no Teste (20%):** avaliação determinística via **Leave-One-Out ($k$ amostras por pedido)** sobre todos os `~126` pedidos multi-SKU do Teste (`~280` casos de teste), garantindo métrica limpa e comparável entre baselines e redes neurais.

## 3. Decisões Fundamentadas de Formulação
1. **Treinamento Multi-Label (1 passada = Top-3) em vez de rodar a rede 2×:**
   - Como os exemplos $2^k - 2$ contêm alvos simultâneos (ex.: `{A} → {B, C}`) e a rede possui `Sigmoid` independente por SKU, **uma única execução (*forward pass*)** ranqueia todos os 34 SKUs e entrega o **Top-3** pronto para o modal de finalização, sem propagar erros (*exposure bias*).
   - Se o cliente clicar em **"+ Adicionar"** em um item sugerido no modal, o carrinho muda e a API pode ser chamada novamente para atualizar as próximas recomendações.
2. **Por que NÃO criar uma classe `"NAO_RECOMENDAR"` com os 2.355 pedidos de 1 item:**
   - No histórico das feiras não havia sistema de recomendação; logo, **ausência de compra ≠ rejeição**.
   - Misturar 78,9% de exemplos com rótulo `"NAO_RECOMENDAR"` induziria a rede a prever "nada" quase sempre para acertar ~79% de forma trivial, destruindo o ranking de cross-sell.
   - Os pedidos de 1 item são aproveitados de forma saudável na **EDA**, no **Baseline de Popularidade** e opcionalmente no modo **Cold-Start** (carrinho vazio $\rightarrow$ destaques por faixa horária), enquanto a **confiança (%)** da rede sinaliza naturalmente a força de cada recomendação no modal.
3. **Modelo Genérico (sem `evento_nome` na entrada):**
   - Garante operação imediata em qualquer feira nova e evita que a rede memorize faltas de estoque históricas de feiras passadas. A disponibilidade real de cada feira é aplicada via **pós-processamento de estoque no Top-$K$** (ver [5_modelagem_e_treino.md](5_modelagem_e_treino.md) e [7_exportacao_e_inferencia.md](7_exportacao_e_inferencia.md)).

## 4. Especificação do Vetor de Entrada $X$ (`dim = 99`) e Alvos $Y$
- **Entrada $X$ (`99` floats `float32` — modelo genérico sem `evento_nome`):**
  - `x_cart_sku_bin` (`34`): presença binária de cada SKU no carrinho parcial;
  - `x_cart_sku_qty` (`34`): `log1p(qtd_pacotes)` de cada SKU no carrinho parcial;
  - `x_cart_base_bin` (`20`): presença binária de cada `produto_base` no carrinho parcial;
  - `x_cart_stats` (`5`, padronizado com `StandardScaler`): `qtd_skus_carrinho`, `log1p(qtd_pacotes_total)`, `log1p(qtd_unidades_total)`, `log1p(valor_parcial)`, `tem_pacote_no_carrinho`;
  - `x_time` (`6`): `hora_sin`, `hora_cos`, `faixa_horaria` (one-hot: 3 colunas), `fim_de_semana` (1 coluna).
- **Alvos $Y$ (Decomposição Multi-Task):**
  - `y_base` (`20`, binário): `produto_base` dos itens mascarados;
  - `y_sku` (`34`, binário): `produto_com_multiplicador` (`sku_norm`) dos itens mascarados;
  - `y_qty` (`34`, contínuo): `log1p(qtd_pacotes)` nos índices onde `y_sku == 1` (`0` nos demais, ignorados via máscara na loss);
  - `cart_mask_sku` (`34`) e `cart_mask_base` (`20`): máscaras binárias indicando itens já presentes no carrinho.

---

## Critérios de Pronto (*Definition of Done*)
- [x] Interseção estritamente vazia de `pedido_id` entre Desenvolvimento (80%) e Teste (20%), e entre treino e validação internos.
- [x] Expansão $2^k - 2$ validada sem duplicatas (`k=2 → 2`, `k=3 → 6`, `k=4 → 14`) e com `x_cart_sku_bin & y_sku == 0` em 100% das amostras.
- [x] Vetor $X$ com dimensão exata `99` e `StandardScaler` ajustado exclusivamente no treino.
- [x] `pytest tests/test_dataset_split.py` passando 100%.
