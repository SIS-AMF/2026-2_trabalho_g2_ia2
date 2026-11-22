[← Etapa Anterior: EDA](3_eda.md) | [Índice](README.md) | [Próxima Etapa: Modelagem e Treino →](5_modelagem_e_treino.md)

# Etapa 4 — Formulação do Problema, Cestas e Split Anti-Leakage

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Transformar as cestas limpas de varejo em exemplos supervisionados `(X_carrinho_parcial, X_contexto) -> (Y_base, Y_sku, Y_qtd_pacotes)` garantindo zero *data leakage* |
| **Módulos** | `src/dataset_builder.py` |
| **Testes** | `tests/test_dataset_split.py` |
| **Entradas** | `data/processed/pedidos_cestas.parquet`, `models/recommender_v1/catalog_metadata.json` |
| **Saídas** | `data/processed/splits/` (índices fixos por `pedido_id`, tensores e pré-processadores ajustados no treino) |
| **Status** | `[ ] Planejado` |

---

## 1. Política Rigorosa Anti-Leakage
1. **Split primeiro, expansão depois:** a divisão dos dados ocorre **estritamente no nível de `pedido_id`** (nunca por linha de item nem por par mascarado):
   - **Treino:** 70% dos pedidos de varejo (`is_pedido_varejo == True`)
   - **Validação:** 15% dos pedidos de varejo
   - **Teste (Hold-out final):** 15% dos pedidos de varejo
   - Estratificação combinada por `is_multi_item` (cesta com 1 vs. $\ge 2$ SKUs) e `evento_nome` (agrupando eventos muito pequenos).
2. **Fit de scalers/encoders apenas no Treino:** `StandardScaler` (para features numéricas do carrinho parcial) e mapas de frequência são ajustados exclusivamente sobre o conjunto de treino.
3. **Validação complementar temporal/por evento (*Leave-Last-Events-Out*):** experimento secundário treinando nas feiras de Mar–Set/2026 e avaliando nas feiras finais de Out–Nov/2026 (*Expo Silveira Martins* e *Festa do Moranguinho*) para medir generalização em novas feiras.

## 2. Estratégia de Geração de Amostras (Mascaramento de Cestas)
Para o cenário principal (**Cross-Sell com Carrinho Ativo**), cada cesta multi-SKU $B = \{(s_1, q_1), \dots, (s_k, q_k)\}$ ($k \ge 2$) gera amostras de treino e avaliação:
- **No Treino (*data augmentation* estrutural sem leakage):**
  - *Leave-One-Out:* para cada SKU $s_i \in B$, cria-se um exemplo com entrada $B \setminus \{s_i\}$ e alvo $\{s_i\}$ (~1.000 pares a partir dos ~440 pedidos multi-SKU de treino).
  - *Subconjuntos de entrada (para cestas com $k \ge 3$):* amostrar subconjuntos próprios não vazios $S \subset B$ (ex.: $|S| = 1$) prevendo todos os itens faltantes $B \setminus S$ simultaneamente — ensinando a rede a recomendar tanto quando há 1 item quanto quando há 2+ itens no carrinho.
  - *Exemplos de Cold-Start (carrinho vazio, ~15% dos batches de treino):* pedidos de 1 ou mais itens com carrinho zerado e contexto ativo (`evento_nome`, `hora`, `dia_semana`), permitindo que a mesma rede também recomende produtos de destaque antes do primeiro item ser adicionado.
- **Na Validação e no Teste:** protocolo determinístico *Leave-One-Out* sobre todas as cestas multi-SKU do respectivo split (garantindo comparação justa e reprodutível entre todos os modelos e baselines).

## 3. Especificação do Vetor de Entrada $X$ e Alvos $Y$
- **Entrada $X$ (`dim ≈ 114`):**
  - `x_cart_sku_bin` (`34`): presença binária de cada SKU no carrinho parcial;
  - `x_cart_sku_qty` (`34`): `log1p(qtd_pacotes)` de cada SKU no carrinho parcial;
  - `x_cart_base_bin` (`20`): presença binária de cada `produto_base` no carrinho parcial;
  - `x_cart_stats` (`5`): `qtd_skus_carrinho`, `log1p(qtd_pacotes_total)`, `log1p(qtd_unidades_total)`, `log1p(valor_parcial)`, `tem_pacote_no_carrinho`;
  - `x_context` (`~21`): `evento_nome` (one-hot com slot `"OUTRO"`), `hora_sin`, `hora_cos`, `faixa_horaria` (one-hot), `fim_de_semana`.
- **Alvos $Y$ (Decomposição Multi-Task):**
  - `y_base` (`20`, binário): `produto_base` dos itens mascarados;
  - `y_sku` (`34`, binário): `produto_com_multiplicador` (`sku_norm`) dos itens mascarados;
  - `y_qty` (`34`, contínuo): `log1p(qtd_pacotes)` nos índices onde `y_sku == 1` (e `0` nos demais, ignorados via máscara na loss);
  - `cart_mask_sku` (`34`) e `cart_mask_base` (`20`): máscaras binárias indicando itens já presentes no carrinho.

---

## Critérios de Pronto (*Definition of Done*)
- [ ] Interseção estritamente vazia de `pedido_id` entre `train`, `val` e `test`.
- [ ] Nenhum item presente em `x_cart_sku_bin == 1` aparece simultaneamente como alvo positivo em `y_sku == 1`.
- [ ] `StandardScaler` e encoders ajustados apenas sobre o conjunto de treino e serializáveis para a Etapa 7.
- [ ] `pytest tests/test_dataset_split.py` passando 100%.
