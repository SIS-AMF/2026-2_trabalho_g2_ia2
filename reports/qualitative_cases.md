# Análise Qualitativa — Exemplos Reais no Conjunto de Teste (20%)

Amostra de cestas reais do conjunto de teste Hold-Out avaliadas pelo `FairMultiTaskRecommender` (Modelo Genérico).

## Caso 1 — Pedido `27defda6...` (Sáb, 13h) — ACERTO NO TOP-3
- **Carrinho de Entrada:** **Cuca Enrolada de Amendoim (Unidade)** (2 pct)
- **Item Real Comprado (Ground Truth):** `Cuca Alemã (Unidade)` (mult `x1`, `1` pct = `1` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Cuca Alemã (Unidade) **[REAL]** | Cuca Alemã | x1 | 1 | 1 | R$ 25.00 | 67.8% |
| 2 | Rapadura de Melado (Unidade) | Rapadura de Melado | x1 | 1 | 1 | R$ 7.00 | 31.7% |
| 3 | Rapadura Assada (Unidade) | Rapadura Assada | x1 | 1 | 1 | R$ 5.00 | 28.0% |

## Caso 2 — Pedido `29ef7221...` (Sáb, 20h) — ACERTO NO TOP-3
- **Carrinho de Entrada:** **Cuca Enrolada de Amendoim (Unidade)** (2 pct)
- **Item Real Comprado (Ground Truth):** `Cuca Alemã (Unidade)` (mult `x1`, `2` pct = `2` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Cuca Alemã (Unidade) **[REAL]** | Cuca Alemã | x1 | 1 | 1 | R$ 25.00 | 62.0% |
| 2 | Rapadura de Melado (Unidade) | Rapadura de Melado | x1 | 1 | 1 | R$ 7.00 | 32.1% |
| 3 | Rapadura Assada (Unidade) | Rapadura Assada | x1 | 1 | 1 | R$ 5.00 | 29.6% |

## Caso 3 — Pedido `31784711...` (Sex, 19h) — ACERTO NO TOP-3
- **Carrinho de Entrada:** **Rapadura Assada (Pacote x6)** (1 pct)
- **Item Real Comprado (Ground Truth):** `Alfajor Preto (Unidade)` (mult `x1`, `1` pct = `1` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Alfajor Preto (Unidade) **[REAL]** | Alfajor Preto | x1 | 1 | 1 | R$ 6.00 | 44.2% |
| 2 | Cuca Enrolada de Amendoim (Unidade) | Cuca Enrolada de Amendoim | x1 | 1 | 1 | R$ 15.00 | 40.6% |
| 3 | Rapadura de Melado (Pacote x3) | Rapadura de Melado | x3 | 1 | 3 | R$ 20.00 | 38.7% |

## Caso 4 — Pedido `39f3a47e...` (Sex, 19h) — ACERTO NO TOP-3
- **Carrinho de Entrada:** **Cuca Enrolada de Amendoim (Unidade)** (1 pct)
- **Item Real Comprado (Ground Truth):** `Cuca Alemã (Unidade)` (mult `x1`, `1` pct = `1` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Cuca Alemã (Unidade) **[REAL]** | Cuca Alemã | x1 | 1 | 1 | R$ 25.00 | 50.9% |
| 2 | Rapadura de Melado (Unidade) | Rapadura de Melado | x1 | 1 | 1 | R$ 7.00 | 35.4% |
| 3 | Rapadura Assada (Pacote x6) | Rapadura Assada | x6 | 1 | 6 | R$ 25.00 | 29.2% |

## Caso 5 — Pedido `0bc6773e...` (Qua, 13h) — ACERTO NO TOP-3
- **Carrinho de Entrada:** **Rapadura Assada (Unidade)** (1 pct)
- **Item Real Comprado (Ground Truth):** `Cuca Enrolada de Amendoim (Unidade)` (mult `x1`, `1` pct = `1` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Rapadura de Melado (Unidade) | Rapadura de Melado | x1 | 1 | 1 | R$ 7.00 | 40.0% |
| 2 | Cuca Enrolada de Amendoim (Unidade) **[REAL]** | Cuca Enrolada de Amendoim | x1 | 1 | 1 | R$ 15.00 | 34.1% |
| 3 | Alfajor Preto (Unidade) | Alfajor Preto | x1 | 1 | 1 | R$ 6.00 | 27.2% |

## Caso 6 — Pedido `2340a331...` (Qui, 12h) — ACERTO NO TOP-3
- **Carrinho de Entrada:** **Alfajor Preto (Unidade)** (2 pct)
- **Item Real Comprado (Ground Truth):** `Alfajor Branco (Unidade)` (mult `x1`, `2` pct = `2` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Rapadura Assada (Unidade) | Rapadura Assada | x1 | 1 | 1 | R$ 5.00 | 43.3% |
| 2 | Alfajor Branco (Unidade) **[REAL]** | Alfajor Branco | x1 | 1 | 1 | R$ 6.00 | 39.8% |
| 3 | Rapadura de Melado (Unidade) | Rapadura de Melado | x1 | 1 | 1 | R$ 7.00 | 39.1% |

## Caso 7 — Pedido `063384d8...` (Sex, 19h) — FORA DO TOP-3
- **Carrinho de Entrada:** **Cri-Cri 200g (Unidade)** (1 pct)
- **Item Real Comprado (Ground Truth):** `Alfajor Preto (Unidade)` (mult `x1`, `1` pct = `1` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Amendoim Temperado (Unidade) | Amendoim Temperado | x1 | 1 | 1 | R$ 8.00 | 42.4% |
| 2 | Rapadura de Melado (Unidade) | Rapadura de Melado | x1 | 1 | 1 | R$ 7.00 | 39.1% |
| 3 | Cuca Enrolada de Amendoim (Unidade) | Cuca Enrolada de Amendoim | x1 | 1 | 1 | R$ 15.00 | 37.6% |

## Caso 8 — Pedido `0c6f54dc...` (Sex, 20h) — FORA DO TOP-3
- **Carrinho de Entrada:** **Rapadura Assada (Pacote x6)** (1 pct)
- **Item Real Comprado (Ground Truth):** `Cri-Cri 200g (Unidade)` (mult `x1`, `1` pct = `1` un.)

| Rank | Produto Recomendado (SKU) | Produto Base | Mult | Qtd Pct | Qtd Un | Preço Pct | Confiança |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | Alfajor Preto (Unidade) | Alfajor Preto | x1 | 1 | 1 | R$ 6.00 | 45.1% |
| 2 | Cuca Enrolada de Amendoim (Unidade) | Cuca Enrolada de Amendoim | x1 | 1 | 1 | R$ 15.00 | 41.4% |
| 3 | Rapadura de Melado (Pacote x3) | Rapadura de Melado | x3 | 1 | 3 | R$ 20.00 | 39.6% |
