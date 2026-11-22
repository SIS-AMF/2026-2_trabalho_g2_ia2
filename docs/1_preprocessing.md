[← Voltar ao Índice](README.md) | [Próxima Etapa: Engenharia de Features →](2_features.md)

# Etapa 1 — Auditoria, Limpeza e Validação Contábil

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Transformar o CSV bruto em uma base auditada, sem inconsistências financeiras, sem ruídos de cadastro e com tipagem rigorosa |
| **Módulos** | `src/config.py`, `src/preprocessing.py` |
| **Testes** | `tests/test_preprocessing.py` |
| **Entrada** | [dataset.csv](../dataset.csv) (`3.815 linhas × 13 colunas`, `3.003 pedidos`) |
| **Saídas** | `data/processed/itens_limpos.parquet`, `reports/preprocessing_audit.json` |
| **Status** | `[ ] Planejado` |

---

## 1. Remoção de Colunas Inválidas (e Regra de Ordem)
1. **Remover `pedido_vendedor` e `item_pedido_user_id`:** ambas as colunas são idênticas em 100% das linhas, possuem apenas 2 UUIDs de login compartilhado entre operadores (além de 1 registro `NaN` na `1 Moranguinho`) e não representam identidade individual de vendedor.
2. **Remover `pedido_tipo_desconto` somente após a Seção 2 (Validação Automatizada de Descontos):** só descartar após provar programaticamente que todos os valores em `pedido_valor_desconto` já estão expressos em moeda (R$).

## 2. Validação Automatizada de Descontos
- **Achado empírico:** entre os `192` pedidos com desconto diferente de zero, `191` têm `pedido_tipo_desconto == 'R$'` e apenas **1 pedido** (`ca605a42-...`, `EXPOBENTO`, `subtotal = 6`, `desconto = 0.12`, `total = 5.88`) está marcado como `'%'` — porém `6.00 - 0.12 = 5.88`, provando que o valor já foi gravado em reais.
- **Regra implementada:**
  1. Verificar por pedido: `abs((pedido_subtotal - pedido_valor_desconto) - pedido_valor_total) < 0.01`.
  2. Se algum futuro registro vier de fato em porcentagem (`abs(subtotal * (1 - desc/100) - total) < 0.01` e `subtotal - desc != total`), converter automaticamente para moeda: `pedido_valor_desconto = subtotal * (desc / 100)`.
  3. Executar `assert` garantindo **0 violações** em todos os pedidos; em seguida, remover `pedido_tipo_desconto`.

## 3. Remoção de Itens com Quantidade Zero e Marcação de Pedidos Gratuitos
- **Itens com `item_pedido_quantidade <= 0`:** existem **2 linhas** no dataset (`ebbed809-...` na `Expo Afubra` e `4680d5c5-...` na `EXPOBENTO`), ambas representando itens cancelados/trocados por pacote no mesmo pedido. **Ação:** remover essas linhas e registrar no log de auditoria.
- **Pedidos gratuitos (`pedido_valor_total == 0`):** existem **2 pedidos** com desconto de 100% (`Alfajor Preto` R$ 6 na `ExpoDireto` e fechamento de R$ 915 em `São Francisco de Assis`). **Ação:** criar flag booleana `is_pedido_gratuito = (pedido_valor_total <= 0)` para rastreabilidade e filtragem na modelagem.

## 4. Saneamento Lexical, Pagamentos, Eventos e Sinônimos
1. **Nomes de produtos:** aplicar `.str.strip()` e colapsar múltiplos espaços internos (`re.sub(r"\s+", " ", s)`) — corrige `"Bolacha de manteiga  (Un)"`, `"Amendoim salgado e temperado "`, `"Pão branco "`, `"Bolacha de natal "`, etc.
2. **Sinônimos de produtos (mapa explícito em `src/config.py`):**
   - Unificar `"Amendoim salgado e temperado"` (Mar/2026, R$ 8,00) $\rightarrow$ `"Amendoim Temperado"` (Mai–Jun/2026, R$ 8,00), pois nunca co-ocorrem na mesma feira e correspondem à renomeação do mesmo produto.
   - Manter `"Amendoim Temperado 100g"` (Jul–Out/2026, R$ 6,00) como produto distinto (gramatura e preço diferentes).
3. **Eventos (`evento_nome`):**
   - Preservar coluna original em `evento_original` e extrair `ponto_venda` quando aplicável.
   - Unificar `"1 Moranguinho"` e `"2-Moranguinho"` (mesma data: 30/10 a 02/11/2026, dois caixas/barracas) $\rightarrow$ `evento_nome = "Festa do Moranguinho"`.
4. **Método de pagamento (`pedido_metodo_pagamento`):**
   - Padronizar capitalização (`"Pix"` e `"PIX"` $\rightarrow$ `"PIX"`).
   - Preservar granularidade em `pedido_metodo_pagamento` e criar versão agrupada `metodo_pagamento_grupo` (`Dinheiro`, `Cartão`, `PIX`, agrupando `Cartão`, `Cartão de Crédito` e `Cartão de Débito`).

## 5. Resolução Determinística de Multiplicadores e Reconciliação Contábil (`139 → 0` erros)
- **Anomalia descoberta na auditoria:**
  - Existem **139 pedidos** onde $\sum (\text{item\_pedido\_quantidade} \times \text{item\_pedido\_preco\_unitario}) \neq \text{pedido\_subtotal}$, **100% concentrados na `ExpoDireto`**. Neles, o PDV salvou a quantidade já em unidades individuais (`4, 6, 3, 16...`), porém salvou o **preço do pacote inteiro** (`20, 25, 80`) na coluna `item_pedido_preco_unitario`.
  - Adicionalmente, **11 linhas iniciais da `ExpoDireto`** (ex.: `Alfajor Preto (Pacote x4)`, `qtd=1`, `preco=20`) e todas as embalagens da **`1 Moranguinho`** (`(Pacote)`, `(Fardo)`) e **`(Pacote x1)`** gravaram a quantidade em **número de pacotes** junto do preço do pacote.

- **Algoritmo determinístico em 3 passos:**
  1. **Extração do `multiplicador` e `tipo_embalagem`:**
     - Se houver padrão `(Tipo xN)` (ex.: `(Pacote x6)`, `(Fardo x3)`, `(Bandeja x5)`, `(Caixa x16)`, `(Display x16)`): `multiplicador = N`, `tipo_embalagem = Tipo`.
       - **Exceção catalogada e verificada:** `Rapadura de Melado Grão Moído (Pacote x1)` a R$ 20,00 (16 ocorrências) $\rightarrow$ `multiplicador = 3` (preço idêntico ao `Pacote x3`: $3 \times \text{R\$ } 6{,}67 = \text{R\$ } 20{,}00$).
       - `Alfajor Preto/Branco (Display x1)` a R$ 5,00 $\rightarrow$ `multiplicador = 1`, `tipo_embalagem = "Display"`.
     - Se houver `(Fardo)` ou `(Pacote)` sem `xN` (exclusivo da `1 Moranguinho`), aplicar tabela validada por preço:

       | Produto Bruto (`1 Moranguinho`) | Preço Registrado | `tipo_embalagem` | `multiplicador` | Preço Unitário Resultante |
       | :--- | :---: | :---: | :---: | :---: |
       | `Cuca Alemã (Fardo)` | R$ 60,00 | `Fardo` | `3` | R$ 20,00 |
       | `Rapadura Assada (Pacote)` | R$ 25,00 | `Pacote` | `6` | R$ 4,17 |
       | `Rapadura de Melado (Pacote)` | R$ 20,00 | `Pacote` | `3` | R$ 6,67 |
       | `Alfajor Preto (Pacote)` | R$ 20,00 | `Pacote` | `4` | R$ 5,00 |
       | `Cuca Enrolada de Amendoim (Pacote)` | R$ 25,00 | `Pacote` | `2` | R$ 12,50 |
       | `Cuca Enrolada de Chocolate (Pacote)` | R$ 30,00 | `Pacote` | `2` | R$ 15,00 |

     - Se `(Un)` ou sem sufixo entre parênteses: `multiplicador = 1`, `tipo_embalagem = "Unidade"`.
  2. **Reconciliação de quantidade e preço por caso:**
     - **Caso A (`multiplicador == 1`):** `qtd_unidades_base = qty`, `qtd_pacotes = qty`, `preco_unitario = preco`, `preco_pacote = preco`.
     - **Caso B (pedido divergente na `ExpoDireto`, `sum_raw != subtotal` e `multiplicador > 1`):** `qty` já está em unidades base e `preco` é o preço do pacote $\rightarrow$ `qtd_unidades_base = qty`, `qtd_pacotes = qty / multiplicador`, `preco_unitario = preco / multiplicador`, `preco_pacote = preco`.
     - **Caso C (registro em pacotes com preço de pacote: 11 linhas não divergentes de pacote na `ExpoDireto`, `(Pacote)`/`(Fardo)` na `1 Moranguinho`, `(Pacote x1)` a R$ 20):** `qtd_unidades_base = qty * multiplicador`, `qtd_pacotes = qty`, `preco_unitario = preco / multiplicador`, `preco_pacote = preco`.
     - **Caso D (padrão normalizado das demais feiras):** `qty` já está em unidades base e `preco` é o preço unitário $\rightarrow$ `qtd_unidades_base = qty`, `qtd_pacotes = qty / multiplicador`, `preco_unitario = preco`, `preco_pacote = preco * multiplicador`.
  3. **Validação contábil obrigatória (invariantes):**
     - `assert` que `abs(sum(qtd_unidades_base * preco_unitario) - pedido_subtotal) < 0.01` em **100% dos 3.003 pedidos** (já validado empiricamente: **0 erros em 3.003 pedidos**).
     - `assert` que `qtd_pacotes` é inteiro exato (`abs(qtd_pacotes - round(qtd_pacotes)) < 1e-6`) em **100% das 3.813 linhas válidas** (já validado: **0 linhas fracionárias**).

## 6. Sinalização de Pedidos Consolidados / Fechamento de Caixa (Outliers)
- **Achado empírico:** ~16 pedidos apresentam comportamento de lançamento consolidado/fechamento em lote ou atacado (ex.: `Feira Polêsine` possui apenas 3 pedidos no total, entre 14h06 e 14h22, com até 565 unidades e R$ 3.696; `ExpoSerra` R$ 3.150; `Wolksfest Agudo` R$ 1.462 às 21h31; `São Francisco de Assis` R$ 915 com desconto 100% às 23h09; `Feira Paraíso` lançamentos consecutivos às 21h44–21h53).
- **Regra:** criar `is_pedido_consolidado = (pedido_subtotal >= 200) | (qtd_total_unidades_pedido >= 40) | (evento_original.isin(["Feira Polêsine", "ExpoSerra"]))` e `is_pedido_varejo = (~is_pedido_consolidado) & (~is_pedido_gratuito)`.
- **Política de uso:** todos os registros permanecem no dataset auditado (`itens_limpos.parquet`) para transparência total, mas o treinamento e a avaliação do recomendador de balcão utilizam `is_pedido_varejo == True` (~2.986 pedidos), documentando o impacto antes/depois.

## 7. Tipagem Forte do Schema Final

| Coluna | Tipo Final | Descrição |
| :--- | :--- | :--- |
| `pedido_id` | `string` | Identificador único do pedido (UUID) |
| `pedido_data` | `datetime64[ns, America/Sao_Paulo]` | Timestamp padronizado com fuso `-03:00` |
| `evento_nome`, `evento_original`, `ponto_venda` | `category` | Feira unificada, nome bruto e ponto de venda |
| `pedido_metodo_pagamento`, `metodo_pagamento_grupo` | `category` | Meio de pagamento padronizado e agrupado |
| `pedido_subtotal`, `pedido_valor_desconto`, `pedido_valor_total` | `float64` | Valores monetários validados em R$ |
| `produto_base`, `tipo_embalagem`, `produto_com_multiplicador` | `category` | Taxonomia normalizada do item |
| `multiplicador`, `qtd_unidades_base`, `qtd_pacotes` | `int64` | Quantidades fisicamente consistentes |
| `preco_unitario`, `preco_pacote`, `item_valor_total` | `float64` | Preços reconciliados em R$ |
| `is_pedido_gratuito`, `is_pedido_consolidado`, `is_pedido_varejo` | `bool` | Flags de auditoria e escopo de modelagem |

---

## Critérios de Pronto (*Definition of Done*)
- [ ] `0` erros de validação de desconto nos `3.003` pedidos e remoção confirmada de `pedido_tipo_desconto`, `pedido_vendedor` e `item_pedido_user_id`.
- [ ] `0` linhas com `item_pedido_quantidade <= 0` (`3.813` itens válidos mantidos).
- [ ] `0` pedidos com erro contábil (`abs(sum(qtd_unidades_base * preco_unitario) - pedido_subtotal) >= 0.01`).
- [ ] `0` linhas com `qtd_pacotes` fracionária; schema final exatamente conforme a Seção 7.
- [ ] `pytest tests/test_preprocessing.py` passando 100% e `reports/preprocessing_audit.json` gerado.
