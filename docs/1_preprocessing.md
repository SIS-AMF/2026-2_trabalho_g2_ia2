[← Voltar ao Índice](README.md) | [Próxima Etapa: Engenharia de Features →](2_features.md)

# Etapa 1 — Auditoria, Limpeza e Validação Contábil

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Transformar o CSV bruto em uma base auditada via **funções puras matemáticas**, sem inconsistências financeiras, sem unir produtos distintos e com tipagem rigorosa |
| **Módulos & Notebooks** | `src/config.py`, `src/preprocessing.py`, `notebooks/01_preprocessamento_e_features.ipynb` |
| **Testes** | `tests/test_preprocessing.py` |
| **Entrada** | [dataset.csv](../dataset.csv) (`3.815 linhas × 13 colunas`, `3.003 pedidos`) |
| **Saídas** | `data/processed/itens_limpos.parquet`, `reports/preprocessing_audit.json` |
| **Status** | `[x] Concluído` |

---

## 0. Princípio Arquitetural: Funções Puras + Transparência via Notebook
- **Funções puras sem dependências externas (`src/preprocessing.py`):** toda regra de validação e cálculo é escrita como uma operação matemática isolada — recebe parâmetros específicos (ex.: `subtotal, desconto, total` ou `nome_limpo, produto_base, preco`) e retorna um valor determinístico, sem depender de estado global ou colunas ocultas de DataFrame.
  - Exemplos de funções puras:
    - `validar_desconto_em_moeda(subtotal: float, valor_desconto: float, valor_total: float, tol: float = 0.01) -> tuple[bool, float]`
    - `normalizar_texto_produto(nome_bruto: str) -> str`
    - `extrair_produto_base(nome_limpo: str, mapa_sinonimos: dict[str, str]) -> str`
    - `extrair_embalagem_e_multiplicador(nome_limpo: str, produto_base: str, preco_bruto: float) -> tuple[str, int]`
    - `reconciliar_quantidade_e_preco(qtd_bruta: int, preco_bruto: float, multiplicador: int, pedido_divergente: bool, preco_eh_pacote: bool) -> tuple[int, int, float, float]`
    - `classificar_pedido_consolidado(subtotal: float, qtd_total_unidades: int, evento_original: str) -> bool`
- **Orquestração:** uma função de alto nível (`executar_preprocessamento(df_bruto)`) apenas itera/aplica essas funções puras em sequência, registrando contagens antes/depois.
- **Inspeção Didática (`notebooks/01_preprocessamento_e_features.ipynb`):** executa cada passo importando as funções de `src/preprocessing.py` e imprime amostras, contadores e validações passo a passo para total clareza.

## 1. Remoção de Colunas Inválidas (e Regra de Ordem)
1. **Remover `pedido_vendedor` e `item_pedido_user_id`:** ambas as colunas são idênticas em 100% das linhas, possuem apenas 2 UUIDs de login compartilhado entre operadores (além de 1 registro `NaN` na `1 Moranguinho`) e não representam identidade individual de vendedor.
2. **Remover `pedido_tipo_desconto` somente após a Seção 2 (Validação Automatizada de Descontos):** só descartar após provar programaticamente que todos os valores em `pedido_valor_desconto` já estão expressos em moeda (R$).

## 2. Validação Automatizada de Descontos
- **Achado empírico:** entre os `192` pedidos com desconto diferente de zero, `191` têm `pedido_tipo_desconto == 'R$'` e apenas **1 pedido** (`ca605a42-...`, `EXPOBENTO`, `subtotal = 6`, `desconto = 0.12`, `total = 5.88`) está marcado como `'%'` — porém `6.00 - 0.12 = 5.88`, provando que o valor já foi gravado em reais.
- **Regra implementada via função pura `validar_desconto_em_moeda`:**
  1. Verificar por pedido: `abs((subtotal - valor_desconto) - valor_total) < 0.01`.
  2. Se algum registro vier de fato em porcentagem (`abs(subtotal * (1 - desc/100) - total) < 0.01` e `subtotal - desc != total`), converter automaticamente para moeda: `subtotal * (desc / 100)`.
  3. Executar `assert` garantindo **0 violações** em todos os pedidos; em seguida, remover `pedido_tipo_desconto`.

## 3. Remoção de Itens com Quantidade Zero e Marcação de Pedidos Gratuitos
- **Itens com `item_pedido_quantidade <= 0`:** existem **2 linhas** no dataset (`ebbed809-...` na `Expo Afubra` e `4680d5c5-...` na `EXPOBENTO`), ambas representando itens cancelados/trocados por pacote no mesmo pedido. **Ação:** remover essas linhas e registrar no log de auditoria.
- **Pedidos gratuitos (`pedido_valor_total == 0`):** existem **2 pedidos** com desconto de 100% (`Alfajor Preto` R$ 6 na `ExpoDireto` e fechamento de R$ 915 em `São Francisco de Assis`). **Ação:** criar flag booleana `is_pedido_gratuito = (pedido_valor_total <= 0)` para rastreabilidade e filtragem na modelagem.

## 4. Saneamento Lexical, Proteção Contra Fusão Indevida e Papel de `evento_nome`
1. **Nomes de produtos (`normalizar_texto_produto`):** aplicar `.strip()` e colapsar múltiplos espaços internos (`re.sub(r"\s+", " ", s)`) — corrige `"Bolacha de manteiga  (Un)"`, `"Amendoim salgado e temperado "`, `"Pão branco "`, `"Bolacha de natal "`, etc.
2. **Proteção estrita contra unir produtos distintos (mapa explícito em `src/config.py`):**
   - **Único sinônimo confirmado pelo negócio:** unificar `"Amendoim salgado e temperado"` (Mar/2026, R$ 8,00) $\rightarrow$ `"Amendoim Temperado"` (Mai–Jun/2026, R$ 8,00), confirmado como **o mesmo produto com troca de embalagem**.
   - **Todos os demais 19 produtos permanecem 100% separados** (ex.: `"Amendoim Temperado 100g"` a R$ 6,00 separado; `"Rapadura de Melado"` separado de `"Rapadura de Melado Grão Moído"`; `"Alfajor Preto"` separado de `"Alfajor Branco"`; todas as cucas, bolachas e pães separados). Nenhuma fusão automática por similaridade textual é permitida.
3. **Eventos (`evento_nome`):**
   - Preservar `evento_original` e unificar `"1 Moranguinho"` + `"2-Moranguinho"` $\rightarrow$ `"Festa do Moranguinho"` (com `ponto_venda`) **para fins de auditoria contábil e EDA**.
   - **Decisão de modelagem:** `evento_nome` **não entra como feature** na rede neural principal (modelo 100% genérico para novas feiras, combinado a um filtro pós-processamento de estoque/disponibilidade no Top-$K$).
4. **Método de pagamento (`pedido_metodo_pagamento`):**
   - Padronizar capitalização (`"Pix"` e `"PIX"` $\rightarrow$ `"PIX"`) e criar `metodo_pagamento_grupo` (`Dinheiro`, `Cartão`, `PIX`).

## 5. Resolução Determinística de Multiplicadores e Reconciliação Contábil (`139 → 0` erros)
- **Anomalia descoberta na auditoria:**
  - Existem **139 pedidos** onde $\sum (\text{item\_pedido\_quantidade} \times \text{item\_pedido\_preco\_unitario}) \neq \text{pedido\_subtotal}$, **100% concentrados na `ExpoDireto`**. Neles, o PDV salvou a quantidade já em unidades individuais (`4, 6, 3, 16...`), porém salvou o **preço do pacote inteiro** (`20, 25, 80`) na coluna `item_pedido_preco_unitario`.
  - Adicionalmente, **11 linhas iniciais da `ExpoDireto`** (ex.: `Alfajor Preto (Pacote x4)`, `qtd=1`, `preco=20`) e todas as embalagens da **`1 Moranguinho`** (`(Pacote)`, `(Fardo)`) e **`(Pacote x1)`** gravaram a quantidade em **número de pacotes** junto do preço do pacote.

- **Algoritmo determinístico em 3 passos (via funções puras):**
  1. **`extrair_embalagem_e_multiplicador(nome_limpo, produto_base, preco_bruto)`:**
     - Padrão `(Tipo xN)`: `multiplicador = N`, `tipo_embalagem = Tipo`.
       - **Exceção verificada:** `Rapadura de Melado Grão Moído (Pacote x1)` a R$ 20,00 (16 ocorrências) $\rightarrow$ `multiplicador = 3` ($3 \times \text{R\$ } 6{,}67 = \text{R\$ } 20{,}00$).
       - `Alfajor Preto/Branco (Display x1)` a R$ 5,00 $\rightarrow$ `multiplicador = 1`, `tipo_embalagem = "Display"`.
     - Padrão `(Fardo)` ou `(Pacote)` sem `xN` (`1 Moranguinho`), validado por preço:

       | Produto Bruto (`1 Moranguinho`) | Preço Registrado | `tipo_embalagem` | `multiplicador` | Preço Unitário Resultante |
       | :--- | :---: | :---: | :---: | :---: |
       | `Cuca Alemã (Fardo)` | R$ 60,00 | `Fardo` | `3` | R$ 20,00 |
       | `Rapadura Assada (Pacote)` | R$ 25,00 | `Pacote` | `6` | R$ 4,17 |
       | `Rapadura de Melado (Pacote)` | R$ 20,00 | `Pacote` | `3` | R$ 6,67 |
       | `Alfajor Preto (Pacote)` | R$ 20,00 | `Pacote` | `4` | R$ 5,00 |
       | `Cuca Enrolada de Amendoim (Pacote)` | R$ 25,00 | `Pacote` | `2` | R$ 12,50 |
       | `Cuca Enrolada de Chocolate (Pacote)` | R$ 30,00 | `Pacote` | `2` | R$ 15,00 |

     - Padrão `(Un)` ou sem parênteses: `multiplicador = 1`, `tipo_embalagem = "Unidade"`.
  2. **`reconciliar_quantidade_e_preco(...)`:**
     - **Caso A (`multiplicador == 1`):** `qtd_unidades_base = qty`, `qtd_pacotes = qty`, `preco_unitario = preco`, `preco_pacote = preco`.
     - **Caso B (pedido divergente na `ExpoDireto`, `sum_raw != subtotal` e `multiplicador > 1`):** `qtd_unidades_base = qty`, `qtd_pacotes = qty / multiplicador`, `preco_unitario = preco / multiplicador`, `preco_pacote = preco`.
     - **Caso C (registro em pacotes com preço de pacote: 11 linhas não divergentes de pacote na `ExpoDireto`, `(Pacote)`/`(Fardo)` na `1 Moranguinho`, `(Pacote x1)` a R$ 20):** `qtd_unidades_base = qty * multiplicador`, `qtd_pacotes = qty`, `preco_unitario = preco / multiplicador`, `preco_pacote = preco`.
     - **Caso D (padrão normalizado das demais feiras):** `qtd_unidades_base = qty`, `qtd_pacotes = qty / multiplicador`, `preco_unitario = preco`, `preco_pacote = preco * multiplicador`.
  3. **Invariantes contábeis obrigatórias:**
     - `assert` que `abs(sum(qtd_unidades_base * preco_unitario) - pedido_subtotal) < 0.01` em **100% dos 3.003 pedidos** (**0 erros**).
     - `assert` que `qtd_pacotes` é inteiro exato em **100% das 3.813 linhas válidas** (**0 linhas fracionárias**).

## 6. Sinalização de Pedidos Consolidados / Fechamento de Caixa (Outliers)
- **Achado empírico:** ~16 pedidos apresentam comportamento de lançamento consolidado/fechamento em lote ou atacado (`Feira Polêsine` com apenas 3 pedidos no total, até 565 unidades / R$ 3.696; `ExpoSerra` R$ 3.150; `Wolksfest Agudo` R$ 1.462 às 21h31; `São Francisco de Assis` R$ 915 com desconto 100% às 23h09; `Feira Paraíso` lançamentos consecutivos às 21h44–21h53).
- **Regra (função pura `classificar_pedido_consolidado`):** `is_pedido_consolidado = (pedido_subtotal >= 200) | (qtd_total_unidades_pedido >= 40) | (evento_original in {"Feira Polêsine", "ExpoSerra"})` e `is_pedido_varejo = (~is_pedido_consolidado) & (~is_pedido_gratuito)`.
- **Política de uso:** mantidos em `itens_limpos.parquet` para auditoria; filtrados (`is_pedido_varejo == True`, `2.984` pedidos) para o recomendador de balcão.

## 7. Tipagem Forte do Schema Final

| Coluna | Tipo Final | Descrição |
| :--- | :--- | :--- |
| `pedido_id` | `string` | Identificador único do pedido (UUID) |
| `pedido_data` | `datetime64[ns, America/Sao_Paulo]` | Timestamp padronizado com fuso `-03:00` |
| `evento_nome`, `evento_original`, `ponto_venda` | `category` | Metadados da feira (usados na auditoria e EDA) |
| `pedido_metodo_pagamento`, `metodo_pagamento_grupo` | `category` | Meio de pagamento padronizado e agrupado |
| `pedido_subtotal`, `pedido_valor_desconto`, `pedido_valor_total` | `float64` | Valores monetários validados em R$ |
| `produto_base`, `tipo_embalagem`, `produto_com_multiplicador` | `category` | Taxonomia normalizada do item |
| `multiplicador`, `qtd_unidades_base`, `qtd_pacotes` | `int64` | Quantidades fisicamente consistentes |
| `preco_unitario`, `preco_pacote`, `item_valor_total` | `float64` | Preços reconciliados em R$ |
| `is_pedido_gratuito`, `is_pedido_consolidado`, `is_pedido_varejo` | `bool` | Flags de auditoria e escopo de modelagem |

---

## Critérios de Pronto (*Definition of Done*)
- [x] Todas as validações implementadas como **funções puras unitárias** com cobertura em `tests/test_preprocessing.py`.
- [x] `0` erros de validação de desconto nos `3.003` pedidos e remoção confirmada de `pedido_tipo_desconto`, `pedido_vendedor` e `item_pedido_user_id`.
- [x] `0` linhas com `item_pedido_quantidade <= 0` (`3.813` itens válidos mantidos) e nenhuma fusão indevida de produtos distintos.
- [x] `0` pedidos com erro contábil e `0` linhas com `qtd_pacotes` fracionária; schema final exatamente conforme a Seção 7.
- [x] `data/processed/itens_limpos.parquet`, `reports/preprocessing_audit.json` e `notebooks/01_preprocessamento_e_features.ipynb` validados; `pytest tests/test_preprocessing.py` 100% verde.
