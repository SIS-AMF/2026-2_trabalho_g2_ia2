[← Etapa Anterior: Pré-processamento](1_preprocessing.md) | [Índice](README.md) | [Próxima Etapa: EDA →](3_eda.md)

# Etapa 2 — Engenharia de Features e Catálogo Oficial

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Formalizar features de produto, embalagem, preço, contexto temporal genérico e perfil de cesta, consolidando a tabela de cestas e o catálogo oficial de SKUs |
| **Módulos & Notebooks** | `src/utils/temporal.py`, `src/data/features.py`, `notebooks/01_preprocessamento_e_features.ipynb` |
| **Testes** | `tests/test_features.py` |
| **Entrada** | `data/processed/itens_limpos.parquet` |
| **Saídas** | `data/processed/pedidos_cestas.parquet`, `data/processed/pedidos_cestas.csv`, `models/recommender_v1/catalog_metadata.json` |
| **Status** | `[x] Concluído` |

---

## 1. Features de Produto, Embalagem e Preço (Nível de Item)
Todas as features são calculadas com funções puras e invariantes matemáticas:
1. **`produto_base`** (`20 classes`): nome do produto sem sufixo de embalagem/multiplicador (ex.: `"Rapadura de Melado"`).
2. **`multiplicador`** (`int >= 1`): quantidade de unidades base contidas em 1 embalagem (ex.: `3`).
3. **`produto_com_multiplicador`** (`sku_norm`, `34 classes`): chave canônica `f"{produto_base} (x{multiplicador})"` + rótulo amigável para exibição (ex.: `"Rapadura de Melado (Pacote x3)"`). Reduz a fragmentação de 58 nomes brutos para **34 SKUs limpos e coerentes**.
4. **`qtd_unidades_base`** (`int >= 1`): quantidade total em unidades individuais (ex.: 2 pacotes de 3 $\rightarrow$ `6`).
5. **`qtd_pacotes`** (`int >= 1`): quantidade de embalagens compradas (`qtd_unidades_base // multiplicador`, ex.: `6 // 3 = 2`).
6. **`preco_unitario`** (`float`): preço efetivo pago por unidade base (ex.: `R$ 6,67` no pacote x3 de `R$ 20,00`).
7. **`preco_pacote`** (`float`): preço efetivo de 1 embalagem (`preco_unitario * multiplicador`, ex.: `R$ 20,00`).
8. **`economia_unitaria_pct`** (`float`): percentual de desconto da embalagem frente ao preço unitário avulso do mesmo `produto_base`.

## 2. Features Contextuais Genéricas (Nível de Pedido)
Extraídas de `pedido_data` e do estado dinâmico do carrinho — **sem dependência de `evento_nome`**, garantindo generalização imediata para qualquer nova feira:
- **Temporais genéricas (`6` features numéricas na entrada da rede):**
  - `hora_sin` e `hora_cos` (codificação cíclica contínua da hora `0–23`);
  - `faixa_horaria` (one-hot: `Manhã <12h`, `Tarde 12h–17h`, `Noite >=18h`);
  - `fim_de_semana` (`0` ou `1`, sábado/domingo).
  - *(Colunas auxiliares `hora`, `dia_semana`, `mes` e `evento_nome` são preservadas no `.parquet` apenas para EDA e auditoria.)*
- **Perfil do Carrinho Parcial (computado dinamicamente durante treino e inferência — sem vazamento do alvo!):**
  - `qtd_itens_distintos_carrinho` (quantos SKUs já estão no carrinho);
  - `qtd_total_pacotes_carrinho` (`log1p`);
  - `qtd_total_unidades_carrinho` (`log1p`);
  - `valor_parcial_carrinho` (`log1p` da soma dos itens já no carrinho);
  - `tem_pacote_no_carrinho` (`0` ou `1`: indica se o cliente já levou alguma embalagem com `multiplicador > 1`).

## 3. Consolidação por Cesta e Catálogo Oficial
- **Agregação por `(pedido_id, produto_com_multiplicador)`:** soma `qtd_pacotes` e `qtd_unidades_base` caso o mesmo SKU apareça repetido (garantia preventiva após normalização) e preserva pedidos onde o mesmo `produto_base` foi levado em 2 embalagens distintas (33 pedidos no dataset).
- **Geração do Catálogo Oficial (`catalog_metadata.json`):** mapeia cada um dos **34 `sku_norm`** para:
  ```json
  {
    "sku_id": 3,
    "sku_norm": "Rapadura Assada (x6)",
    "produto_base": "Rapadura Assada",
    "base_id": 15,
    "tipo_embalagem": "Pacote",
    "display_name": "Rapadura Assada (Pacote x6)",
    "multiplicador": 6,
    "preco_unitario_ref": 4.17,
    "preco_pacote_ref": 25.00,
    "frequencia_varejo": 297
  }
  ```
  Esse catálogo alimenta simultaneamente o **pós-processamento determinístico da rede neural** e o **catálogo mocado do frontend React + Vite (`front/`)**.

---

## Critérios de Pronto (*Definition of Done*)
- [x] Invariante `qtd_pacotes * multiplicador == qtd_unidades_base` válida em 100% dos itens.
- [x] Invariante `abs(preco_unitario * multiplicador - preco_pacote) < 0.02` válida em 100% dos itens.
- [x] Exatamente `20` classes canônicas de `produto_base` e `34` classes de `produto_com_multiplicador` no catálogo oficial.
- [x] `data/processed/pedidos_cestas.parquet` e `catalog_metadata.json` gerados sem valores nulos nas colunas de modelagem.
- [x] `pytest tests/test_features.py` passando 100%.
