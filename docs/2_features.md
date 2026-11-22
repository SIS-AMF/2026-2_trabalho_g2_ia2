[← Etapa Anterior: Pré-processamento](1_preprocessing.md) | [Índice](README.md) | [Próxima Etapa: EDA →](3_eda.md)

# Etapa 2 — Engenharia de Features e Catálogo Oficial

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Formalizar features de produto, embalagem, preço, contexto temporal e perfil de cesta, consolidando a tabela de cestas e o catálogo oficial de SKUs |
| **Módulos** | `src/features.py` |
| **Testes** | `tests/test_features.py` |
| **Entrada** | `data/processed/itens_limpos.parquet` |
| **Saídas** | `data/processed/pedidos_cestas.parquet`, `models/recommender_v1/catalog_metadata.json` |
| **Status** | `[ ] Planejado` |

---

## 1. Features de Produto, Embalagem e Preço (Nível de Item)
Todas as features previstas no escopo inicial são formalizadas com invariantes matemáticas:
1. **`produto_base`** (`20 classes`): nome do produto sem sufixo de embalagem/multiplicador (ex.: `"Rapadura de Melado"`).
2. **`multiplicador`** (`int >= 1`): quantidade de unidades base contidas em 1 embalagem (ex.: `3`).
3. **`produto_com_multiplicador`** (`sku_norm`, `34 classes`): chave canônica `f"{produto_base} (x{multiplicador})"` + rótulo amigável para exibição (ex.: `"Rapadura de Melado (Pacote x3)"`). Reduz a fragmentação de 58 nomes brutos para **34 SKUs limpos e coerentes**.
4. **`qtd_unidades_base`** (`int >= 1`): quantidade total em unidades individuais (ex.: se o cliente comprou 2 pacotes de 3, `qtd_unidades_base = 6`).
5. **`qtd_pacotes`** (`int >= 1`): quantidade de embalagens compradas (`qtd_unidades_base // multiplicador`, ex.: `6 // 3 = 2`).
6. **`preco_unitario`** (`float`): preço efetivo pago por unidade base (ex.: `R$ 6,67` no pacote x3 de `R$ 20,00`).
7. **`preco_pacote`** (`float`): preço efetivo de 1 embalagem (`preco_unitario * multiplicador`, ex.: `R$ 20,00`).
8. **`economia_unitaria_pct`** (`float`): percentual de desconto da embalagem frente ao preço unitário avulso do mesmo `produto_base` (captura atratividade promocional do pacote).

## 2. Features Contextuais (Nível de Pedido)
Extraídas de `pedido_data`, `evento_nome` e do estado do pedido:
- **Temporais:** `hora` (`0–23`), `hora_sin` e `hora_cos` (codificação cíclica contínua), `faixa_horaria` (`Manhã <12h`, `Tarde 12h–17h`, `Noite >=18h`), `dia_semana` (`0–6`), `fim_de_semana` (`bool`), `mes` (`3–11`).
- **Contexto da Feira:** `evento_nome` (one-hot/embedding com suporte à categoria `"OUTRO"` para feiras inéditas em produção).
- **Perfil do Carrinho Parcial (computado dinamicamente durante treino e inferência — sem vazamento do alvo!):**
  - `qtd_itens_distintos_carrinho` (quantos SKUs já estão no carrinho);
  - `qtd_total_pacotes_carrinho` e `qtd_total_unidades_carrinho`;
  - `valor_parcial_carrinho` (`log1p` da soma dos itens já no carrinho);
  - `tem_pacote_no_carrinho` (`bool`: indica se o cliente já demonstrou propensão a levar embalagens múltiplas).

## 3. Consolidação por Cesta e Catálogo Oficial
- **Agregação por `(pedido_id, produto_com_multiplicador)`:** soma `qtd_pacotes` e `qtd_unidades_base` caso um mesmo SKU apareça repetido (0 ocorrências brutas, garantia preventiva após normalização) e preserva pedidos onde o mesmo `produto_base` foi levado em 2 embalagens distintas (33 pedidos no dataset).
- **Geração do Catálogo Oficial (`catalog_metadata.json`):** mapeia cada um dos **34 `sku_norm`** para `{sku_id, produto_base, base_id, tipo_embalagem, display_name, multiplicador, preco_unitario_ref, preco_pacote_ref, frequencia_varejo}`, garantindo que o modelo nunca produza uma combinação inexistente.

---

## Critérios de Pronto (*Definition of Done*)
- [ ] Invariante `qtd_pacotes * multiplicador == qtd_unidades_base` válida em 100% dos itens.
- [ ] Invariante `abs(preco_unitario * multiplicador - preco_pacote) < 0.02` válida em 100% dos itens.
- [ ] Exatamente `20` classes canônicas de `produto_base` e `34` classes de `produto_com_multiplicador` no catálogo oficial.
- [ ] `data/processed/pedidos_cestas.parquet` e `catalog_metadata.json` gerados sem valores nulos nas colunas de modelagem.
- [ ] `pytest tests/test_features.py` passando 100%.
