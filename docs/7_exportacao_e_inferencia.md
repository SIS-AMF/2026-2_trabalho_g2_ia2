[← Etapa Anterior: Avaliação](6_avaliacao.md) | [Índice](README.md) | [Próxima Etapa: Interface e Entrega →](8_interface_e_entrega.md)

# Etapa 7 — Exportação Versionada do Modelo, Serviço de Inferência e API FastAPI

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Empacotar o modelo treinado e seus artefatos em formato versionado, implementar o `RecommenderService` com **filtro de estoque Top-$K$** e expor uma **API REST simples em FastAPI** para servir o frontend React + Vite |
| **Módulos** | `src/export.py`, `src/inference.py`, `api/main.py` |
| **Testes** | `tests/test_model_and_export.py` (*round-trip save → load → predict*, filtro de estoque e endpoints da API) |
| **Entradas** | Melhor checkpoint de `FairMultiTaskRecommender`, `StandardScaler` do treino, `catalog_metadata.json`, métricas do teste |
| **Saídas** | Pacote `models/recommender_v1/` + API REST `api/main.py` |
| **Status** | `[x] Concluído` |

---

## 1. Anatomia do Pacote Exportado (`models/recommender_v1/`)
1. **`model_weights.pt`:** pesos treinados (`model.state_dict()`) + dicionário de hiperparâmetros arquiteturais (`in_dim=99`, `hidden_dims`, `dropout`, `n_bases=20`, `n_skus=34`, `lambda_hier`) para reconstrução exata do módulo PyTorch.
2. **`model_scripted.pt` *(opcional/bônus)*:** versão exportada via `torch.jit.trace` / `torch.jit.script` para portabilidade desacoplada da classe Python.
3. **`preprocessing_artifacts.joblib`:** `StandardScaler` das 5 estatísticas do carrinho e dicionários `base_to_idx` / `sku_to_idx`.
4. **`catalog_metadata.json`:** tabela oficial dos 34 SKUs (`produto_base`, `tipo_embalagem`, `multiplicador`, preços de referência e `display_name`).
5. **`training_report.json`:** data/hora, seed, hiperparâmetros vencedores, histórico de épocas e métricas reais no Teste 20% (incluindo `hit_rate_at_3` e `ndcg_at_3`).

## 2. Contrato do Serviço de Inferência (`src/inference.py`)
Interface Python `RecommenderService.from_artifact("models/recommender_v1/")` com suporte nativo ao **filtro pós-processamento de estoque (`available_skus`)**:

```python
service.recommend(
    cart_items=[{"sku_norm": "Alfajor Preto (x1)", "qtd_pacotes": 2}],
    hora=15,
    dia_semana=5,
    top_k=3,
    available_skus=None,  # Opcional: lista de SKUs com estoque > 0; itens fora dela são pulados no ranking!
)
```

Retorno tipado e ordenado (100% consistente com o catálogo):

```json
[
  {
    "rank": 1,
    "sku_norm": "Rapadura Assada (x6)",
    "produto_base": "Rapadura Assada",
    "produto_com_multiplicador": "Rapadura Assada (Pacote x6)",
    "tipo_embalagem": "Pacote",
    "multiplicador": 6,
    "qtd_pacotes": 1,
    "qtd_unidades_base": 6,
    "preco_unitario": 4.17,
    "preco_pacote": 25.00,
    "valor_estimado": 25.00,
    "confianca": 0.68
  }
]
```

## 3. Camada Auxiliar de Upsell de Embalagem
Além das Top-$K$ recomendações de cross-sell geradas pela rede neural, o `RecommenderService` expõe `suggest_package_upsells(cart_items, available_skus)`:
- Detecta quando o cliente selecionou unidades avulsas perto de uma embalagem promocional disponível em estoque (ex.: 2 unidades avulsas de *Rapadura de Melado* a R$ 14,00 $\rightarrow$ sugerir *Pacote x3* por R$ 20,00).

## 4. API REST Leve em FastAPI (`api/main.py`)
Backend simples (~80–100 linhas) que carrega `RecommenderService.from_artifact("models/recommender_v1/")` na inicialização (com CORS liberado para o servidor Vite) e expõe 3 endpoints claros:

| Método & Rota | Finalidade | Payload / Resposta |
| :--- | :--- | :--- |
| `GET /api/catalog` | Retorna os 34 SKUs oficiais de `catalog_metadata.json` para popular o catálogo mocado do PDV | Lista de produtos (id, nome, base, embalagem, multiplicador, preços) |
| `POST /api/recommend` | Recebe o carrinho atual, horário/dia e opcionalmente `available_skus` (itens em estoque) | `{ "recommendations": [...Top-3...], "upsells": [...], "model_metrics": { "hit_rate_at_3": ... } }` |
| `GET /api/metrics` | Retorna o `training_report.json` com métricas finais no teste (20%) e resumo da arquitetura | JSON de métricas e hiperparâmetros |

Execução local: `uvicorn api.main:app --reload --port 8000`.

---

## Critérios de Pronto (*Definition of Done*)
- [x] Exportação executável via comando único (`python -m src.export`) gerando todos os artefatos em `models/recommender_v1/`.
- [x] Teste *round-trip* (`save → load → recommend`) reproduzindo exatamente os mesmos scores (`atol=1e-6`).
- [x] Garantia testada de que nenhum SKU presente em `cart_items` aparece nas recomendações.
- [x] Garantia testada de que, ao remover um SKU do `available_skus` (simulando falta de estoque), o ranking **pula automaticamente para o próximo item disponível** mantendo Top-3 válido.
- [x] Endpoints FastAPI (`GET /api/catalog`, `POST /api/recommend`, `GET /api/metrics`) validados via `TestClient`; `pytest tests/test_model_and_export.py` passando 100%.
