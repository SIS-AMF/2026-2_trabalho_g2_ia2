[← Etapa Anterior: Avaliação](6_avaliacao.md) | [Índice](README.md) | [Próxima Etapa: Interface e Entrega →](8_interface_e_entrega.md)

# Etapa 7 — Exportação Versionada do Modelo e Serviço de Inferência

| Campo | Detalhe |
| :--- | :--- |
| **Objetivo** | Empacotar o modelo treinado e todos os artefatos de pré-processamento/catálogo em formato versionado e expor uma API Python limpa para inferência sem re-treinamento |
| **Módulos** | `src/export.py`, `src/inference.py` |
| **Testes** | `tests/test_model_and_export.py` (testes de *round-trip save → load → predict* e regras de negócio) |
| **Entradas** | Melhor checkpoint de `FairMultiTaskRecommender`, scalers/encoders do treino, `catalog_metadata.json`, métricas finais |
| **Saídas** | Pacote completo em `models/recommender_v1/` |
| **Status** | `[ ] Planejado` |

---

## 1. Anatomia do Pacote Exportado (`models/recommender_v1/`)
1. **`model_weights.pt`:** pesos treinados (`model.state_dict()`) + dicionário de hiperparâmetros arquiteturais (`in_dim`, `hidden_dims`, `dropout`, `n_bases`, `n_skus`, `lambda_hier`) para reconstrução exata do módulo PyTorch.
2. **`model_scripted.pt` *(opcional/bônus)*:** versão exportada via `torch.jit.trace` / `torch.jit.script` para portabilidade desacoplada da classe Python.
3. **`preprocessing_artifacts.joblib`:** `StandardScaler`, codificadores de `evento_nome`/`faixa_horaria` e dicionários `base_to_idx` / `sku_to_idx`.
4. **`catalog_metadata.json`:** tabela oficial dos 34 SKUs (`produto_base`, `tipo_embalagem`, `multiplicador`, preços de referência e display name).
5. **`training_report.json`:** data/hora, seed, hiperparâmetros vencedores, histórico de épocas e métricas no teste.

## 2. Contrato do Serviço de Inferência (`src/inference.py`)
Interface simples `RecommenderService.from_artifact("models/recommender_v1/")` com método principal:

```python
service.recommend(
    cart_items=[{"sku_norm": "Alfajor Preto (x1)", "qtd_pacotes": 2}],
    evento_nome="Expointer_2026",
    hora=15,
    dia_semana=5,
    top_k=3,
)
```

Retorno tipado e ordenado (100% consistente com o catálogo):

```json
[
  {
    "rank": 1,
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
Além das Top-$K$ recomendações de cross-sell geradas pela rede neural, o `RecommenderService` expõe `suggest_package_upsells(cart_items)`:
- Detecta quando o cliente selecionou unidades avulsas perto de uma embalagem promocional do mesmo `produto_base` (ex.: 2 unidades avulsas de *Rapadura de Melado* a R$ 14,00 $\rightarrow$ sugerir *Pacote x3* por R$ 20,00 com economia por unidade).
- Mantém clara a separação entre **predição neural (cross-sell)** e **regra auxiliar determinística (upsell de embalagem)**.

---

## Critérios de Pronto (*Definition of Done*)
- [ ] Exportação executável via comando único (`python -m src.export`) gerando todos os artefatos em `models/recommender_v1/`.
- [ ] Teste *round-trip* (`save → load → recommend`) reproduzindo exatamente os mesmos scores (`atol=1e-6`).
- [ ] Garantia testada de que nenhum SKU presente em `cart_items` aparece nas recomendações retornadas.
- [ ] Garantia testada de que `qtd_unidades_base == qtd_pacotes * multiplicador` e `valor_estimado ≈ qtd_pacotes * preco_pacote` em 100% das respostas, inclusive para feiras inéditas (`evento_nome` desconhecido caindo em `"OUTRO"`) e carrinho vazio (cold-start).
- [ ] `pytest tests/test_model_and_export.py` passando 100%.
