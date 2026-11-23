interface CatalogBase {
  base_id: number;
  produto_base: string;
  preco_unitario_avulso_ref: number;
}

export interface CatalogSku {
  sku_id: number;
  sku_norm: string;
  produto_base: string;
  base_id: number;
  tipo_embalagem: string;
  display_name: string;
  multiplicador: number;
  preco_unitario_ref: number;
  preco_pacote_ref: number;
  economia_unitaria_pct: number;
  frequencia_varejo: number;
  frequencia_total: number;
}

export interface CatalogResponse {
  n_bases: number;
  n_skus: number;
  bases: CatalogBase[];
  skus: CatalogSku[];
}

export interface ProductGroup {
  base_id: number;
  produto_base: string;
  skus: CatalogSku[];
}

export interface CartEntry {
  sku: CatalogSku;
  qtd_pacotes: number;
  fromAI?: boolean;
}

export interface RecommendationItem {
  rank: number;
  sku_id: number;
  sku_norm: string;
  produto_base: string;
  produto_com_multiplicador: string;
  tipo_embalagem: string;
  multiplicador: number;
  qtd_pacotes: number;
  qtd_unidades_base: number;
  preco_unitario: number;
  preco_pacote: number;
  valor_estimado: number;
  confianca: number;
}

export interface UpsellItem {
  origem_sku_norm: string;
  origem_display_name: string;
  qtd_unidades_no_carrinho: number;
  custo_atual: number;
  sugestao_sku_norm: string;
  sugestao_display_name: string;
  multiplicador: number;
  preco_pacote: number;
  preco_unitario_pacote: number;
  economia_unitaria_pct: number;
  mensagem: string;
}

export interface ModelMetricsSummary {
  version: string;
  model_name: string;
  hit_rate_at_1: number;
  hit_rate_at_3: number;
  hit_rate_at_5: number;
  ndcg_at_3: number;
  base_hit_rate_at_3: number;
  coverage_at_3: number;
  qty_mae: number;
  qty_exact_acc: number;
}

export interface RecommendResponse {
  recommendations: RecommendationItem[];
  upsells: UpsellItem[];
  model_metrics: ModelMetricsSummary;
  context_used: {
    hora: number;
    dia_semana: number;
    stock_filter_active: boolean;
  };
}

export interface SimulatedSaleResult {
  numeroVenda: string;
  dataHora: string;
  formaPagamento: string;
  itens: CartEntry[];
  subtotal: number;
  desconto: number;
  total: number;
  acrescimoIA: number;
  itensAdicionadosIA: number;
}
