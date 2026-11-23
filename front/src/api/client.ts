import catalogFallback from "../data/catalogFallback.json";
import type {
  CatalogResponse,
  ModelMetricsSummary,
  RecommendResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export async function fetchCatalog(): Promise<{
  catalog: CatalogResponse;
  online: boolean;
}> {
  try {
    const res = await fetch(`${API_BASE}/api/catalog`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = (await res.json()) as CatalogResponse;
    return { catalog: data, online: true };
  } catch {
    return {
      catalog: catalogFallback as CatalogResponse,
      online: false,
    };
  }
}

export async function fetchMetrics(): Promise<ModelMetricsSummary | null> {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) return null;
    const data = (await res.json()) as { summary: ModelMetricsSummary };
    return data.summary;
  } catch {
    return null;
  }
}

export async function fetchRecommendations(params: {
  cart_items: { sku_norm: string; qtd_pacotes: number }[];
  hora: number;
  dia_semana: number;
  top_k?: number;
  available_skus?: string[];
}): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      cart_items: params.cart_items,
      hora: params.hora,
      dia_semana: params.dia_semana,
      top_k: params.top_k ?? 3,
      available_skus: params.available_skus,
    }),
  });
  if (!res.ok) {
    throw new Error(
      `Falha ao consultar API de recomendação (HTTP ${res.status}). Verifique se o backend FastAPI está rodando na porta 8000.`,
    );
  }
  return (await res.json()) as RecommendResponse;
}

export function formatBRL(value: number): string {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(value);
}
