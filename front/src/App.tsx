import {
  Boxes,
  Brain,
  Clock,
  Minus,
  PackageSearch,
  Plus,
  Search,
  ShoppingBag,
  Sparkles,
  Store,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchCatalog,
  fetchMetrics,
  fetchRecommendations,
  formatBRL,
} from "./api/client";
import { PDVRecommendationModal } from "./components/PDVRecommendationModal";
import { PDVSuccessModal } from "./components/PDVSuccessModal";
import { ProductPresentationModal } from "./components/ProductPresentationModal";
import { StockManagerModal } from "./components/StockManagerModal";
import type {
  CartEntry,
  CatalogResponse,
  CatalogSku,
  ModelMetricsSummary,
  ProductGroup,
  RecommendationItem,
  RecommendResponse,
  SimulatedSaleResult,
  UpsellItem,
} from "./types";

const DIAS_SEMANA = [
  { value: 0, label: "Seg" },
  { value: 1, label: "Ter" },
  { value: 2, label: "Qua" },
  { value: 3, label: "Qui" },
  { value: 4, label: "Sex" },
  { value: 5, label: "Sáb" },
  { value: 6, label: "Dom" },
];

export function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<ModelMetricsSummary | null>(null);

  // Estado do PDV
  const [searchTerm, setSearchTerm] = useState("");
  const [cart, setCart] = useState<CartEntry[]>([]);
  const [descontoReais, setDescontoReais] = useState<number>(0);
  const [outOfStockSkus, setOutOfStockSkus] = useState<Set<string>>(new Set());

  // Contexto temporal (padrão: sábado 15h ou horário atual)
  const now = new Date();
  const [hora, setHora] = useState<number>(now.getHours());
  const [diaSemana, setDiaSemana] = useState<number>(
    (now.getDay() + 6) % 7, // Converte JS (0=Dom) para Python (0=Seg..6=Dom)
  );

  // Modais
  const [selectedGroup, setSelectedGroup] = useState<ProductGroup | null>(null);
  const [isStockModalOpen, setIsStockModalOpen] = useState(false);
  const [isRecModalOpen, setIsRecModalOpen] = useState(false);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [recData, setRecData] = useState<RecommendResponse | null>(null);
  const [addedSkusInSession, setAddedSkusInSession] = useState<Set<string>>(
    new Set(),
  );
  const [saleResult, setSaleResult] = useState<SimulatedSaleResult | null>(null);
  const [saleCounter, setSaleCounter] = useState(1);

  useEffect(() => {
    void (async () => {
      const { catalog: loaded, online } = await fetchCatalog();
      setCatalog(loaded);
      setApiOnline(online);
      const m = await fetchMetrics();
      if (m) setMetrics(m);
    })();
  }, []);

  const skuByNorm = useMemo(() => {
    const map = new Map<string, CatalogSku>();
    if (catalog) {
      for (const s of catalog.skus) map.set(s.sku_norm, s);
    }
    return map;
  }, [catalog]);

  // Agrupar os 34 SKUs nos 20 Produtos Base (mesma UX do PDVProductCard + ProductPresentationModal)
  const productGroups = useMemo<ProductGroup[]>(() => {
    if (!catalog) return [];
    const byBase = new Map<number, ProductGroup>();
    for (const base of catalog.bases) {
      byBase.set(base.base_id, {
        base_id: base.base_id,
        produto_base: base.produto_base,
        skus: [],
      });
    }
    for (const sku of catalog.skus) {
      byBase.get(sku.base_id)?.skus.push(sku);
    }
    const groups = Array.from(byBase.values()).map((g) => ({
      ...g,
      skus: [...g.skus].sort((a, b) => a.multiplicador - b.multiplicador),
    }));
    const term = searchTerm.trim().toLowerCase();
    const filtered = term
      ? groups.filter(
          (g) =>
            g.produto_base.toLowerCase().includes(term) ||
            g.skus.some((s) => s.display_name.toLowerCase().includes(term)),
        )
      : groups;
    return filtered.sort((a, b) =>
      a.produto_base.localeCompare(b.produto_base, "pt-BR", {
        sensitivity: "base",
      }),
    );
  }, [catalog, searchTerm]);

  const addSkuToCart = useCallback(
    (sku: CatalogSku, qty = 1, fromAI = false) => {
      setCart((prev) => {
        const idx = prev.findIndex((e) => e.sku.sku_norm === sku.sku_norm);
        if (idx >= 0) {
          const updated = [...prev];
          updated[idx] = {
            ...updated[idx],
            qtd_pacotes: updated[idx].qtd_pacotes + qty,
            fromAI: updated[idx].fromAI || fromAI,
          };
          return updated;
        }
        return [...prev, { sku, qtd_pacotes: qty, fromAI }];
      });
    },
    [],
  );

  const updateCartQty = useCallback((skuNorm: string, delta: number) => {
    setCart((prev) =>
      prev
        .map((e) =>
          e.sku.sku_norm === skuNorm
            ? { ...e, qtd_pacotes: e.qtd_pacotes + delta }
            : e,
        )
        .filter((e) => e.qtd_pacotes > 0),
    );
  }, []);

  const setCartQty = useCallback((skuNorm: string, qty: number) => {
    if (qty <= 0) return;
    setCart((prev) =>
      prev.map((e) =>
        e.sku.sku_norm === skuNorm ? { ...e, qtd_pacotes: qty } : e,
      ),
    );
  }, []);

  const removeFromCart = useCallback((skuNorm: string) => {
    setCart((prev) => prev.filter((e) => e.sku.sku_norm !== skuNorm));
  }, []);

  const subtotal = useMemo(
    () =>
      cart.reduce(
        (acc, e) => acc + e.qtd_pacotes * e.sku.preco_pacote_ref,
        0,
      ),
    [cart],
  );
  const descontoAplicado = Math.min(Math.max(0, descontoReais), subtotal);
  const totalFinal = Math.max(0, subtotal - descontoAplicado);
  const totalUnidades = useMemo(
    () =>
      cart.reduce((acc, e) => acc + e.qtd_pacotes * e.sku.multiplicador, 0),
    [cart],
  );

  const availableSkusList = useMemo(() => {
    if (!catalog) return undefined;
    if (outOfStockSkus.size === 0) return undefined;
    return catalog.skus
      .filter((s) => !outOfStockSkus.has(s.sku_norm))
      .map((s) => s.sku_norm);
  }, [catalog, outOfStockSkus]);

  const consultarRecomendacoes = useCallback(
    async (cartSnapshot: CartEntry[]) => {
      setRecLoading(true);
      setRecError(null);
      try {
        const res = await fetchRecommendations({
          cart_items: cartSnapshot.map((e) => ({
            sku_norm: e.sku.sku_norm,
            qtd_pacotes: e.qtd_pacotes,
          })),
          hora,
          dia_semana: diaSemana,
          top_k: 3,
          available_skus: availableSkusList,
        });
        setRecData(res);
        setMetrics(res.model_metrics);
        setApiOnline(true);
      } catch (err) {
        setRecError(err instanceof Error ? err.message : "Erro inesperado");
      } finally {
        setRecLoading(false);
      }
    },
    [hora, diaSemana, availableSkusList],
  );

  const handleOpenFinalize = () => {
    if (cart.length === 0) return;
    setAddedSkusInSession(new Set());
    setIsRecModalOpen(true);
    void consultarRecomendacoes(cart);
  };

  const handleAddRecommendation = (rec: RecommendationItem) => {
    const sku = skuByNorm.get(rec.sku_norm);
    if (!sku) return;
    addSkuToCart(sku, rec.qtd_pacotes, true);
    setAddedSkusInSession((prev) => new Set(prev).add(rec.sku_norm));
  };

  const handleApplyUpsell = (upsell: UpsellItem) => {
    const pkgSku = skuByNorm.get(upsell.sugestao_sku_norm);
    if (!pkgSku) return;
    setCart((prev) => {
      const semOrigem = prev.filter(
        (e) => e.sku.sku_norm !== upsell.origem_sku_norm,
      );
      const existsIdx = semOrigem.findIndex(
        (e) => e.sku.sku_norm === pkgSku.sku_norm,
      );
      if (existsIdx >= 0) {
        const copy = [...semOrigem];
        copy[existsIdx] = {
          ...copy[existsIdx],
          qtd_pacotes: copy[existsIdx].qtd_pacotes + 1,
          fromAI: true,
        };
        return copy;
      }
      return [...semOrigem, { sku: pkgSku, qtd_pacotes: 1, fromAI: true }];
    });
    setAddedSkusInSession((prev) => new Set(prev).add(pkgSku.sku_norm));
  };

  const handleConfirmSale = (formaPagamento: string) => {
    const itensIA = cart.filter((e) => e.fromAI);
    const acrescimoIA = itensIA.reduce(
      (acc, e) => acc + e.qtd_pacotes * e.sku.preco_pacote_ref,
      0,
    );
    const result: SimulatedSaleResult = {
      numeroVenda: `SIM-${String(saleCounter).padStart(4, "0")}`,
      dataHora: new Date().toLocaleString("pt-BR"),
      formaPagamento,
      itens: cart,
      subtotal,
      desconto: descontoAplicado,
      total: totalFinal,
      acrescimoIA,
      itensAdicionadosIA: itensIA.length,
    };
    setSaleCounter((c) => c + 1);
    setIsRecModalOpen(false);
    setSaleResult(result);
  };

  const handleNewSale = () => {
    setSaleResult(null);
    setCart([]);
    setDescontoReais(0);
    setAddedSkusInSession(new Set());
  };

  const toggleOutOfStock = (skuNorm: string) => {
    setOutOfStockSkus((prev) => {
      const next = new Set(prev);
      if (next.has(skuNorm)) next.delete(skuNorm);
      else next.add(skuNorm);
      return next;
    });
  };

  const handleCardClick = (group: ProductGroup) => {
    const disponiveis = group.skus.filter(
      (s) => !outOfStockSkus.has(s.sku_norm),
    );
    if (disponiveis.length === 0) return;
    if (group.skus.length === 1) {
      addSkuToCart(group.skus[0], 1, false);
    } else {
      setSelectedGroup(group);
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-cream text-cocoa-dark">
      {/* Top Bar estilo Feira PDV */}
      <header className="shrink-0 px-4 py-2.5 bg-white border-b-2 border-cocoa-dark/10 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-primary text-white flex items-center justify-center shadow-sm">
            <Store size={19} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-black text-sm sm:text-base uppercase tracking-tight text-cocoa-dark">
                Feira PDV — Terminal Inteligente
              </h1>
              <span
                className={`text-[10px] font-extrabold px-2 py-0.5 rounded-full flex items-center gap-1 ${
                  apiOnline
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-amber-100 text-amber-800"
                }`}
              >
                <Brain size={11} />
                {apiOnline ? "IA PyTorch Online" : "Catálogo Local (Inicie FastAPI)"}
              </span>
            </div>
            <p className="text-[11px] text-cocoa-dark/55">
              Rede Neural Multi-Task (Genérica) •{" "}
              {metrics
                ? `Hit Rate@3: ${(metrics.hit_rate_at_3 * 100).toFixed(1)}% SKU / ${(metrics.base_hit_rate_at_3 * 100).toFixed(1)}% Base`
                : "34 SKUs canônicos / 20 produtos base"}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Seletor discreto de contexto temporal */}
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-cream border border-cocoa-dark/12 text-xs">
            <Clock size={14} className="text-primary" />
            <select
              aria-label="Dia da semana"
              value={diaSemana}
              onChange={(e) => setDiaSemana(Number(e.target.value))}
              className="bg-transparent font-bold outline-none cursor-pointer"
            >
              {DIAS_SEMANA.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
            <span className="text-cocoa-dark/30">•</span>
            <select
              aria-label="Hora da venda"
              value={hora}
              onChange={(e) => setHora(Number(e.target.value))}
              className="bg-transparent font-mono font-bold outline-none cursor-pointer"
            >
              {Array.from({ length: 16 }, (_, i) => i + 8).map((h) => (
                <option key={h} value={h}>
                  {String(h).padStart(2, "0")}:00
                </option>
              ))}
            </select>
          </div>

          {/* Botão Simulador de Estoque */}
          <button
            type="button"
            onClick={() => setIsStockModalOpen(true)}
            className={`px-3 py-1.5 rounded-xl border text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer ${
              outOfStockSkus.size > 0
                ? "bg-rose-50 border-rose-300 text-rose-800 hover:bg-rose-100"
                : "bg-white border-cocoa-dark/15 text-cocoa-dark hover:bg-cocoa-dark/5"
            }`}
          >
            <Boxes size={14} />
            <span>
              Estoque{" "}
              {outOfStockSkus.size > 0
                ? `(${outOfStockSkus.size} esgotado)`
                : "(34/34)"}
            </span>
          </button>
        </div>
      </header>

      {/* Conteúdo Principal: Catálogo (Esquerda) + Carrinho (Direita) */}
      <main className="flex-1 min-h-0 flex flex-col sm:flex-row overflow-hidden">
        {/* Coluna Catálogo */}
        <section className="flex-1 flex flex-col p-3 gap-3 min-h-0 overflow-hidden">
          <div className="relative shrink-0">
            <Search
              size={16}
              className="absolute left-3.5 top-1/2 -translate-y-1/2 text-cocoa-dark/40"
            />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar produto ou embalagem (ex.: Cuca, Rapadura, Alfajor, Pacote)..."
              className="w-full pl-10 pr-9 py-2.5 rounded-xl bg-white border border-cocoa-dark/15 text-sm outline-none focus:border-primary shadow-2xs"
            />
            {searchTerm && (
              <button
                type="button"
                onClick={() => setSearchTerm("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-cocoa-dark/40 hover:text-cocoa-dark cursor-pointer"
              >
                <X size={16} />
              </button>
            )}
          </div>

          {productGroups.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-cocoa-dark/50">
              <PackageSearch size={32} className="mb-2 opacity-60" />
              <p className="font-bold text-sm">Nenhum produto encontrado</p>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto min-h-0 scrollbar-thin grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2.5 auto-rows-max pr-1">
              {productGroups.map((group) => {
                const ativos = group.skus.filter(
                  (s) => !outOfStockSkus.has(s.sku_norm),
                );
                const esgotado = ativos.length === 0;
                return (
                  <button
                    key={group.base_id}
                    type="button"
                    disabled={esgotado}
                    onClick={() => handleCardClick(group)}
                    className={`relative flex flex-col justify-between p-2.5 rounded-xl border text-left transition-all ${
                      esgotado
                        ? "bg-cocoa-dark/5 border-cocoa-dark/10 opacity-45 cursor-not-allowed"
                        : "bg-white border-cocoa-dark/12 hover:border-primary/60 hover:shadow-lg hover:shadow-primary/10 active:scale-[0.98] cursor-pointer"
                    }`}
                  >
                    <div className="flex items-center justify-between w-full mb-1.5">
                      <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-cream border border-cocoa-dark/10 text-cocoa-dark/65">
                        {group.skus.length > 1
                          ? `${ativos.length}/${group.skus.length} emb.`
                          : "Avulso"}
                      </span>
                      <span
                        className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ${
                          esgotado
                            ? "bg-rose-100 text-rose-700"
                            : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                        }`}
                      >
                        {esgotado ? "Esgotado" : "Disponível"}
                      </span>
                    </div>

                    <h3 className="font-black text-xs text-cocoa-dark leading-snug uppercase mb-2">
                      {group.produto_base}
                    </h3>

                    <div className="mt-auto border-t border-cocoa-dark/8 w-full pt-1.5 space-y-0.5">
                      {group.skus.slice(0, 3).map((sku) => (
                        <div
                          key={sku.sku_norm}
                          className={`flex items-center justify-between text-[11px] ${
                            outOfStockSkus.has(sku.sku_norm)
                              ? "line-through opacity-40"
                              : ""
                          }`}
                        >
                          <span className="font-mono text-cocoa-dark/55 truncate">
                            {sku.multiplicador > 1
                              ? `${sku.tipo_embalagem} x${sku.multiplicador}`
                              : "Unidade"}
                          </span>
                          <span className="font-mono font-black text-primary">
                            {formatBRL(sku.preco_pacote_ref)}
                          </span>
                        </div>
                      ))}
                      {group.skus.length > 3 && (
                        <p className="text-[10px] font-bold text-primary/80 text-right">
                          +{group.skus.length - 3} opções...
                        </p>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        {/* Coluna Carrinho (adaptada de PDVCart.tsx) */}
        <aside className="w-full sm:w-88 lg:w-96 shrink-0 flex flex-col h-full min-h-0 border-t-2 sm:border-t-0 sm:border-l-2 border-cocoa-dark/10 bg-white/80 p-3">
          <div className="flex items-center justify-between pb-2 border-b border-cocoa-dark/10">
            <div className="flex items-center gap-2">
              <ShoppingBag size={18} className="text-primary" />
              <h2 className="font-black text-sm uppercase tracking-wide text-cocoa-dark">
                Carrinho ({cart.length})
              </h2>
            </div>
            {cart.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  setCart([]);
                  setDescontoReais(0);
                }}
                className="text-[11px] font-bold text-rose-600 hover:underline cursor-pointer"
              >
                Limpar
              </button>
            )}
          </div>

          {cart.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-cocoa-dark/45">
              <div className="w-12 h-12 rounded-2xl bg-cocoa-dark/5 flex items-center justify-center mb-2">
                <ShoppingBag size={24} />
              </div>
              <p className="text-xs font-bold text-cocoa-dark/70">
                Carrinho vazio
              </p>
              <p className="text-[11px] mt-1">
                Clique nos produtos ao lado para montar uma cesta e testar a recomendação da Rede Neural.
              </p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col gap-2 overflow-y-auto min-h-0 scrollbar-thin pr-1 py-2">
              {cart.map((entry) => {
                const sub = entry.qtd_pacotes * entry.sku.preco_pacote_ref;
                return (
                  <div
                    key={entry.sku.sku_norm}
                    className="flex flex-col p-2 border border-cocoa-dark/12 rounded-xl bg-white shadow-2xs"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="font-black text-xs text-cocoa-dark uppercase leading-tight">
                        {entry.sku.display_name}
                        {entry.fromAI && (
                          <span className="ml-1.5 text-[9px] font-black px-1.5 py-0.5 rounded bg-amber-100 text-primary">
                            IA
                          </span>
                        )}
                      </p>
                      <button
                        type="button"
                        onClick={() => removeFromCart(entry.sku.sku_norm)}
                        aria-label={`Remover ${entry.sku.display_name}`}
                        className="p-1 text-rose-500 hover:bg-rose-50 rounded-lg cursor-pointer shrink-0"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>

                    <div className="flex items-center justify-between text-[10px] text-cocoa-dark/55 font-mono mt-0.5 mb-2">
                      <span>
                        {formatBRL(entry.sku.preco_pacote_ref)} / pct (
                        {entry.qtd_pacotes * entry.sku.multiplicador} un.)
                      </span>
                      <span className="font-bold text-primary">
                        x{entry.sku.multiplicador}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center bg-cream rounded-lg border border-cocoa-dark/12">
                        <button
                          type="button"
                          onClick={() => updateCartQty(entry.sku.sku_norm, -1)}
                          className="w-7 h-7 flex items-center justify-center text-cocoa-dark/70 hover:text-rose-600 cursor-pointer"
                        >
                          <Minus size={14} />
                        </button>
                        <input
                          type="number"
                          min={1}
                          value={entry.qtd_pacotes}
                          onChange={(e) => {
                            const val = Number.parseInt(e.target.value, 10);
                            if (!Number.isNaN(val) && val > 0) {
                              setCartQty(entry.sku.sku_norm, val);
                            }
                          }}
                          className="w-8 h-7 bg-transparent text-center text-xs font-mono font-black outline-none"
                        />
                        <button
                          type="button"
                          onClick={() => updateCartQty(entry.sku.sku_norm, 1)}
                          className="w-7 h-7 flex items-center justify-center text-cocoa-dark/70 hover:text-emerald-600 cursor-pointer"
                        >
                          <Plus size={14} />
                        </button>
                      </div>

                      <span className="font-mono font-black text-sm text-primary">
                        {formatBRL(sub)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Resumo + Desconto + Botão Finalizar Venda */}
          <div className="shrink-0 flex flex-col gap-2 pt-2 border-t border-cocoa-dark/10 mt-auto">
            {cart.length > 0 && (
              <div className="flex items-center justify-between gap-2 text-xs">
                <label
                  htmlFor="desconto-input"
                  className="text-cocoa-dark/65 font-semibold"
                >
                  Desconto (R$):
                </label>
                <input
                  id="desconto-input"
                  type="number"
                  min={0}
                  step={1}
                  value={descontoReais || ""}
                  placeholder="0,00"
                  onChange={(e) =>
                    setDescontoReais(Math.max(0, Number(e.target.value) || 0))
                  }
                  className="w-24 px-2 py-1 rounded-lg border border-cocoa-dark/15 text-right font-mono font-bold bg-white outline-none focus:border-primary"
                />
              </div>
            )}

            <div className="p-2.5 rounded-xl bg-cream border border-cocoa-dark/10 space-y-1 text-xs">
              <div className="flex justify-between text-cocoa-dark/60">
                <span>Subtotal ({totalUnidades} un. base)</span>
                <span className="font-mono font-semibold">
                  {formatBRL(subtotal)}
                </span>
              </div>
              {descontoAplicado > 0 && (
                <div className="flex justify-between text-emerald-700">
                  <span>Desconto</span>
                  <span className="font-mono">
                    -{formatBRL(descontoAplicado)}
                  </span>
                </div>
              )}
              <div className="flex justify-between pt-1.5 border-t border-cocoa-dark/10 text-sm font-black">
                <span>Total</span>
                <span className="font-mono text-primary">
                  {formatBRL(totalFinal)}
                </span>
              </div>
            </div>

            <button
              type="button"
              disabled={cart.length === 0}
              onClick={handleOpenFinalize}
              className={`w-full py-3 px-4 rounded-xl font-black text-xs sm:text-sm uppercase tracking-wider flex items-center justify-center gap-2 transition-all border-2 shadow-lg ${
                cart.length === 0
                  ? "bg-cocoa-dark/5 border-cocoa-dark/10 text-cocoa-dark/35 cursor-not-allowed shadow-none"
                  : "bg-primary hover:bg-primary-hover text-white border-primary/20 shadow-primary/25 active:scale-[0.99] cursor-pointer"
              }`}
            >
              <Sparkles size={16} />
              <span>Finalizar Venda (Ver Top-3 IA)</span>
            </button>
          </div>
        </aside>
      </main>

      {/* Modais */}
      <ProductPresentationModal
        group={selectedGroup}
        outOfStockSkus={outOfStockSkus}
        onSelect={(sku) => addSkuToCart(sku, 1, false)}
        onClose={() => setSelectedGroup(null)}
      />

      <StockManagerModal
        isOpen={isStockModalOpen}
        skus={catalog?.skus ?? []}
        outOfStockSkus={outOfStockSkus}
        onToggleSku={toggleOutOfStock}
        onResetAll={() => setOutOfStockSkus(new Set())}
        onClose={() => setIsStockModalOpen(false)}
      />

      <PDVRecommendationModal
        isOpen={isRecModalOpen}
        isLoading={recLoading}
        error={recError}
        data={recData}
        totalFinal={totalFinal}
        outOfStockCount={outOfStockSkus.size}
        addedSkusInSession={addedSkusInSession}
        onAddRecommendation={handleAddRecommendation}
        onApplyUpsell={handleApplyUpsell}
        onRefresh={() => void consultarRecomendacoes(cart)}
        onConfirmSale={handleConfirmSale}
        onClose={() => setIsRecModalOpen(false)}
      />

      <PDVSuccessModal
        saleResult={saleResult}
        onNewSale={handleNewSale}
        onClose={() => setSaleResult(null)}
      />
    </div>
  );
}
export default App;
