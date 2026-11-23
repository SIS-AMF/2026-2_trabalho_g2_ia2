import {
  AlertTriangle,
  ArrowRight,
  Brain,
  Check,
  Loader2,
  Plus,
  RefreshCw,
  Sparkles,
  Tag,
  X,
} from "lucide-react";
import { useState } from "react";
import { formatBRL } from "../api/client";
import type {
  RecommendationItem,
  RecommendResponse,
  UpsellItem,
} from "../types";

interface Props {
  isOpen: boolean;
  isLoading: boolean;
  error: string | null;
  data: RecommendResponse | null;
  totalFinal: number;
  outOfStockCount: number;
  addedSkusInSession: Set<string>;
  onAddRecommendation: (item: RecommendationItem) => void;
  onApplyUpsell: (upsell: UpsellItem) => void;
  onRefresh: () => void;
  onConfirmSale: (formaPagamento: string) => void;
  onClose: () => void;
}

const PAYMENT_OPTIONS = ["Dinheiro", "PIX", "Cartão"] as const;

export function PDVRecommendationModal({
  isOpen,
  isLoading,
  error,
  data,
  totalFinal,
  outOfStockCount,
  addedSkusInSession,
  onAddRecommendation,
  onApplyUpsell,
  onRefresh,
  onConfirmSale,
  onClose,
}: Props) {
  const [formaPagamento, setFormaPagamento] = useState<string>("PIX");

  if (!isOpen) return null;

  const metrics = data?.model_metrics;

  return (
    <div className="fixed inset-0 z-50 bg-black/55 backdrop-blur-xs flex items-center justify-center p-3 sm:p-5">
      <div className="bg-white border border-cocoa-dark/15 rounded-2xl w-full max-w-3xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Cabeçalho IA */}
        <div className="p-4 bg-gradient-to-r from-amber-50 via-cream to-orange-50 border-b border-cocoa-dark/10 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="w-11 h-11 rounded-2xl bg-primary text-white flex items-center justify-center shadow-md shadow-primary/25 shrink-0">
              <Sparkles size={22} />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-black text-base text-cocoa-dark uppercase tracking-tight">
                  Sugestão de Cross-Sell — Rede Neural (Top-3)
                </h2>
                {metrics && (
                  <span
                    title="Taxa de acerto real (Hit Rate@3) medida no conjunto de teste Hold-Out (20%)"
                    className="inline-flex items-center gap-1 text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-cocoa-dark text-cream"
                  >
                    <Brain size={11} />
                    Taxa de Acerto Teste (Hit@3):{" "}
                    {(metrics.hit_rate_at_3 * 100).toFixed(1)}% SKU |{" "}
                    {(metrics.base_hit_rate_at_3 * 100).toFixed(1)}% Base
                  </span>
                )}
              </div>
              <p className="text-xs text-cocoa-dark/65 mt-0.5">
                Ofereça um destes itens complementares antes de concluir a venda simulada.
                {outOfStockCount > 0 && (
                  <span className="ml-1.5 font-bold text-primary">
                    (Filtro ativo: {outOfStockCount} SKU(s) esgotados pulados automaticamente)
                  </span>
                )}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar modal de recomendação"
            className="p-1.5 rounded-full text-cocoa-dark/50 hover:bg-black/10 cursor-pointer shrink-0"
          >
            <X size={18} />
          </button>
        </div>

        {/* Corpo */}
        <div className="p-4 overflow-y-auto scrollbar-thin flex-1 space-y-4">
          {isLoading ? (
            <div className="py-12 flex flex-col items-center justify-center gap-3 text-cocoa-dark/60">
              <Loader2 size={32} className="animate-spin text-primary" />
              <p className="text-sm font-bold">
                Consultando FairMultiTaskRecommender (PyTorch)...
              </p>
            </div>
          ) : error ? (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3">
              <AlertTriangle size={20} className="shrink-0 mt-0.5" />
              <div className="text-xs space-y-1">
                <p className="font-bold">Não foi possível obter recomendações da API.</p>
                <p>{error}</p>
                <button
                  type="button"
                  onClick={onRefresh}
                  className="mt-2 inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-rose-700 text-white font-bold cursor-pointer"
                >
                  <RefreshCw size={12} /> Tentar novamente
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Alertas auxiliares de Upsell de Embalagem */}
              {data && data.upsells.length > 0 && (
                <div className="space-y-2">
                  {data.upsells.map((upsell) => (
                    <div
                      key={upsell.sugestao_sku_norm}
                      className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                    >
                      <div className="flex items-start gap-2.5 text-emerald-900">
                        <Tag size={17} className="text-emerald-600 shrink-0 mt-0.5" />
                        <div className="text-xs">
                          <span className="font-black uppercase text-[10px] tracking-wider text-emerald-700 block">
                            Oportunidade de Upsell de Embalagem
                          </span>
                          <p className="font-medium mt-0.5">{upsell.mensagem}</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => onApplyUpsell(upsell)}
                        className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold shrink-0 cursor-pointer"
                      >
                        Trocar p/ Pacote
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Grid Top-3 Recomendações */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {data?.recommendations.map((rec) => {
                  const pctConfianca = Math.min(100, Math.round(rec.confianca * 100));
                  const jaAdicionado = addedSkusInSession.has(rec.sku_norm);
                  const isTop1 = rec.rank === 1;

                  return (
                    <div
                      key={rec.sku_norm}
                      className={`relative rounded-2xl border-2 p-3.5 flex flex-col justify-between transition-all ${
                        isTop1
                          ? "border-primary bg-amber-50/40 shadow-md shadow-primary/10"
                          : "border-cocoa-dark/12 bg-white"
                      }`}
                    >
                      <div>
                        {/* Rank + Confiança */}
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <span
                            className={`text-[10px] font-black uppercase px-2 py-0.5 rounded-full ${
                              isTop1
                                ? "bg-primary text-white"
                                : "bg-cocoa-dark/10 text-cocoa-dark"
                            }`}
                          >
                            #{rec.rank} {isTop1 ? "Melhor Escolha" : "Recomendado"}
                          </span>
                          <span className="text-xs font-mono font-black text-primary">
                            {pctConfianca}% afinidade
                          </span>
                        </div>

                        {/* Barra de confiança */}
                        <div className="w-full h-1.5 rounded-full bg-cocoa-dark/10 overflow-hidden mb-3">
                          <div
                            className="h-full rounded-full bg-primary transition-all duration-500"
                            style={{ width: `${pctConfianca}%` }}
                          />
                        </div>

                        {/* Título completo + produto base */}
                        <h3 className="font-black text-sm text-cocoa-dark leading-snug uppercase">
                          {rec.produto_com_multiplicador}
                        </h3>
                        <p className="text-[11px] text-cocoa-dark/55 mt-0.5">
                          Base: <strong>{rec.produto_base}</strong>
                        </p>

                        {/* Ficha completa (multiplicador, qtd pacotes, qtd unitária, preços) */}
                        <div className="mt-3 p-2.5 rounded-xl bg-cream/80 border border-cocoa-dark/8 space-y-1 text-[11px]">
                          <div className="flex justify-between">
                            <span className="text-cocoa-dark/60">Embalagem / Mult.</span>
                            <span className="font-bold text-cocoa-dark">
                              {rec.tipo_embalagem} (x{rec.multiplicador})
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-cocoa-dark/60">Qtd. Sugerida</span>
                            <span className="font-bold text-cocoa-dark">
                              {rec.qtd_pacotes} pct ({rec.qtd_unidades_base} un.)
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-cocoa-dark/60">Preço Unitário</span>
                            <span className="font-mono font-semibold">
                              {formatBRL(rec.preco_unitario)}
                            </span>
                          </div>
                          <div className="flex justify-between pt-1 border-t border-cocoa-dark/10">
                            <span className="font-bold text-cocoa-dark">Acréscimo</span>
                            <span className="font-mono font-black text-primary text-xs">
                              +{formatBRL(rec.valor_estimado)}
                            </span>
                          </div>
                        </div>
                      </div>

                      <button
                        type="button"
                        disabled={jaAdicionado}
                        onClick={() => onAddRecommendation(rec)}
                        className={`mt-3 w-full py-2.5 px-3 rounded-xl text-xs font-black uppercase tracking-wide flex items-center justify-center gap-1.5 transition-all ${
                          jaAdicionado
                            ? "bg-emerald-100 text-emerald-800 border border-emerald-300 cursor-default"
                            : "bg-primary hover:bg-primary-hover text-white shadow-sm cursor-pointer active:scale-[0.98]"
                        }`}
                      >
                        {jaAdicionado ? (
                          <>
                            <Check size={14} /> Adicionado ao Carrinho
                          </>
                        ) : (
                          <>
                            <Plus size={14} /> Adicionar (+{formatBRL(rec.valor_estimado)})
                          </>
                        )}
                      </button>
                    </div>
                  );
                })}
              </div>

              {addedSkusInSession.size > 0 && (
                <div className="flex items-center justify-between p-2.5 rounded-xl bg-amber-50 border border-amber-200 text-xs text-cocoa-dark">
                  <span>
                    Carrinho atualizado com sugestão da IA! Deseja recalcular novas sugestões?
                  </span>
                  <button
                    type="button"
                    onClick={onRefresh}
                    className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-cocoa-dark text-cream font-bold hover:opacity-90 cursor-pointer shrink-0"
                  >
                    <RefreshCw size={12} /> Recalcular Top-3
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        {/* Rodapé — Venda Simulada */}
        <div className="p-4 border-t border-cocoa-dark/10 bg-cream/60 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold uppercase text-cocoa-dark/60">
              Pgto Simulado:
            </span>
            <div className="flex rounded-xl border border-cocoa-dark/15 overflow-hidden bg-white">
              {PAYMENT_OPTIONS.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => setFormaPagamento(opt)}
                  className={`px-2.5 py-1.5 text-xs font-bold transition-colors cursor-pointer ${
                    formaPagamento === opt
                      ? "bg-cocoa-dark text-cream"
                      : "text-cocoa-dark/70 hover:bg-cocoa-dark/5"
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-2.5 rounded-xl border border-cocoa-dark/15 text-xs font-bold text-cocoa-dark hover:bg-cocoa-dark/5 cursor-pointer"
            >
              Voltar
            </button>
            <button
              type="button"
              onClick={() => onConfirmSale(formaPagamento)}
              className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs sm:text-sm font-black uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 cursor-pointer active:scale-[0.98]"
            >
              <span>Concluir Venda ({formatBRL(totalFinal)})</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
