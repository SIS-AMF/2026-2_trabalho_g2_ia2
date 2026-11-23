import { CheckCircle2, RotateCcw, Sparkles, X } from "lucide-react";
import { formatBRL } from "../api/client";
import type { SimulatedSaleResult } from "../types";

interface Props {
  saleResult: SimulatedSaleResult | null;
  onNewSale: () => void;
  onClose: () => void;
}

export function PDVSuccessModal({ saleResult, onNewSale, onClose }: Props) {
  if (!saleResult) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/55 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="border border-cocoa-dark/15 rounded-2xl w-full max-w-md shadow-2xl overflow-hidden flex flex-col max-h-[90vh] bg-white">
        {/* Header Sucesso */}
        <div className="p-5 bg-emerald-50/70 flex flex-col items-center text-center relative border-b border-cocoa-dark/10">
          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar comprovante"
            className="absolute top-3 right-3 p-1 rounded-full text-cocoa-dark/40 hover:bg-black/5 cursor-pointer"
          >
            <X size={18} />
          </button>

          <div className="w-14 h-14 rounded-full bg-emerald-500 text-white flex items-center justify-center mb-3 shadow-md shadow-emerald-500/25">
            <CheckCircle2 size={32} />
          </div>

          <h2 className="text-base font-black text-cocoa-dark uppercase tracking-wide">
            Venda Simulada Concluída!
          </h2>
          <p className="text-[11px] text-cocoa-dark/55 mt-1">
            {saleResult.numeroVenda} • {saleResult.dataHora} • Modo Demonstração IA
          </p>
        </div>

        {/* Itens e Resumo */}
        <div className="p-4 flex flex-col gap-3 overflow-y-auto scrollbar-thin">
          {saleResult.itensAdicionadosIA > 0 && (
            <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 text-amber-900 font-bold">
                <Sparkles size={16} className="text-primary shrink-0" />
                <span>
                  Cross-sell IA: +{saleResult.itensAdicionadosIA} item(ns) aceito(s)
                </span>
              </div>
              <span className="font-mono font-black text-primary">
                +{formatBRL(saleResult.acrescimoIA)}
              </span>
            </div>
          )}

          <div className="p-3 bg-cream/70 rounded-xl border border-cocoa-dark/10 space-y-2">
            <p className="text-[10px] font-black uppercase tracking-wider text-cocoa-dark/45">
              Itens da Venda
            </p>
            <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1 scrollbar-thin">
              {saleResult.itens.map((entry) => (
                <div
                  key={entry.sku.sku_norm}
                  className="flex items-center justify-between text-xs gap-2"
                >
                  <span className="text-cocoa-dark truncate">
                    {entry.qtd_pacotes}x {entry.sku.display_name}
                    {entry.fromAI && (
                      <span className="ml-1.5 text-[9px] font-black px-1.5 py-0.5 rounded bg-primary/15 text-primary uppercase">
                        IA
                      </span>
                    )}
                  </span>
                  <span className="font-mono font-semibold text-cocoa-dark/80 shrink-0">
                    {formatBRL(entry.qtd_pacotes * entry.sku.preco_pacote_ref)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 bg-cocoa-dark/[0.02] rounded-xl border border-cocoa-dark/10 space-y-1.5 text-xs">
            <div className="flex items-center justify-between text-cocoa-dark/65">
              <span>Forma de Pagamento (Simulada)</span>
              <span className="font-bold text-cocoa-dark">
                {saleResult.formaPagamento}
              </span>
            </div>
            {saleResult.desconto > 0 && (
              <div className="flex items-center justify-between text-emerald-700">
                <span>Desconto</span>
                <span className="font-mono">-{formatBRL(saleResult.desconto)}</span>
              </div>
            )}
            <div className="flex items-center justify-between pt-2 border-t border-cocoa-dark/10 font-bold">
              <span className="text-cocoa-dark">Total Simulado</span>
              <span className="font-mono font-black text-primary text-base">
                {formatBRL(saleResult.total)}
              </span>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-cocoa-dark/10 bg-cream/40">
          <button
            type="button"
            onClick={onNewSale}
            className="w-full py-3 px-4 rounded-xl bg-primary text-white text-xs font-black uppercase tracking-wider shadow-md shadow-primary/20 hover:bg-primary-hover active:scale-[0.98] transition-all flex items-center justify-center gap-2 cursor-pointer"
          >
            <RotateCcw size={15} />
            <span>Iniciar Nova Venda</span>
          </button>
        </div>
      </div>
    </div>
  );
}
