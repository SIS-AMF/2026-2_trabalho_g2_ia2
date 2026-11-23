import { Boxes, RotateCcw, X } from "lucide-react";
import type { CatalogSku } from "../types";

interface Props {
  isOpen: boolean;
  skus: CatalogSku[];
  outOfStockSkus: Set<string>;
  onToggleSku: (skuNorm: string) => void;
  onResetAll: () => void;
  onClose: () => void;
}

export function StockManagerModal({
  isOpen,
  skus,
  outOfStockSkus,
  onToggleSku,
  onResetAll,
  onClose,
}: Props) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white border border-cocoa-dark/15 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        <div className="p-4 border-b border-cocoa-dark/10 flex items-center justify-between bg-cream/70">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
              <Boxes size={18} />
            </div>
            <div>
              <h3 className="font-black text-sm text-cocoa-dark uppercase">
                Simulador de Estoque da Feira (Filtro Top-K da IA)
              </h3>
              <p className="text-xs text-cocoa-dark/60">
                Desmarque produtos sem estoque: a IA pula automaticamente para o próximo SKU no ranking.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-full text-cocoa-dark/50 hover:bg-cocoa-dark/10 cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        <div className="p-4 overflow-y-auto scrollbar-thin grid grid-cols-1 sm:grid-cols-2 gap-2">
          {skus.map((sku) => {
            const emEstoque = !outOfStockSkus.has(sku.sku_norm);
            return (
              <label
                key={sku.sku_norm}
                className={`flex items-center justify-between gap-2 p-2.5 rounded-xl border text-xs cursor-pointer transition-all ${
                  emEstoque
                    ? "bg-white border-cocoa-dark/15 hover:border-primary/40"
                    : "bg-rose-50/70 border-rose-200 text-rose-800"
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <input
                    type="checkbox"
                    checked={emEstoque}
                    onChange={() => onToggleSku(sku.sku_norm)}
                    className="accent-primary w-4 h-4 shrink-0"
                  />
                  <span className={`truncate font-semibold ${!emEstoque ? "line-through opacity-75" : ""}`}>
                    {sku.display_name}
                  </span>
                </div>
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0 ${
                    emEstoque
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-rose-200 text-rose-800"
                  }`}
                >
                  {emEstoque ? "Ativo" : "Esgotado"}
                </span>
              </label>
            );
          })}
        </div>

        <div className="p-3 border-t border-cocoa-dark/10 bg-cream/40 flex items-center justify-between">
          <span className="text-xs text-cocoa-dark/60">
            Ativos: <strong>{skus.length - outOfStockSkus.size}</strong> / {skus.length} SKUs
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onResetAll}
              className="px-3 py-1.5 rounded-xl border border-cocoa-dark/15 text-xs font-bold text-cocoa-dark hover:bg-cocoa-dark/5 flex items-center gap-1.5 cursor-pointer"
            >
              <RotateCcw size={13} />
              Repor Todos
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-1.5 rounded-xl bg-primary text-white text-xs font-bold hover:bg-primary-hover cursor-pointer"
            >
              Aplicar
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
