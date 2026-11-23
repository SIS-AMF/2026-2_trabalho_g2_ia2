import { Package, X } from "lucide-react";
import { formatBRL } from "../api/client";
import type { CatalogSku, ProductGroup } from "../types";

interface Props {
  group: ProductGroup | null;
  outOfStockSkus: Set<string>;
  onSelect: (sku: CatalogSku) => void;
  onClose: () => void;
}

export function ProductPresentationModal({
  group,
  outOfStockSkus,
  onSelect,
  onClose,
}: Props) {
  if (!group) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white border border-cocoa-dark/15 rounded-2xl w-full max-w-md shadow-2xl overflow-hidden">
        <div className="p-4 border-b border-cocoa-dark/10 flex items-center justify-between bg-cream/60">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
              <Package size={18} />
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-wider text-cocoa-dark/50">
                Escolha a apresentação
              </p>
              <h3 className="font-black text-sm text-cocoa-dark uppercase">
                {group.produto_base}
              </h3>
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

        <div className="p-4 flex flex-col gap-2 max-h-[65vh] overflow-y-auto scrollbar-thin">
          {group.skus.map((sku) => {
            const semEstoque = outOfStockSkus.has(sku.sku_norm);
            return (
              <button
                key={sku.sku_norm}
                type="button"
                disabled={semEstoque}
                onClick={() => {
                  onSelect(sku);
                  onClose();
                }}
                className={`flex items-center justify-between p-3 rounded-xl border text-left transition-all ${
                  semEstoque
                    ? "bg-cocoa-dark/5 border-cocoa-dark/10 opacity-50 cursor-not-allowed"
                    : "bg-white border-cocoa-dark/15 hover:border-primary hover:bg-primary/[0.03] cursor-pointer active:scale-[0.99]"
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sm text-cocoa-dark">
                      {sku.tipo_embalagem}{" "}
                      {sku.multiplicador > 1 ? `x${sku.multiplicador}` : "(Unidade)"}
                    </span>
                    {sku.economia_unitaria_pct > 0 && (
                      <span className="text-[10px] font-black px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700">
                        -{sku.economia_unitaria_pct.toFixed(0)}% / un
                      </span>
                    )}
                    {semEstoque && (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-rose-100 text-rose-700">
                        Sem estoque
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-cocoa-dark/60 mt-0.5 font-mono">
                    {formatBRL(sku.preco_unitario_ref)} / un • {sku.multiplicador} un/emb.
                  </p>
                </div>
                <span className="font-mono font-black text-base text-primary">
                  {formatBRL(sku.preco_pacote_ref)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
