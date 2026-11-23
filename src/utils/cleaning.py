"""Funções puras de auditoria, saneamento, multiplicadores e reconciliação contábil.

Cada função desta camada comporta-se como uma operação matemática determinística:
recebe valores escalares específicos e retorna um resultado direto, sem estado
global nem dependência de DataFrame ou I/O.
"""

from __future__ import annotations

import re

from src.config import (
    EVENTOS_CONSOLIDADOS,
    LIMITE_SUBTOTAL_CONSOLIDADO,
    LIMITE_UNIDADES_CONSOLIDADO,
    MAPA_EMBALAGEM_SEM_MULTIPLICADOR,
    MAPA_PAGAMENTO_GRUPO,
    MAPA_PAGAMENTO_PADRAO,
    PONTO_VENDA_PADRAO,
    TOLERANCIA_INTEIRO,
    TOLERANCIA_MONETARIA,
)

_REGEX_EMBALAGEM_COM_MULT = re.compile(
    r"\((Pacote|Fardo|Bandeja|Caixa|Display)\s+x(\d+)\)\s*$",
    flags=re.IGNORECASE,
)
_REGEX_EMBALAGEM_SEM_MULT = re.compile(
    r"\((Pacote|Fardo)\)\s*$",
    flags=re.IGNORECASE,
)
_REGEX_PARENTESES = re.compile(r"\s*\([^)]*\)\s*")


def validar_desconto_em_moeda(
    subtotal: float,
    valor_desconto: float,
    valor_total: float,
    tolerancia: float = TOLERANCIA_MONETARIA,
) -> tuple[bool, float, str]:
    """Valida se o desconto já está em moeda (R$) ou converte se estiver em %.

    Retorna:
        (valido, desconto_em_reais, status)
        onde status in {"ja_em_moeda", "convertido_de_percentual", "inconsistente"}.
    """
    sub = float(subtotal)
    desc = float(valor_desconto)
    tot = float(valor_total)

    if abs((sub - desc) - tot) < tolerancia:
        return True, desc, "ja_em_moeda"

    desconto_se_pct = sub * (desc / 100.0)
    if abs((sub - desconto_se_pct) - tot) < tolerancia:
        return True, round(desconto_se_pct, 4), "convertido_de_percentual"

    return False, desc, "inconsistente"


def item_quantidade_valida(quantidade: int | float) -> bool:
    """Retorna True somente para itens com quantidade estritamente positiva."""
    return float(quantidade) > 0.0


def classificar_pedido_gratuito(valor_total: float) -> bool:
    """Identifica pedidos com valor total zerado (desconto de 100%)."""
    return float(valor_total) <= 0.0


def normalizar_texto_produto(nome_bruto: str) -> str:
    """Remove espaços nas extremidades e colapsa múltiplos espaços internos."""
    return re.sub(r"\s+", " ", str(nome_bruto)).strip()


def extrair_produto_base(
    nome_limpo: str,
    mapa_sinonimos: dict[str, str] | None = None,
) -> str:
    """Extrai o nome do produto base sem sufixos entre parênteses.

    Aplica apenas substituição exata presente em ``mapa_sinonimos`` (nunca faz
    fusão automática por similaridade para evitar unir produtos distintos).
    """
    sem_parenteses = _REGEX_PARENTESES.sub(" ", nome_limpo)
    base = re.sub(r"\s+", " ", sem_parenteses).strip()
    if mapa_sinonimos:
        return mapa_sinonimos.get(base, base)
    return base


def normalizar_evento(
    evento_bruto: str,
    mapa_eventos: dict[str, tuple[str, str]] | None = None,
    ponto_venda_padrao: str = PONTO_VENDA_PADRAO,
) -> tuple[str, str]:
    """Normaliza o nome do evento e extrai o ponto de venda quando aplicável."""
    evento_limpo = re.sub(r"\s+", " ", str(evento_bruto)).strip()
    if mapa_eventos and evento_limpo in mapa_eventos:
        return mapa_eventos[evento_limpo]
    return evento_limpo, ponto_venda_padrao


def normalizar_metodo_pagamento(
    metodo_bruto: str,
    mapa_padrao: dict[str, str] | None = None,
    mapa_grupo: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Padroniza o método de pagamento e retorna (metodo_padronizado, grupo)."""
    limpo = re.sub(r"\s+", " ", str(metodo_bruto)).strip()
    padrao_map = MAPA_PAGAMENTO_PADRAO if mapa_padrao is None else mapa_padrao
    grupo_map = MAPA_PAGAMENTO_GRUPO if mapa_grupo is None else mapa_grupo

    padronizado = padrao_map.get(limpo, limpo)
    grupo = grupo_map.get(padronizado, padronizado)
    return padronizado, grupo


def extrair_embalagem_e_multiplicador(
    nome_limpo: str,
    produto_base: str,
    preco_bruto: float,
    mapa_sem_mult: dict[tuple[str, str], int] | None = None,
) -> tuple[str, int]:
    """Resolve deterministamente (tipo_embalagem, multiplicador) de um item."""
    match_mult = _REGEX_EMBALAGEM_COM_MULT.search(nome_limpo)
    if match_mult:
        tipo_embalagem = match_mult.group(1).capitalize()
        multiplicador = int(match_mult.group(2))
        # Exceção verificada: Rapadura de Melado Grão Moído (Pacote x1) a R$ 20,00
        # corresponde ao pacote de 3 unidades (3 x R$ 6,67 = R$ 20,00).
        if (
            tipo_embalagem == "Pacote"
            and multiplicador == 1
            and produto_base == "Rapadura de Melado Grão Moído"
            and float(preco_bruto) >= 18.0
        ):
            return "Pacote", 3
        return tipo_embalagem, multiplicador

    match_sem_mult = _REGEX_EMBALAGEM_SEM_MULT.search(nome_limpo)
    if match_sem_mult:
        tipo_embalagem = match_sem_mult.group(1).capitalize()
        mapa = (
            MAPA_EMBALAGEM_SEM_MULTIPLICADOR
            if mapa_sem_mult is None
            else mapa_sem_mult
        )
        chave = (produto_base, tipo_embalagem)
        if chave not in mapa:
            raise ValueError(
                f"Embalagem sem multiplicador explícito não mapeada: {chave} "
                f"(produto='{nome_limpo}', preco={preco_bruto})"
            )
        return tipo_embalagem, mapa[chave]

    return "Unidade", 1


def identificar_registro_em_pacotes(
    nome_limpo: str,
    evento_original: str,
    multiplicador: int,
    pedido_divergente: bool,
) -> bool:
    """Identifica o Caso C (quantidade registrada em pacotes com preço de pacote)."""
    if multiplicador <= 1 or pedido_divergente:
        return False
    return (
        evento_original == "ExpoDireto"
        or "(Pacote)" in nome_limpo
        or "(Fardo)" in nome_limpo
        or "(Pacote x1)" in nome_limpo
    )


def reconciliar_quantidade_e_preco(
    qtd_bruta: int,
    preco_bruto: float,
    multiplicador: int,
    pedido_divergente: bool,
    registro_em_pacotes: bool,
    tolerancia_inteiro: float = TOLERANCIA_INTEIRO,
) -> tuple[int, int, float, float]:
    """Reconcilia quantidades e preços retornando:
    (qtd_unidades_base, qtd_pacotes, preco_unitario, preco_pacote).

    Casos:
        A) multiplicador == 1: venda unitária direta.
        B) pedido_divergente (ExpoDireto): qtd já em unidades base, preço é de pacote.
        C) registro_em_pacotes (ExpoDireto 11 linhas / 1 Moranguinho / Pacote x1):
           qtd está em pacotes e preço é de pacote.
        D) padrão normalizado das demais feiras: qtd em unidades base, preço unitário.
    """
    qtd = int(qtd_bruta)
    preco = float(preco_bruto)
    mult = int(multiplicador)

    if qtd <= 0:
        raise ValueError(f"Quantidade deve ser positiva, recebido: {qtd}")
    if mult <= 0:
        raise ValueError(f"Multiplicador deve ser positivo, recebido: {mult}")

    if mult == 1:
        return qtd, qtd, preco, preco

    if pedido_divergente:
        qtd_pacotes_float = qtd / mult
        if abs(qtd_pacotes_float - round(qtd_pacotes_float)) > tolerancia_inteiro:
            raise ValueError(
                f"qtd_pacotes fracionária no Caso B: qtd={qtd}, mult={mult}"
            )
        return qtd, int(round(qtd_pacotes_float)), preco / mult, preco

    if registro_em_pacotes:
        return qtd * mult, qtd, preco / mult, preco

    qtd_pacotes_float = qtd / mult
    if abs(qtd_pacotes_float - round(qtd_pacotes_float)) > tolerancia_inteiro:
        raise ValueError(
            f"qtd_pacotes fracionária no Caso D: qtd={qtd}, mult={mult}"
        )
    return qtd, int(round(qtd_pacotes_float)), preco, preco * mult


def montar_sku_canonico(produto_base: str, multiplicador: int) -> str:
    """Gera a chave canônica produto_com_multiplicador (ex.: 'Rapadura Assada (x6)')."""
    return f"{produto_base} (x{int(multiplicador)})"


def calcular_valor_item(qtd_unidades_base: int, preco_unitario: float) -> float:
    """Calcula o valor monetário total de um item reconciliado."""
    return round(int(qtd_unidades_base) * float(preco_unitario), 6)


def validar_fechamento_pedido(
    soma_itens: float,
    pedido_subtotal: float,
    tolerancia: float = TOLERANCIA_MONETARIA,
) -> bool:
    """Valida se a soma dos valores dos itens fecha com o subtotal do pedido."""
    return abs(float(soma_itens) - float(pedido_subtotal)) < tolerancia


def classificar_pedido_consolidado(
    subtotal: float,
    qtd_total_unidades: int,
    evento_original: str,
    limite_subtotal: float = LIMITE_SUBTOTAL_CONSOLIDADO,
    limite_unidades: int = LIMITE_UNIDADES_CONSOLIDADO,
    eventos_consolidados: frozenset[str] = EVENTOS_CONSOLIDADOS,
) -> bool:
    """Sinaliza pedidos consolidados / fechamento de caixa / atacado."""
    return (
        float(subtotal) >= limite_subtotal
        or int(qtd_total_unidades) >= limite_unidades
        or evento_original in eventos_consolidados
    )


def classificar_pedido_varejo(is_consolidado: bool, is_gratuito: bool) -> bool:
    """Retorna True se o pedido pertence ao escopo de balcão (varejo)."""
    return (not bool(is_consolidado)) and (not bool(is_gratuito))
