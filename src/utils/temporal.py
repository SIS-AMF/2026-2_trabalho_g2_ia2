"""Funções puras de codificação temporal cíclica, economia unitária e rótulos."""

from __future__ import annotations

import math


def calcular_economia_unitaria_pct(
    preco_unitario: float,
    preco_avulso_ref: float,
) -> float:
    """Calcula o percentual de economia unitária de uma embalagem frente ao avulso."""
    if preco_avulso_ref <= 0.0:
        return 0.0
    economia = (1.0 - (float(preco_unitario) / float(preco_avulso_ref))) * 100.0
    return round(max(0.0, economia), 2)


def codificar_hora_ciclica(hora: int) -> tuple[float, float]:
    """Codifica a hora (0..23) em coordenadas cíclicas (sin, cos)."""
    angulo = 2.0 * math.pi * (int(hora) % 24) / 24.0
    return round(math.sin(angulo), 6), round(math.cos(angulo), 6)


def classificar_faixa_horaria(hora: int) -> str:
    """Classifica a hora em Manhã (<12h), Tarde (12h-17h) ou Noite (>=18h)."""
    h = int(hora)
    if h < 12:
        return "Manhã"
    if h < 18:
        return "Tarde"
    return "Noite"


def eh_fim_de_semana(dia_semana: int) -> bool:
    """Retorna True para sábado (5) ou domingo (6) na convenção pandas."""
    return int(dia_semana) >= 5


def montar_display_name(
    produto_base: str,
    tipo_embalagem: str,
    multiplicador: int,
) -> str:
    """Monta o rótulo amigável para exibição no PDV e relatórios."""
    mult = int(multiplicador)
    if mult <= 1:
        return f"{produto_base} (Unidade)"
    return f"{produto_base} ({tipo_embalagem} x{mult})"
