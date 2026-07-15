"""Testes da paleta de marca (src/theme.py).

Módulo puro de constantes/conversões; validamos o mapeamento hex->RGB e a cor por
severidade, que alimentam tanto o front-end (visual_report/main) quanto o PDF.
"""

from __future__ import annotations

from src import theme
from src.stride_models import SEVERITY_ORDER


def test_hex_to_rgb_basic():
    assert theme.hex_to_rgb("#000000") == (0, 0, 0)
    assert theme.hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert theme.hex_to_rgb("#EC0868") == (236, 8, 104)
    # aceita sem o '#'
    assert theme.hex_to_rgb("EC0868") == (236, 8, 104)


def test_severity_colors_cover_all_severities():
    # Toda severidade do schema tem uma cor definida (nenhuma cai no fallback).
    for sev in SEVERITY_ORDER:
        assert sev in theme.SEVERITY_COLORS
        assert theme.severity_color(sev) == theme.SEVERITY_COLORS[sev]


def test_severity_color_rgb_matches_hex():
    for sev, hex_color in theme.SEVERITY_COLORS.items():
        assert theme.severity_color_rgb(sev) == theme.hex_to_rgb(hex_color)


def test_unknown_severity_falls_back_without_raising():
    # Severidade fora do schema não lança — devolve o fallback neutro.
    color = theme.severity_color("Inexistente")
    assert color == theme._SEVERITY_FALLBACK
    assert theme.severity_color_rgb("Inexistente") == theme.hex_to_rgb(color)


def test_brand_tokens_are_valid_hex():
    for token in (theme.INK, theme.NAVY, theme.MAGENTA, theme.MAGENTA_HI, theme.TEXT_HI):
        assert token.startswith("#") and len(token) == 7
        theme.hex_to_rgb(token)  # não deve lançar
