"""Funciones para construir fechas mensuales."""

from __future__ import annotations

import polars as pl

from src.meteolab.constantes import MESES, Tabla


def agregar_fecha_mensual(mensuales: Tabla) -> Tabla:
    """Agrega month y una fecha nativa de Polars."""
    return mensuales.with_columns(
        pl.col("period").replace(MESES).alias("month"),
        pl.datetime(pl.col("year"), pl.col("period"), 1).alias("fecha"),
    )
