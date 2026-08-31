"""Funciones para revisar nulos y claves temporales."""

from __future__ import annotations

import polars as pl

from src.meteolab.constantes import Tabla


def resumen_de_nulos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Devuelve conteos y porcentajes de nulos por columna."""

    nulos_df = pl.DataFrame(
        {
            "columna": temperaturas.columns,
            "nulos": [
                temperaturas[c].null_count() for c in temperaturas.columns
            ],
            "porcentaje": [
                temperaturas[c].null_count() / temperaturas.height * 100
                for c in temperaturas.columns
            ],
        }
    )
    return nulos_df


def claves_repetidas(temperaturas: Tabla) -> Tabla:
    """Cuenta repeticiones de país, año y periodo."""
    conteo_df = (
        temperaturas.group_by(["country", "year", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )
    return conteo_df


def limpiar_temperaturas(temperaturas: Tabla) -> Tabla:
    """Conserva el contrato de periodos y los nulos válidos."""
    mensuales = (
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    )
    limpio_df = temperaturas.filter(
        pl.col("period").is_in(mensuales)
        & pl.col("temperature_c").is_not_null()
    )
    return limpio_df
