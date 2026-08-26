"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera import Check

from src.meteolab.constantes import PERIODOS_VALIDOS

ESQUEMA_TEMPERATURAS = pa.DataFrameSchema(
    {
        "country": pa.Column(pl.String),
        "iso_alpha2": pa.Column(pl.String),
        "iso_alpha3": pa.Column(pl.String),
        "year": pa.Column(pl.Int64, Check.in_range(1901, 2025)),
        "period": pa.Column(pl.String, Check.isin(PERIODOS_VALIDOS)),
        "temperature_c": pa.Column(pl.Float64, nullable=True),
        "parameter": pa.Column(pl.String, Check.eq("Mean Temperature")),
        "units": pa.Column(pl.String, Check.eq("degrees Celsius")),
        "source_file": pa.Column(pl.String),
    }
)


def comparar_esquema(temperaturas: pl.DataFrame) -> list[str]:
    """Devuelve diferencias entre el esquema real y el esperado."""
    raise NotImplementedError(
        "Completen comparar_esquema antes de ejecutar el programa."
    )


def validar_esquema(temperaturas: pl.DataFrame) -> None:
    """Comprueba los nombres y tipos de las columnas."""
    raise NotImplementedError(
        "Completen validar_esquema antes de ejecutar el programa."
    )


def validar_datos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Valida tipos, periodos, unidades y valores faltantes."""
    raise NotImplementedError(
        "Completen validar_datos antes de ejecutar el programa."
    )


def casos_que_fallan(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Devuelve los incumplimientos sin ocultar sus columnas."""
    raise NotImplementedError(
        "Completen casos_que_fallan antes de ejecutar el programa."
    )
