"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera import Check

from src.meteolab.constantes import ESQUEMA_CRU, PERIODOS_VALIDOS

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
    diferencias: list[str] = []
    esquema = temperaturas.schema

    for columna, tipo in ESQUEMA_CRU.items():
        if columna not in esquema:
            diferencias.append(f"Falta la columna '{columna}'.")
        elif esquema[columna] != tipo:
            diferencias.append(
                f"La columna '{columna}' tiene tipo {esquema[columna]}, "
                f"se esperaba {tipo}."
            )

    for columna in esquema:
        if columna not in ESQUEMA_CRU:
            diferencias.append(f"Sobra la columna '{columna}'.")

    return diferencias


def validar_esquema(temperaturas: pl.DataFrame) -> None:
    """Comprueba los nombres y tipos de las columnas."""
    diferencias = comparar_esquema(temperaturas)
    if diferencias:
        raise ValueError(diferencias[0])


def validar_datos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Valida tipos, periodos, unidades y valores faltantes."""
    validar_esquema(temperaturas)
    return ESQUEMA_TEMPERATURAS.validate(temperaturas)


def casos_que_fallan(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Devuelve los incumplimientos sin ocultar sus columnas."""
    try:
        ESQUEMA_TEMPERATURAS.validate(temperaturas, lazy=True)
    except pa.errors.SchemaErrors as error:
        return error.failure_cases
    return pl.DataFrame()
