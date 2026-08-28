"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

ESQUEMA_TEMPERATURAS = pa.DataFrameSchema({})


def comparar_esquema(temperaturas: pl.DataFrame) -> list[str]:
    """Devuelve diferencias entre el esquema real y el esperado."""
    esquema_notebook = {
        "country": pl.String,
        "iso_alpha2": pl.String,
        "iso_alpha3": pl.String,
        "year": pl.Int64,
        "period": pl.String,
        "temperature_c": pl.Float64,
        "parameter": pl.String,
        "units": pl.String,
        "source_file": pl.String,
    }
    diferencias = []
    for columna, tipo in esquema_notebook.items():
        if temperaturas.schema.get(columna) != tipo:
            diferencias.append(temperaturas.schema.get(columna))
    return diferencias
    # raise NotImplementedError(
    #     "Completen comparar_esquema antes de ejecutar el programa."
    # )


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
