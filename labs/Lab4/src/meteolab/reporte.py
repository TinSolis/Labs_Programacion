"""Pipelines lazy para analizar únicamente temperaturas mensuales."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.meteolab.carga import escanear_temperaturas
from src.meteolab.constantes import PAISES_COMPARACION, RUTA_CSV
from src.meteolab.derivadas import agregar_fecha_mensual
from src.meteolab.limpieza import limpiar_temperaturas
from src.meteolab.metricas import (
    anomalias_mensuales,
    resumen_anual_desde_mensuales,
    resumen_mensual,
)


def _mensuales_lazy(
    ruta: Path,
    paises: list[str] | tuple[str, ...],
) -> pl.LazyFrame:
    return agregar_fecha_mensual(
        limpiar_temperaturas(
            escanear_temperaturas(ruta).filter(
                pl.col("iso_alpha3").is_in(paises)
            )
        )
    ).sort("country", "fecha")


def pipeline_mensual(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.LazyFrame:
    """Construye el flujo mensual sin ejecutarlo."""
    return _mensuales_lazy(ruta, paises)


def pipeline_resumen_mensual(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.LazyFrame:
    """Construye la climatología mensual."""
    return resumen_mensual(_mensuales_lazy(ruta, paises))


def pipeline_resumen_anual(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.LazyFrame:
    """Calcula medias anuales desde meses limpios."""
    return resumen_anual_desde_mensuales(_mensuales_lazy(ruta, paises))


def pipeline_anomalias(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
    umbral: float = 2.0,
) -> pl.LazyFrame:
    """Construye el flujo de anomalías mensuales."""
    return anomalias_mensuales(_mensuales_lazy(ruta, paises), umbral=umbral)


def ejecutar_reporte(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.DataFrame:
    """Materializa la climatología mensual."""
    return pipeline_resumen_mensual(ruta, paises).collect()


def plan_de_ejecucion(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
    optimizado: bool = True,
) -> str:
    """Devuelve el plan lazy como texto."""
    return pipeline_mensual(ruta, paises).explain(optimized=optimizado)
