"""Agregaciones sobre las temperaturas medias mensuales."""

from __future__ import annotations

import polars as pl

from src.meteolab.constantes import Tabla


def _filtrar_paises(
    mensuales: Tabla,
    paises: list[str] | tuple[str, ...] | None,
) -> Tabla:
    if paises is None:
        return mensuales
    return mensuales.filter(pl.col("iso_alpha3").is_in(paises))


def resumen_mensual(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calcula la climatología mensual por país."""
    mensuales = _filtrar_paises(mensuales, paises)
    return (
        mensuales.group_by("iso_alpha3", "country", "month")
        .agg(
            pl.len().alias("observaciones"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
        .sort("iso_alpha3", "month")
    )


def resumen_anual_desde_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calcula medias anuales usando únicamente filas mensuales."""
    mensuales = _filtrar_paises(mensuales, paises)
    return (
        mensuales.group_by("iso_alpha3", "country", "year")
        .agg(
            pl.len().alias("meses_disponibles"),
            pl.col("temperature_c").mean().alias("temperature_mean"),
        )
        .sort("iso_alpha3", "year")
    )


def anomalias_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    umbral: float = 2.0,
) -> pl.DataFrame | pl.LazyFrame:
    """Marca anomalías usando una ventana por país y mes."""
    return (
        mensuales.with_columns(
            pl.col("temperature_c")
            .mean()
            .over("iso_alpha3", "month")
            .alias("temperature_mean_month"),
        )
        .with_columns(
            (
                (pl.col("temperature_c") - pl.col("temperature_mean_month"))
                / pl.col("temperature_c").std().over("iso_alpha3", "month")
            ).alias("standardized_anomaly"),
        )
        .with_columns(
            (pl.col("standardized_anomaly").abs() > umbral).alias("is_anomaly"),
        )
    )
