"""Agregaciones sobre las temperaturas medias mensuales."""

from __future__ import annotations

import polars as pl

from src.meteolab.limpieza import limpiar_temperaturas


def resumen_mensual(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calcula la climatología mensual por país."""
    if paises is None:
        return mensuales.group_by(["country", "iso_alpha3", "month"]).agg(
            pl.len().alias("observaciones"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
    else:
        return (
            mensuales.group_by(["country", "iso_alpha3", "month"])
            .agg(
                pl.len().alias("observaciones"),
                pl.col("temperature_c")
                .mean()
                .round(2)
                .alias("temperature_mean"),
            )
            .filter(pl.col("iso_alpha3").is_in(paises))
        )


def resumen_anual_desde_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calcula medias anuales usando únicamente filas mensuales."""
    mensuales = limpiar_temperaturas(mensuales)
    if paises is None:
        return mensuales.group_by(["country", "iso_alpha3", "year"]).agg(
            pl.len().alias("meses_disponibles"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
    else:
        return (
            mensuales.group_by(["country", "iso_alpha3", "year"])
            .agg(
                pl.len().alias("observaciones"),
                pl.col("meses_disponibles")
                .mean()
                .round(2)
                .alias("temperature_mean"),
            )
            .filter(pl.col("iso_alpha3").is_in(paises))
        )


def anomalias_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    umbral: float = 2.0,
) -> pl.DataFrame | pl.LazyFrame:
    """Marca anomalías usando una ventana por país y mes."""
    return mensuales.with_columns(
        pl.col("temperature_c")
        .mean()
        .round(2)
        .over(["iso_alpha3", "month"])
        .alias("temperature_mean_month"),
        (
            (
                pl.col("temperature_c")
                - pl.col("temperature_c").mean().over(["iso_alpha3", "month"])
            )
            / pl.col("temperature_c").std().over(["iso_alpha3", "month"])
        ).alias("standardized_anomaly"),
    ).with_columns(
        (pl.col("standardized_anomaly").abs() >= umbral)
        .fill_null(False)
        .alias("is_anomaly")
    )
