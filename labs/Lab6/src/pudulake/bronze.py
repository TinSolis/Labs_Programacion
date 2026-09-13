"""Ingesta reproducible de las fuentes Parquet hacia Bronze."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_sources(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee las cuatro fuentes crudas y conserva exactamente su esquema."""
    parquet_container = {}
    for file in raw_dir.glob("*.parquet"):
        parquet_container[file.stem] = pl.read_parquet(file)
    for file in ["customers", "orders", "order_items", "payments"]:
        if file not in parquet_container:
            raise FileNotFoundError(f"No se encuentra {file}.")
    return parquet_container

    # raise NotImplementedError(
    #     "Completen read_sources antes de ejecutar el programa."
    # )
