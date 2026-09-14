"""Ingesta reproducible de las fuentes Parquet hacia Bronze."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_sources(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee las cuatro fuentes crudas y conserva exactamente su esquema."""
    orders = pl.read_parquet(raw_dir / "orders.parquet")
    customers = pl.read_parquet(raw_dir / "customers.parquet")
    order_items = pl.read_parquet(raw_dir / "order_items.parquet")
    payments = pl.read_parquet(raw_dir / "payments.parquet")
    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "payments": payments,
    }
