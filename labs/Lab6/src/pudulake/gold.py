"""Productos analíticos Gold de Pudubella."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def build_rfm_exclusions(
    orders: pl.DataFrame, payments: pl.DataFrame
) -> pl.DataFrame:
    """Registra órdenes entregadas sin pago para excluirlas de RFM."""
    pagos = payments.select("order_id").unique()
    return (
        orders.filter(pl.col("order_status") == "delivered")
        .join(pagos, on="order_id", how="anti")
        .select("order_id", "customer_id", "order_purchase_timestamp")
        .with_columns(pl.lit("delivered_order_without_payment").alias("reason"))
    )


def build_sales_daily(
    orders: pl.DataFrame, items: pl.DataFrame
) -> pl.DataFrame:
    """Construye ventas de ítems por fecha de compra y órdenes entregadas."""
    entregadas = orders.filter(pl.col("order_status") == "delivered")
    unidas = entregadas.join(items, on="order_id")
    return (
        unidas.with_columns(
            pl.col("order_purchase_timestamp").dt.date().alias("sale_date")
        )
        .group_by("sale_date")
        .agg(
            pl.col("price").sum().alias("items_sold_value"),
            pl.col("order_id").n_unique().alias("delivered_orders"),
        )
        .sort("sale_date")
    )


def _assign_segment(
    recency_days: int,
    frequency: int,
    monetary: float,
    rules: dict[str, dict[str, float | int]],
) -> str:
    for name, rule in rules.items():
        ok = True
        if (
            "max_recency_days" in rule
            and recency_days > rule["max_recency_days"]
        ):
            ok = False
        if (
            "min_recency_days" in rule
            and recency_days < rule["min_recency_days"]
        ):
            ok = False
        if "min_frequency" in rule and frequency < rule["min_frequency"]:
            ok = False
        if "min_monetary" in rule and monetary < rule["min_monetary"]:
            ok = False
        if ok:
            return name.capitalize()
    return "Other"


def build_customer_rfm(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    payments: pl.DataFrame,
    segments: dict[str, Any],
) -> pl.DataFrame:
    """Calcula RFM de compras entregadas y aplica reglas congeladas."""
    rules = segments["segments"]
    excluded = build_rfm_exclusions(orders, payments)["order_id"]

    pagos_por_orden = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("order_payment_total")
    )

    elegibles = (
        orders.filter(pl.col("order_status") == "delivered")
        .join(pagos_por_orden, on="order_id", how="inner")
        .filter(~pl.col("order_id").is_in(excluded))
        .join(customers, on="customer_id", how="left")
    )

    rfm = elegibles.group_by("customer_unique_id").agg(
        pl.col("order_purchase_timestamp").max().alias("last_purchase"),
        pl.col("order_id").n_unique().alias("frequency"),
        pl.col("order_payment_total").sum().alias("monetary"),
    )

    ultima_global = elegibles.select(
        pl.col("order_purchase_timestamp").max()
    ).item()
    fecha_referencia = ultima_global.date() + timedelta(days=1)

    rfm = rfm.with_columns(
        (pl.lit(fecha_referencia) - pl.col("last_purchase").dt.date())
        .dt.total_days()
        .alias("recency_days")
    )

    segmentos = [
        _assign_segment(
            row["recency_days"], row["frequency"], row["monetary"], rules
        )
        for row in rfm.iter_rows(named=True)
    ]
    return rfm.with_columns(pl.Series("segment", segmentos))
