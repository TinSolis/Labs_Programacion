"""Transformaciones y reglas críticas de las entidades Silver."""

from __future__ import annotations

import polars as pl

from src.pudulake.contracts import ContractViolation

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

ALLOWED_STATUSES = [
    "approved",
    "canceled",
    "created",
    "delivered",
    "invoiced",
    "processing",
    "shipped",
    "unavailable",
]

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _assert_no_invalid_dates(frame: pl.DataFrame) -> None:
    for col in DATE_COLS:
        if frame.schema[col] == pl.Null:
            continue

        texto = pl.col(col).cast(pl.String)
        invalidas = frame.filter(
            texto.is_not_null()
            & texto.str.strptime(
                pl.Datetime, format=DATE_FORMAT, strict=False
            ).is_null()
        )
        if invalidas.height > 0:
            raise ContractViolation(f"{col} contiene fechas no interpretables.")


def build_orders(orders: pl.DataFrame) -> pl.DataFrame:
    """Tipa fechas de órdenes y comprueba su secuencia temporal."""
    _assert_no_invalid_dates(orders)

    result = orders.with_columns(
        [
            pl.col(col)
            .cast(pl.String)
            .str.strptime(pl.Datetime, format=DATE_FORMAT)
            for col in DATE_COLS
        ]
    )

    if (
        result.filter(~pl.col("order_status").is_in(ALLOWED_STATUSES)).height
        > 0
    ):
        raise ContractViolation(
            "order_status contiene valores fuera del contrato."
        )

    result = result.with_columns(
        (
            (pl.col("order_status") == "delivered")
            & pl.col("order_delivered_customer_date").is_null()
        ).alias("delivery_timestamp_missing")
    )

    if (
        result.filter(
            (pl.col("order_status") == "delivered")
            & pl.col("order_delivered_customer_date").is_not_null()
            & (
                pl.col("order_delivered_customer_date")
                < pl.col("order_purchase_timestamp")
            )
        ).height
        > 0
    ):
        raise ContractViolation(
            "Hay órdenes delivered con entrega anterior a la compra."
        )

    return result


def build_customers(customers: pl.DataFrame) -> pl.DataFrame:
    """Conserva clientes y verifica la relación uno a uno con customer_id."""
    return customers.select("customer_id", "customer_unique_id")


def build_order_items(items: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los ítems no tengan precios ni fletes negativos."""
    for col in ("price", "freight_value"):
        if items.filter(pl.col(col) < 0).height > 0:
            raise ContractViolation(f"{col} contiene montos negativos.")
        if (
            items.filter(
                pl.col(col).is_not_null() & ~pl.col(col).is_finite()
            ).height
            > 0
        ):
            raise ContractViolation(f"{col} contiene valores no finitos.")

    return items


def build_payments(payments: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los pagos no tengan montos negativos."""
    if payments.filter(pl.col("payment_value") < 0).height > 0:
        raise ContractViolation("payment_value contiene montos negativos.")

    if (
        payments.filter(
            pl.col("payment_value").is_not_null()
            & ~pl.col("payment_value").is_finite()
        ).height
        > 0
    ):
        raise ContractViolation("payment_value contiene valores no finitos.")

    return payments


def validate_relationships(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    payments: pl.DataFrame,
) -> None:
    """Verifica las claves foráneas antes de construir productos Gold."""
    checks = [
        (
            "orders.customer_id -> customers.customer_id",
            orders.join(
                customers.select("customer_id"),
                on="customer_id",
                how="anti",
            ).height,
        ),
        (
            "order_items.order_id -> orders.order_id",
            items.join(
                orders.select("order_id"),
                on="order_id",
                how="anti",
            ).height,
        ),
        (
            "payments.order_id -> orders.order_id",
            payments.join(
                orders.select("order_id"),
                on="order_id",
                how="anti",
            ).height,
        ),
    ]

    for nombre, huerfanas in checks:
        if huerfanas > 0:
            raise ContractViolation(
                f"Relación huérfana: {nombre} ({huerfanas} filas)."
            )
