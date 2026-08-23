"""Kernels de convolución que deben definir para la Etapa 6."""

import numpy as np

# Su código aquí: agreguen al menos cinco tuplas (nombre, kernel).
KERNELS: list[tuple[str, np.ndarray]] = [
    # Mantiene la imagen sin cambios.
    ("identidad", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # Resalta los bordes y cambios bruscos de intensidad.
    ("laplaciano", np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])),
    # Aumenta la nitidez resaltando los detalles de la imagen.
    ("enfoque", np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
    # Suaviza la imagen promediando cada píxel con sus vecinos.
    (
        "desenfoque",
        np.array(
            [
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
            ]
        ),
    ),
    # Da un efecto de relieve usando las diferencias entre píxeles vecinos.
    ("relieve", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])),
]
