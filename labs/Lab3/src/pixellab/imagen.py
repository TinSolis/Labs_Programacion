"""Clase ``Imagen``: contenedor de imágenes sobre el que se opera con NumPy."""

from __future__ import annotations

import numpy as np


class Imagen:
    """Contenedor de imágenes RGB.

    Completen el constructor y los operadores de esta clase siguiendo el
    contrato del enunciado y los tests de ``tests/test_imagen.py``.
    """

    def __init__(self, img: np.ndarray) -> None:
        # Su código aquí
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Debes entregar un arreglo de numpy como argumento del constructor de Imagen"
            )
        elif img.ndim != 3:
            raise ValueError(
                "Debes entregar un arreglo de numpy con 3 dimensiones"
            )
        elif img.shape[-1] != 3:
            raise ValueError("Debes entregar un arreglo de numpy con 3 canales")
        self.imagen = img

    def __add__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        img = self.imagen.copy()
        try:
            other_value = obtener_operador(img, other)
            return Imagen(saturar_check(img + other_value))
        except ValueError as err:
            raise ValueError("Las dimensiones no calzan") from err

    def __radd__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        return self.__add__(other)

    def __sub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        img = self.imagen.copy()
        try:
            other_value = obtener_operador(img, other)
            return Imagen(saturar_check(img - other_value))
        except ValueError as err:
            raise ValueError("Las dimensiones no calzan") from err

    def __rsub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        img = self.imagen.copy()
        try:
            other_value = obtener_operador(img, other)
            return Imagen(saturar_check(other_value - img))
        except ValueError as err:
            raise ValueError("Las dimensiones no calzan") from err

    def __mul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        img = self.imagen.copy()
        try:
            other_value = obtener_operador(img, other)
            if isinstance(other_value, np.ndarray):
                other_dim = other.ndim
                if img.shape[other_dim:] != other.shape[:other_dim]:
                    raise ValueError("no se coinciden las dimensiones.")
            return Imagen(saturar_check(img * other))
        except ValueError as err:
            raise ValueError("Las dimensiones no calzan") from err

    def __rmul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        return self.__mul__(other)


def obtener_operador(
    img: np.ndarray, other: int | float | np.ndarray | Imagen
) -> int | float | np.ndarray:
    img_size = img.shape
    if isinstance(other, Imagen):
        return obtener_operador(img, other.imagen)
    elif isinstance(other, np.ndarray) and (img_size != other.shape):
        raise ValueError("Debes entregar un arreglo de misma dimension")
    return other


def saturar_check(img: np.ndarray):
    imagen = img.copy().astype(int)
    imagen[imagen > 255] = 255
    imagen[imagen < 0] = 0
    return imagen
