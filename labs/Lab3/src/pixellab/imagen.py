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

    def _obtener_operando(
        self, other: int | float | np.ndarray | Imagen
    ) -> int | float | np.ndarray:
        # checkear si es que es imagen
        if isinstance(other, Imagen):
            # checkear si calza la imagen
            if other.imagen.shape != self.imagen.shape:
                # si no error
                raise ValueError("Las dimensiones de las imágenes no calzan")
            # si esta bien se de vuelve
            return other.imagen
        # si no es imagen se de vuelve
        return other

    # funcionar para saturar los limites
    def _saturar(self, resultado: np.ndarray) -> Imagen:
        resultado = resultado.astype(int)
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(np.copy(resultado))

    def __add__(self, other: int | float | np.ndarray | Imagen) -> Imagen:

        operando = self._obtener_operando(other)
        return self._saturar(self.imagen + operando)

    def __radd__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        return self.__add__(other)

    def __sub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        operando = self._obtener_operando(other)
        return self._saturar(self.imagen - operando)

    def __rsub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        operando = self._obtener_operando(other)
        return self._saturar(operando - self.imagen)

    def __mul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        operando = self._obtener_operando(other)
        return self._saturar(self.imagen * operando)

    def __rmul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        return self.__mul__(other)
