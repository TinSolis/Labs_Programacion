"""Operaciones de procesamiento de imágenes para completar."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from src.pixellab.imagen import Imagen


class LibImagen:
    """Filtros y transformaciones que reciben y retornan ``Imagen``."""

    def to_negative(self, img_in: Imagen) -> Imagen:
        return img_in.__rsub__(255)

    def to_gray(self, img_in: Imagen) -> Imagen:
        # Su código aquí
        gray_R = img_in.imagen[:, :, 0] * 0.299
        gray_G = img_in.imagen[:, :, 1] * 0.587
        gray_B = img_in.imagen[:, :, 2] * 0.114
        gray = gray_R + gray_G + gray_B
        gray_saturado = saturar_check(
            np.stack([gray, gray, gray], axis=2).round().astype(int)
        )
        return Imagen(gray_saturado)

    def get_channel(self, img_in: Imagen, channel: str) -> Imagen:
        # Su código aquí
        base = np.zeros(img_in.imagen.shape, dtype=int)
        if channel == "r":
            base[:, :, 0] += img_in.imagen[:, :, 0]
        elif channel == "g":
            base[:, :, 1] += img_in.imagen[:, :, 1]
        elif channel == "b":
            base[:, :, 2] += img_in.imagen[:, :, 2]
        else:
            raise ValueError(
                f"Canal '{channel}' no válido. Valores posibles: 'r', 'g', o 'b'."
            )
        return Imagen(base)

    def flip(self, img_in: Imagen, axis: str) -> Imagen:
        # Su código aquí
        imagen = img_in.imagen.copy()
        if axis == "h":
            return Imagen(imagen[:, ::-1, :])
        elif axis == "v":
            return Imagen(imagen[::-1, :, :])
        else:
            raise ValueError(
                f"Eje '{axis}' no válido. Valores posibles: 'h' (horizontal) o 'v' (vertical)."
            )

    def set_saturation(self, img_in: Imagen, C: float) -> Imagen:
        # Su código aquí
        imagen = img_in.imagen.copy()
        img_gray = self.to_gray(img_in)
        scale = C * (Imagen(imagen) - img_gray).imagen.round().astype(int)
        return img_gray + scale

    def set_contrast(self, img_in: Imagen, C: float) -> Imagen:
        # Su código aquí
        F = 259 * (C + 255) / (255 * (259 - C))
        img_cpy = img_in.imagen.copy()
        last_step = F * (img_cpy - 128).round().astype(int)
        return Imagen(last_step) + 128

    def conv_channel(self, img_in: Imagen, kernel: np.ndarray) -> Imagen:
        """
        La función recibe una imagen y un kernel, aplicando una convolución 2D a cada canal de la imagen utilizando el
        kernel proporcionado. Los parámetros mode="same" y boundary="symm" de convolve2d permiten que la imagen
        resultante conserve el mismo alto y ancho que la imagen original. Para cada canal de color, convolve2d aplica
        el kernel sobre la vecindad de cada píxel y calcula el valor correspondiente del píxel de salida, recorriendo
        la imagen de izquierda a derecha y de arriba hacia abajo.
        Una vez finalizada la convolución, los valores se saturan al rango [0, 255] para mantener valores válidos
        para una imagen RGB.

        Parámetros:
            img_in: Imagen de entrada.
            kernel: Kernel 2D utilizado para realizar la convolución.
        Retorna:
            Imagen resultante después de aplicar la convolución a todos los canales.
        """
        img = img_in.imagen
        img_out = []
        for i in range(img.shape[-1]):
            img_channel = convolve2d(
                img[:, :, i], kernel, mode="same", boundary="symm"
            )
            img_out.append(img_channel)
        new_image = np.stack(img_out, axis=2)
        new_image[new_image > 255], new_image[new_image < 0] = 255, 0
        return Imagen(new_image.astype(int))


def saturar_check(img: np.ndarray) -> np.ndarray:
    imagen = img.copy().astype(int)
    imagen[imagen > 255] = 255
    imagen[imagen < 0] = 0
    return imagen
