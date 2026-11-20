"""Carga del modelo desde su artefacto."""

from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from .artifacts import ErrorArtefacto, leer_metadata
from .constants import FEATURES


def cargar_modelo(directorio: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Carga el artefacto de `directorio` y devuelve (modelo, metadata).

    Antes de cargar el modelo comprueba que las variables de la metadata sean
    exactamente `FEATURES`, en ese orden, y que `sklearn_version` coincida con
    la versión de scikit-learn instalada. Si algo no calza lanza
    `ErrorArtefacto` con un mensaje que nombra la diferencia.
    """
    metadata = leer_metadata(directorio)
    features = metadata.get("features")
    if (features is not None) and features != FEATURES:
        raise ErrorArtefacto(
            "Las variables de la metadata no es exactamente FEATURES aclarados."
        )
    sk_ver = metadata.get("sklearn_version")
    if sk_ver != sklearn.__version__:
        raise ErrorArtefacto(
            f"Las versiones de scikit-learn no se coinciden, versión del metadato: {sk_ver} ."
        )

    modelo = joblib.load(Path(directorio) / "model.joblib")

    return modelo, metadata
