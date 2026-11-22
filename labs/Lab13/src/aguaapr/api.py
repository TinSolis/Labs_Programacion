"""Rutas de inferencia: información del modelo, vida y predicción."""

from fastapi import APIRouter, Depends

from src.aguaapr.artifacts import a_tabla, predecir
from src.aguaapr.contracts import RespuestaPrediccion, Servicios
from src.aguaapr.deps import crear_dependencia_clave
from src.aguaapr.validation import MedicionAgua


def crear_router_inferencia(servicios: Servicios) -> APIRouter:
    """Construye el router con tres rutas.

    - `GET /`: describe el modelo. Responde las métricas de validación de la
      metadata, la entrada (cada columna con su rango) y la
      salida.
    - `GET /health`: ruta de vida, pública. Responde
      `{"estado": "ok"}`.
    - `POST /predict`: recibe una `MedicionAgua` en el cuerpo y responde una
      `RespuestaPrediccion` con la predicción y su probabilidad. Exige la
      clave de API con `crear_dependencia_clave`.
    """
    router = APIRouter()

    @router.get("/")
    def describir_modelo():
        return {
            "metricas_validacion": servicios.metadata["metricas_validacion"],
            "entrada": {
                "ph": [0, 14],
                "Hardness": [0, 500],
                "Solids": [0, 100000],
                "Chloramines": [0, 15],
                "Sulfate": [0, 600],
                "Conductivity": [0, 1000],
                "Organic_carbon": [0, 50],
                "Trihalomethanes": [0, 200],
                "Turbidity": [0, 10],
            },
            "salida": {"prediccion": "0 = no potable, 1 = potable"},
        }

    # GET /health
    @router.get("/health")
    def health():
        return {"estado": "ok"}

    # POST /predict
    @router.post("/predict")
    def predict(
        medicion: MedicionAgua,
        _: None = Depends(crear_dependencia_clave(servicios)),
    ):
        tabla = a_tabla([medicion])
        clases, probabilidades = predecir(servicios.modelo, tabla)

        return RespuestaPrediccion(
            prediccion=int(clases[0]),
            probabilidad=round(float(probabilidades[0]), 4),
        )

    return router
