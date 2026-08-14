from .sensores import Sensor


def contar_riesgos(sensores: list[Sensor], lecturas: dict) -> dict[str, int]:
    conteo = {}
    for sensor in sensores:
        cantidad = 0
        for valor in lecturas.get(sensor.nombre, []):
            if sensor.es_riesgo(valor):
                cantidad += 1
        conteo[sensor.nombre] = cantidad
    print(conteo)
    return conteo
