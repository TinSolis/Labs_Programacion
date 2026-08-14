from src.agroalerta.reporte import contar_riesgos
from src.agroalerta.sensores import (
    SensorHumedad,
    SensorTemperatura,
    SensorViento,
)

temp1 = SensorTemperatura(0, 40)
wind1 = SensorViento(25)
humi1 = SensorHumedad(85)

# La función es_riesgo() retorna True si para el sensor


def test_temperatura_riesgosa():
    assert temp1.es_riesgo(-5)


def test_temperatura_no_riesgosa():
    assert not temp1.es_riesgo(20)


def test_viento_no_riesgoso():
    assert not wind1.es_riesgo(15)


def test_contar_riesgos():
    sensors = [temp1, wind1, humi1]

    datos = {
        "temperatura": [-5, 20, 40, 50],
        "viento": [15, 26],
        "humedad": [56, 88],
    }

    assert contar_riesgos(sensors, datos) == {
        "temperatura": 2,
        "humedad": 1,
        "viento": 1,
    }
