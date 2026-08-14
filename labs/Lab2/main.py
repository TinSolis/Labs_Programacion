# import sys
# sys.path.insert(0, str(Path(__file__).parent / "src"))
import argparse
from pathlib import Path

from src.agroalerta.datos import cargar_lecturas
from src.agroalerta.reporte import contar_riesgos
from src.agroalerta.sensores import (
    SensorHumedad,
    SensorTemperatura,
    SensorViento,
)


def main():
    parser = argparse.ArgumentParser(description="AgroAlerta")
    parser.add_argument("--fecha", default="2026-06-15")
    args = parser.parse_args()

    sensores = [
        SensorTemperatura(0, 40),
        SensorViento(25),
        SensorHumedad(85),
    ]

    lecturas = cargar_lecturas(Path(r"data\lecturas.csv"), args.fecha)
    conteo = contar_riesgos(sensores, lecturas)

    print(f"Estación Parcela Norte — {args.fecha}")
    print(f"Temperatura    {conteo['temperatura']} lecturas en riesgo")
    print(f"Viento         {conteo['viento']} lecturas en riesgo")
    print(f"Humedad        {conteo['humedad']} lecturas en riesgo")
    print()
    print(
        f"Total: {conteo['temperatura'] + conteo['viento'] + conteo['humedad']} situaciones de riesgo"
    )


if __name__ == "__main__":
    main()
