from abc import ABC, abstractmethod


class Sensor(ABC):
    def __init__(
        self,
        nombre: str,
        unidad: str,
    ) -> None:
        self.nombre = nombre
        self.unidad = unidad

    @abstractmethod
    def es_riesgo(self, valor: float) -> bool: ...

    @property
    @abstractmethod
    def range_seguro(self): ...


class SensorTemperatura(Sensor):
    def __init__(self, minimo: float, maximo: float) -> None:
        super().__init__("temperatura", "°C")
        self._minimo = minimo
        self._maximo = maximo

    def es_riesgo(self, valor: float) -> bool:
        return valor < self._minimo or valor > self._maximo

    @property
    def range_seguro(self):
        raise NotImplementedError


class SensorViento(Sensor):
    def __init__(self, maximo: float) -> None:
        super().__init__("viento", "km/h")
        self._maximo = maximo

    def es_riesgo(self, valor: float) -> bool:
        return valor > self._maximo

    @property
    def range_seguro(self):
        raise NotImplementedError


class SensorHumedad(Sensor):
    def __init__(self, maximo: float) -> None:
        super().__init__("humedad", "%")
        self._maximo = maximo

    def es_riesgo(self, valor: float) -> bool:
        return valor > self._maximo

    @property
    def range_seguro(self):
        raise NotImplementedError
