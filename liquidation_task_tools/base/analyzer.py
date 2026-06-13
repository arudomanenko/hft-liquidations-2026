from abc import ABC, abstractmethod


class FeatureAnalyzer(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def calculate(self, *data) -> float:
        pass

    @property
    def name(self) -> str:
        return self._name

