from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Feature(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def calculate_max_used_ts(self, **data) -> np.ndarray:
        raise NotImplementedError

