import numpy as np
from abc import abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from liquidation_task_tools.base import FeatureAnalyzer


class ModelBasedFeatureAnalyzer(FeatureAnalyzer):
    def __init__(self, name: str):
        super().__init__(name)
        self._model = None

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        pass

    def calculate(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._fit(X, y)
        return float(self._score(X, y))


class LinearRegressionAnalyzer(ModelBasedFeatureAnalyzer):
    def __init__(self):
        super().__init__("linear_regression_r2")

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self._model = LinearRegression()
        self._model.fit(X, y)

    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self._model.predict(X)
        return r2_score(y, preds)


class TreeR2Analyzer(ModelBasedFeatureAnalyzer):
    def __init__(self):
        super().__init__("tree_r2")

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self._model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
        )
        self._model.fit(X, y)

    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self._model.predict(X)
        return r2_score(y, preds)

