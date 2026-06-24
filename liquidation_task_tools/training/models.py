from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class CatBoostChunkRegressor:
    def __init__(self, **model_params: Any) -> None:
        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor(**model_params)
        self._fitted = False

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CatBoostChunkRegressor":
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if self._fitted:
            fit_kwargs["init_model"] = self._model
        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(X.shape[0], dtype=np.float32)
        return np.asarray(self._model.predict(X), dtype=np.float32)


def build_model(model_type: str, task: str, model_params: Optional[Dict[str, Any]] = None) -> Any:
    if task.lower() != "regression":
        raise ValueError("Only 'regression' task is supported")
    if model_type.lower() != "catboost_chunk":
        raise ValueError("Only 'catboost_chunk' model_type is supported")
    params = dict(
        {
            "loss_function": "RMSE",
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.03,
            "random_seed": 42,
            "allow_writing_files": False,
            "verbose": 200,
        }
        | dict(model_params or {})
    )
    return CatBoostChunkRegressor(**params)


def supports_partial_fit(model: Any) -> bool:
    return callable(getattr(model, "partial_fit", None))


def supports_warm_start_growth(model: Any) -> bool:
    return bool(getattr(model, "warm_start", False)) and callable(getattr(model, "fit", None))

