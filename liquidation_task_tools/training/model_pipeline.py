from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import polars as pl

from liquidation_task_tools.base import Feature
from liquidation_task_tools.loaders import ParquetDataLoader
from liquidation_task_tools.validation import ValidationErrorReason, validate

from .models import supports_partial_fit

Chunk = Mapping[str, pl.DataFrame]
TargetBuilder = Callable[[Chunk], np.ndarray]
SampleWeightBuilder = Callable[[Chunk], Optional[np.ndarray]]
GroupIdBuilder = Callable[[Chunk], np.ndarray]


@dataclass(frozen=True)
class FeatureSpec:
    feature: Feature
    source_map: Dict[str, str]
    params: Dict[str, Any] = field(default_factory=dict)


class ModelPipeline:
    def __init__(
        self,
        model: Any,
        feature_specs: Sequence[FeatureSpec],
        data_loader: ParquetDataLoader,
        sample_weight_builder: Optional[SampleWeightBuilder],
        target_builder: TargetBuilder,
        group_id_builder: Optional[GroupIdBuilder] = None,
    ):
        self._feature_specs = list(feature_specs)
        self._data_loader = data_loader
        self._target_builder = target_builder
        self._sample_weight_builder = sample_weight_builder
        self._group_id_builder = group_id_builder
        self._model = model

    @property
    def model(self) -> Any:
        return self._model

    def _build_feature_kwargs(self, spec: FeatureSpec, chunk: Chunk) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for arg_name, chunk_key in spec.source_map.items():
            if chunk_key not in chunk:
                raise KeyError(f"Missing '{chunk_key}' in chunk for feature '{spec.feature.name}'")
            kwargs[arg_name] = chunk[chunk_key]
        kwargs.update(spec.params)
        return kwargs

    def _compute_feature_matrix(self, chunk: Chunk) -> np.ndarray:
        columns = []

        for spec in self._feature_specs:
            feature_kwargs = self._build_feature_kwargs(spec, chunk)
            feature_values, feature_ts, max_used_ts = spec.feature.calculate(**feature_kwargs)

            feature_values = np.asarray(feature_values)
            feature_ts = np.asarray(feature_ts)
            max_used_ts = np.asarray(max_used_ts)

            trades_ts = feature_ts
            trades_df = feature_kwargs.get("trades")
            trades_ts = trades_df["timestamp"].to_numpy()

            validation_result = validate(
                features=feature_values,
                trades_ts=trades_ts,
                max_used_ts=max_used_ts,
                features_ts=feature_ts,
            )
            if validation_result != ValidationErrorReason.OK:
                raise ValueError(f"Feature validation failed with reason: {validation_result.name}")

            columns.append(feature_values.astype(np.float32, copy=False))

        X = np.hstack(columns)
        return X

    def _compute_target(self, chunk: Chunk) -> np.ndarray:
        return np.asarray(self._target_builder(chunk))

    def _compute_sample_weights(self, chunk: Chunk) -> Optional[np.ndarray]:
        if self._sample_weight_builder is None:
            return None
        weights = self._sample_weight_builder(chunk)
        if weights is None:
            return None
        return np.asarray(weights)

    def _compute_group_id(self, chunk: Chunk) -> Optional[np.ndarray]:
        if self._group_id_builder is None:
            return None
        return np.asarray(self._group_id_builder(chunk))

    @staticmethod
    def _filter_training_rows(
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        group_id: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        keep_mask = np.isfinite(y)
        if sample_weight is not None:
            keep_mask = keep_mask & np.isfinite(sample_weight)

        if keep_mask.ndim != 1:
            keep_mask = keep_mask.reshape(-1)

        if keep_mask.shape[0] != y.shape[0]:
            raise ValueError("Filtered row mask size must match target size")

        if np.all(keep_mask):
            return X, y, sample_weight, group_id

        X_filtered = X[keep_mask]
        y_filtered = y[keep_mask]
        sample_weight_filtered = sample_weight[keep_mask] if sample_weight is not None else None
        group_id_filtered = group_id[keep_mask] if group_id is not None else None
        return X_filtered, y_filtered, sample_weight_filtered, group_id_filtered

    def _fit_stream(self, data_stream: Iterable[Chunk]) -> int:
        trained_chunks = 0
        first_chunk = True

        for chunk in data_stream:
            X = self._compute_feature_matrix(chunk)
            y = self._compute_target(chunk)
            if y.shape[0] != X.shape[0]:
                raise ValueError("Target size must match feature matrix rows")

            sample_weight = self._compute_sample_weights(chunk)
            if sample_weight is not None and sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight size must match feature matrix rows")
            group_id = self._compute_group_id(chunk)
            if group_id is not None and group_id.shape[0] != X.shape[0]:
                raise ValueError("group_id size must match feature matrix rows")

            X, y, sample_weight, group_id = self._filter_training_rows(
                X, y, sample_weight, group_id
            )
            if X.shape[0] == 0:
                continue

            self._fit_on_chunk(X, y, sample_weight, group_id, is_first_chunk=first_chunk)
            first_chunk = False
            trained_chunks += 1

        return trained_chunks

    @staticmethod
    def _build_fit_kwargs(
        sample_weight: Optional[np.ndarray], group_id: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if group_id is not None:
            fit_kwargs["group_id"] = group_id
        return fit_kwargs

    def _partial_fit_first_chunk_kwargs(self) -> Dict[str, Any]:
        return {}

    def _fit_partial_fit_chunk(
        self, X: np.ndarray, y: np.ndarray, fit_kwargs: Dict[str, Any], is_first_chunk: bool
    ) -> None:
        if is_first_chunk:
            fit_kwargs.update(self._partial_fit_first_chunk_kwargs())
        self._model.partial_fit(X, y, **fit_kwargs)

    def _fit_on_chunk(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        group_id: Optional[np.ndarray],
        is_first_chunk: bool,
    ) -> None:
        fit_kwargs = self._build_fit_kwargs(sample_weight, group_id)

        if supports_partial_fit(self._model):
            self._fit_partial_fit_chunk(X, y, fit_kwargs, is_first_chunk)
            return

        raise ValueError("Model must implement partial_fit")

    def fit(
        self,
    ) -> "ModelPipeline":
        self._data_loader.reset()

        try:
            trained_chunks = self._fit_stream(self._data_loader)
        finally:
            self._data_loader.reset()

        if trained_chunks == 0:
            raise ValueError("No non-empty chunks were used for training")

        return self

    def _prepare_prediction_matrix(self, chunk: Chunk) -> np.ndarray:
        X = self._compute_feature_matrix(chunk)
        return X

    def predict_chunk(self, chunk: Chunk, proba: bool = True) -> np.ndarray:
        X = self._prepare_prediction_matrix(chunk)
        if proba and hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return self._model.predict(X)


class RankingPipeline(ModelPipeline):
    pass
