import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from liquidation_task_tools.base import Feature
from liquidation_task_tools.training.model_pipeline import (
    ClassificationPipeline,
    FeatureSpec,
    ModelPipeline,
    RegressionPipeline,
)


class IdentityTradeValueFeature(Feature):
    def __init__(self) -> None:
        super().__init__(name="identity_trade_value")

    def calculate(self, **data):
        trades = data["trades"]
        feature_values = trades["value"].to_numpy().reshape(-1, 1)
        feature_ts = trades["timestamp"].to_numpy()
        max_used_ts = feature_ts
        return feature_values, feature_ts, max_used_ts

    def calculate_max_used_ts(self, **data):
        return data["trades"]["timestamp"].to_numpy()


class DummyPartialFitClassifier:
    def __init__(self) -> None:
        self.partial_fit_calls = 0
        self.first_call_classes = None
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self._threshold = 0.0

    def partial_fit(self, X, y, sample_weight=None, classes=None):
        self.partial_fit_calls += 1
        if self.first_call_classes is None and classes is not None:
            self.first_call_classes = np.asarray(classes)
            self.classes_ = np.asarray(classes)
        self._threshold = float(np.median(X[:, 0]))
        return self

    def predict(self, X):
        return (X[:, 0] >= self._threshold).astype(np.int8)

    def predict_proba(self, X):
        labels = self.predict(X).astype(np.float32)
        prob_class_1 = np.where(labels > 0, 0.8, 0.2).astype(np.float32)
        return np.column_stack([1.0 - prob_class_1, prob_class_1])


class DummyWarmStartRegressor:
    def __init__(self, n_estimators: int = 1) -> None:
        self.warm_start = True
        self.n_estimators = n_estimators
        self.fit_calls = 0
        self.n_estimators_history = []
        self._prediction = 0.0

    def set_params(self, **params):
        if "n_estimators" in params:
            self.n_estimators = int(params["n_estimators"])
        return self

    def fit(self, X, y, sample_weight=None):
        self.fit_calls += 1
        self.n_estimators_history.append(int(self.n_estimators))
        self._prediction = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._prediction, dtype=np.float32)


class StaticChunkLoader:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0
        self.reset_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def reset(self):
        self._idx = 0
        self.reset_calls += 1


def make_chunk(timestamps, values):
    return {
        "trades": pl.DataFrame(
            {
                "timestamp": np.asarray(timestamps, dtype=np.int64),
                "value": np.asarray(values, dtype=np.float32),
            }
        )
    }


def build_classification_target(chunk):
    values = chunk["trades"]["value"].to_numpy()
    return (values > 0.0).astype(np.int8)


def build_regression_target(chunk):
    values = chunk["trades"]["value"].to_numpy()
    return values * 0.5 + 1.0


def build_feature_specs():
    return [
        FeatureSpec(
            feature=IdentityTradeValueFeature(),
            source_map={"trades": "trades"},
        )
    ]


class TestModelPipeline(unittest.TestCase):
    def test_model_pipeline_partial_fit_and_predict(self):
        chunk = make_chunk([1, 2, 3, 4], [-2.0, -1.0, 1.0, 2.0])
        loader = StaticChunkLoader([chunk])
        model = DummyPartialFitClassifier()
        pipeline = ModelPipeline(
            model=model,
            feature_specs=build_feature_specs(),
            data_loader=loader,
            target_builder=build_classification_target,
            sample_weight_builder=lambda c: np.ones(c["trades"].height, dtype=np.float32),
        )

        pipeline.fit()
        predictions = pipeline.predict_chunk(chunk, proba=False)

        self.assertEqual(model.partial_fit_calls, 1)
        self.assertIsNone(model.first_call_classes)
        self.assertGreaterEqual(loader.reset_calls, 2)
        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))

    def test_model_pipeline_logs_every_i_chunk(self):
        chunks = [
            make_chunk([1, 2], [-2.0, 2.0]),
            make_chunk([3, 4], [-1.0, 1.0]),
            make_chunk([5, 6], [-0.5, 0.5]),
        ]
        pipeline = ModelPipeline(
            model=DummyPartialFitClassifier(),
            feature_specs=build_feature_specs(),
            data_loader=StaticChunkLoader(chunks),
            target_builder=build_classification_target,
            sample_weight_builder=lambda c: np.ones(c["trades"].height, dtype=np.float32),
        )

        with patch("builtins.print") as mock_print:
            pipeline.fit(log_i_chunk=2)

        self.assertEqual(mock_print.call_count, 2)
        self.assertIn("chunk=2", mock_print.call_args_list[0].args[0])
        self.assertEqual(mock_print.call_args_list[1].args[0], "Training finished: chunks=3")

    def test_model_pipeline_logs_every_chunk_by_default(self):
        chunks = [
            make_chunk([1, 2], [-2.0, 2.0]),
            make_chunk([3, 4], [-1.0, 1.0]),
            make_chunk([5, 6], [-0.5, 0.5]),
        ]
        pipeline = ModelPipeline(
            model=DummyPartialFitClassifier(),
            feature_specs=build_feature_specs(),
            data_loader=StaticChunkLoader(chunks),
            target_builder=build_classification_target,
            sample_weight_builder=lambda c: np.ones(c["trades"].height, dtype=np.float32),
        )

        with patch("builtins.print") as mock_print:
            pipeline.fit()

        self.assertEqual(mock_print.call_count, 4)
        self.assertIn("chunk=1", mock_print.call_args_list[0].args[0])
        self.assertIn("chunk=2", mock_print.call_args_list[1].args[0])
        self.assertIn("chunk=3", mock_print.call_args_list[2].args[0])
        self.assertEqual(mock_print.call_args_list[3].args[0], "Training finished: chunks=3")

    def test_model_pipeline_fit_raises_on_invalid_log_i_chunk(self):
        chunk = make_chunk([1, 2], [-1.0, 1.0])
        pipeline = ModelPipeline(
            model=DummyPartialFitClassifier(),
            feature_specs=build_feature_specs(),
            data_loader=StaticChunkLoader([chunk]),
            target_builder=build_classification_target,
            sample_weight_builder=lambda c: np.ones(c["trades"].height, dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            pipeline.fit(log_i_chunk=0)

        with self.assertRaises(ValueError):
            pipeline.fit(log_i_chunk=-2)


class TestClassificationPipeline(unittest.TestCase):
    def test_classification_pipeline_builds_filter_and_keep_masks(self):
        chunk = make_chunk([10, 20, 30, 40], [-3.0, -0.5, 0.5, 3.0])
        loader = StaticChunkLoader([chunk])
        pipeline = ClassificationPipeline(
            model=DummyPartialFitClassifier(),
            feature_specs=build_feature_specs(),
            data_loader=loader,
            target_builder=build_classification_target,
            sample_weight_builder=lambda c: np.ones(c["trades"].height, dtype=np.float32),
            classes=np.array([0, 1], dtype=np.int8),
        )

        pipeline.fit()
        probas = pipeline.predict_chunk(chunk, proba=True)
        filter_mask = pipeline.predict_filter_mask_chunk(chunk, threshold=0.5)
        keep_mask = pipeline.predict_keep_mask_chunk(chunk, threshold=0.5)

        self.assertEqual(probas.shape, (4, 2))
        np.testing.assert_array_equal(
            pipeline.model.first_call_classes,
            np.array([0, 1], dtype=np.int8),
        )
        self.assertEqual(filter_mask.dtype, np.int8)
        self.assertEqual(keep_mask.dtype, np.int8)
        np.testing.assert_array_equal(keep_mask, (1 - filter_mask).astype(np.int8))


class TestRegressionPipeline(unittest.TestCase):
    def test_regression_pipeline_warm_start_fit_and_predict(self):
        first_chunk = make_chunk([100, 200, 300, 400], [1.0, 2.0, 3.0, 4.0])
        second_chunk = make_chunk([500, 600, 700, 800], [5.0, 6.0, 7.0, 8.0])
        loader = StaticChunkLoader([first_chunk, second_chunk])

        model = DummyWarmStartRegressor(n_estimators=1)
        pipeline = RegressionPipeline(
            model=model,
            feature_specs=build_feature_specs(),
            data_loader=loader,
            target_builder=build_regression_target,
            sample_weight_builder=lambda c: np.ones(c["trades"].height, dtype=np.float32),
            warm_start_estimators_step=5,
        )

        pipeline.fit()
        predictions = pipeline.predict_chunk(first_chunk)

        self.assertEqual(model.fit_calls, 2)
        self.assertEqual(model.n_estimators_history, [5, 10])
        self.assertGreaterEqual(loader.reset_calls, 2)
        self.assertEqual(predictions.shape, (4,))


if __name__ == "__main__":
    unittest.main()
