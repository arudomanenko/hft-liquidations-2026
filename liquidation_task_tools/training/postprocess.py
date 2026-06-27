from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from liquidation_task_tools.constants import SECOND
from liquidation_task_tools.labeling import calculate_model_pnl
from liquidation_task_tools.loaders import ParquetDataLoader

from .experiment import (
    BINANCE_BTC_BBO,
    BINANCE_BTC_TRADES,
    HORIZONS_SEC,
    build_sample_weights,
)
from .model_pipeline import RegressionPipeline


def collect_predictions(
    pipeline: RegressionPipeline,
    loader: ParquetDataLoader,
    tau_idx: int,
    on_chunk: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores_parts: list[np.ndarray] = []
    pnl_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []

    loader.reset()
    try:
        for chunk_idx, chunk in enumerate(loader, start=1):
            scores = np.asarray(pipeline.predict_chunk(chunk, proba=False), dtype=np.float32).reshape(-1)
            stats = calculate_model_pnl(
                chunk[BINANCE_BTC_TRADES],
                chunk[BINANCE_BTC_BBO],
                horizons_sec=HORIZONS_SEC,
            )
            scores_parts.append(scores)
            pnl_parts.append(np.asarray(stats["trade_pnl"][:, tau_idx], dtype=np.float32))
            valid_parts.append(np.asarray(stats["valid_mask"][:, tau_idx], dtype=bool))
            weights_parts.append(build_sample_weights(chunk).astype(np.float32, copy=False))
            if on_chunk is not None:
                on_chunk(chunk_idx)
    finally:
        loader.reset()

    return (
        np.concatenate(scores_parts),
        np.concatenate(pnl_parts),
        np.concatenate(valid_parts),
        np.concatenate(weights_parts),
    )


def select_eligible_rows(
    scores: np.ndarray,
    pnl: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    horizon_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(valid_mask)
    if eligible.size == 0:
        raise ValueError(f"No valid rows in evaluation window for horizon {horizon_sec}s")
    return scores[eligible], pnl[eligible], weights[eligible]


def worst_first_cumulatives(
    scores: np.ndarray,
    pnl: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    worst_first_idx = np.argsort(scores)[::-1]
    worst_weights = weights[worst_first_idx]
    worst_weighted_pnl = (pnl[worst_first_idx] * worst_weights).astype(np.float64)

    cum_filtered_weight = np.concatenate(([0.0], np.cumsum(worst_weights, dtype=np.float64)))
    cum_filtered_weighted_pnl = np.concatenate(([0.0], np.cumsum(worst_weighted_pnl, dtype=np.float64)))

    total_weight = float(weights.sum(dtype=np.float64))
    total_weighted_pnl = float((pnl * weights).sum(dtype=np.float64))
    pnl_all = np.nan if total_weight == 0.0 else total_weighted_pnl / total_weight

    return cum_filtered_weight, cum_filtered_weighted_pnl, total_weight, total_weighted_pnl, pnl_all


def eval_days(loader: ParquetDataLoader) -> float:
    return max(1.0, (loader.data_end_ts - loader.data_start_ts) / SECOND / (24 * 60 * 60))
