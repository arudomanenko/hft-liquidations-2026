from __future__ import annotations

import math

import numpy as np
import polars as pl

from liquidation_task_tools.loaders import ParquetDataLoader

from . import postprocess


def weighted_regression_errors(
    scores: np.ndarray,
    pnl: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    target_badness = -pnl
    err = scores - target_badness
    weight_sum = float(weights.sum(dtype=np.float64))
    if weight_sum == 0.0:
        return np.nan, np.nan
    rmse = math.sqrt(float((weights * err * err).sum(dtype=np.float64)) / weight_sum)
    mae = float((weights * np.abs(err)).sum(dtype=np.float64)) / weight_sum
    return rmse, mae


def compute_filter_fraction_metrics(
    *,
    experiment_name: str,
    horizon_sec: int,
    eligible_size: int,
    eval_days: float,
    max_filter_fraction: float,
    cum_filtered_weight: np.ndarray,
    cum_filtered_weighted_pnl: np.ndarray,
    total_weight: float,
    total_weighted_pnl: float,
    pnl_all: float,
    weighted_rmse: float,
    weighted_mae: float,
) -> pl.DataFrame:
    max_filter_fraction = float(np.clip(max_filter_fraction, 0.0, 1.0))
    frac_steps = max(1, int(round(max_filter_fraction * 100)))

    rows = []
    for frac in np.linspace(0.0, max_filter_fraction, frac_steps + 1):
        k = int(eligible_size * frac)

        filtered_weight = float(cum_filtered_weight[k])
        filtered_weighted_pnl = float(cum_filtered_weighted_pnl[k])

        kept_weight = total_weight - filtered_weight
        kept_weighted_pnl = total_weighted_pnl - filtered_weighted_pnl

        pnl_kept = np.nan if kept_weight == 0.0 else kept_weighted_pnl / kept_weight
        pnl_filtered = np.nan if filtered_weight == 0.0 else filtered_weighted_pnl / filtered_weight

        rows.append(
            {
                "experiment": experiment_name,
                "horizon_sec": horizon_sec,
                "filter_fraction": float(frac),
                "score_bps": pnl_kept - pnl_all,
                "pnl_all_bps": pnl_all,
                "pnl_kept_bps": pnl_kept,
                "pnl_filtered_bps": pnl_filtered,
                "filtered_turnover_per_day": filtered_weight / eval_days,
                "kept_turnover_per_day": kept_weight / eval_days,
                "weighted_rmse_badness_bps": weighted_rmse,
                "weighted_mae_badness_bps": weighted_mae,
            }
        )
    return pl.DataFrame(rows)


def evaluate_predictions(
    scores: np.ndarray,
    pnl: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    loader: ParquetDataLoader,
    *,
    experiment_name: str,
    horizon_sec: int,
    max_filter_fraction: float,
) -> pl.DataFrame:
    eligible_scores, eligible_pnl, eligible_weights = postprocess.select_eligible_rows(
        scores, pnl, valid_mask, weights, horizon_sec
    )
    weighted_rmse, weighted_mae = weighted_regression_errors(eligible_scores, eligible_pnl, eligible_weights)
    cum_filtered_weight, cum_filtered_weighted_pnl, total_weight, total_weighted_pnl, pnl_all = (
        postprocess.worst_first_cumulatives(eligible_scores, eligible_pnl, eligible_weights)
    )
    return compute_filter_fraction_metrics(
        experiment_name=experiment_name,
        horizon_sec=horizon_sec,
        eligible_size=eligible_scores.size,
        eval_days=postprocess.eval_days(loader),
        max_filter_fraction=max_filter_fraction,
        cum_filtered_weight=cum_filtered_weight,
        cum_filtered_weighted_pnl=cum_filtered_weighted_pnl,
        total_weight=total_weight,
        total_weighted_pnl=total_weighted_pnl,
        pnl_all=pnl_all,
        weighted_rmse=weighted_rmse,
        weighted_mae=weighted_mae,
    )
