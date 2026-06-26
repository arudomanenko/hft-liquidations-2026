from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from liquidation_task_tools.constants import BYBIT_LIQUIDATION_DELAY_US, SECOND
from liquidation_task_tools.features import (
    BboDepthImbalanceValue,
    BboMicroPricePremiumBps,
    BboMidDeltaBps,
    BboMidSmoothDeltaBps,
    BboOrderFlowImbalance,
    BboOrderFlowImbalanceNorm,
    BboSpreadBps,
    BboTopDepthLog,
    BboVolumeImbalance,
    LiqudationClusterStrength,
    LiqudationClusterTotalNotional,
    TradeBboEdgeBps,
    TradeFlowCountLog,
    TradeFlowImbalance,
    TradeFlowNotionalLog,
    TradeFlowToxicity,
    TradeSide,
    TradeSideLiquidationStrength,
    TradeSignedNotionalLog,
)
from liquidation_task_tools.labeling import calculate_model_pnl
from liquidation_task_tools.loaders import ParquetDataLoader
from liquidation_task_tools.training import FeatureSpec, RegressionPipeline, build_model


BINANCE_BTC_BBO = "binance_booktickers_btc"
BINANCE_BTC_LIQ = "binance_liquidations_btc"
BINANCE_BTC_TRADES = "binance_trades_btc"
BYBIT_BTC_LIQ = "bybit_liquidations_btc"

PARQUET_NAMES = [
    BINANCE_BTC_BBO,
    BINANCE_BTC_LIQ,
    BINANCE_BTC_TRADES,
    BYBIT_BTC_LIQ,
]

HORIZONS_SEC = (30, 120, 300)
TARGET_MAX_HORIZON_SEC = max(HORIZONS_SEC)
MAX_FEATURE_LOOKBACK_SEC = 301
MIN_TURNOVER_PER_DAY = 500_000.0


@dataclass(frozen=True)
class Experiment:
    name: str


def _resolve_data_root() -> Path:
    candidates = (
        Path.cwd() / "liquidation_task_tools" / "data",
        Path.cwd() / "liquidation_task" / "data",
        Path.cwd() / "data",
        Path.cwd().parent / "liquidation_task_tools" / "data",
        Path.cwd().parent / "liquidation_task" / "data",
        Path.cwd().parent / "data",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find data directory in known locations")


def _fs(feature, source_map: dict[str, str], window_us: int | None = None, **params) -> FeatureSpec:
    feature_params = dict(params)
    if window_us is not None:
        feature_params["window_sec"] = window_us
    return FeatureSpec(feature=feature, source_map=source_map, params=feature_params)


def _binance_liq_fs(feature, window_us: int) -> FeatureSpec:
    return _fs(
        feature,
        {"liquidations": BINANCE_BTC_LIQ, "trades": BINANCE_BTC_TRADES},
        window_us=window_us,
    )


def _bybit_liq_fs(feature, window_us: int) -> FeatureSpec:
    return _fs(
        feature,
        {"liquidations": BYBIT_BTC_LIQ, "trades": BINANCE_BTC_TRADES},
        window_us=window_us,
        timestamp_shift_us=BYBIT_LIQUIDATION_DELAY_US,
    )


def _bbo_fs(feature, window_us: int) -> FeatureSpec:
    return _fs(
        feature,
        {"bbo": BINANCE_BTC_BBO, "trades": BINANCE_BTC_TRADES},
        window_us=window_us,
    )


def _trade_fs(feature, window_us: int | None = None) -> FeatureSpec:
    return _fs(feature, {"trades": BINANCE_BTC_TRADES}, window_us=window_us)


def _build_feature_specs(horizon_sec: int) -> list[FeatureSpec]:
    w5 = 5 * SECOND
    w15 = 15 * SECOND
    w30 = 30 * SECOND

    if horizon_sec == 30:
        return [
            _trade_fs(TradeSide()),
            _trade_fs(TradeSignedNotionalLog()),
            _bbo_fs(BboSpreadBps(), w30),
            _bbo_fs(BboTopDepthLog(), w30),
            _bbo_fs(BboVolumeImbalance(name="bbo_vol_imb_15s"), w15),
            _bbo_fs(BboVolumeImbalance(name="bbo_vol_imb_30s"), w30),
            _bbo_fs(BboDepthImbalanceValue(name="bbo_depth_imb_value_15s"), w15),
            _bbo_fs(BboDepthImbalanceValue(name="bbo_depth_imb_value_30s"), w30),
            _bbo_fs(BboMicroPricePremiumBps(name="bbo_microprice_premium_15s"), w15),
            _bbo_fs(BboMicroPricePremiumBps(name="bbo_microprice_premium_30s"), w30),
            _bbo_fs(TradeBboEdgeBps(name="trade_bbo_edge_15s"), w15),
            _bbo_fs(TradeBboEdgeBps(name="trade_bbo_edge_30s"), w30),
            _bbo_fs(BboMidSmoothDeltaBps(name="bbo_mid_smooth_delta_15s"), w15),
            _bbo_fs(BboMidSmoothDeltaBps(name="bbo_mid_smooth_delta_30s"), w30),
            _bbo_fs(BboMidDeltaBps(name="bbo_mid_delta_bps_5s"), w5),
            _bbo_fs(BboMidDeltaBps(name="bbo_mid_delta_bps_15s"), w15),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_5s"), w5),
            _bbo_fs(BboOrderFlowImbalanceNorm(name="bbo_ofi_norm_5s"), w5),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_15s"), w15),
            _bbo_fs(BboOrderFlowImbalanceNorm(name="bbo_ofi_norm_15s"), w15),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_30s"), w30),
            _trade_fs(TradeFlowImbalance(name="trade_flow_imbalance_5s"), w5),
            _trade_fs(TradeFlowImbalance(name="trade_flow_imbalance_15s"), w15),
            _trade_fs(TradeFlowToxicity(name="trade_flow_toxicity_5s"), w5),
            _trade_fs(TradeFlowToxicity(name="trade_flow_toxicity_15s"), w15),
            _trade_fs(TradeFlowNotionalLog(name="trade_flow_notional_log_15s"), w15),
            _trade_fs(TradeFlowNotionalLog(name="trade_flow_notional_log_30s"), w30),
            _trade_fs(TradeFlowCountLog(name="trade_flow_count_log_15s"), w15),
            _trade_fs(TradeFlowCountLog(name="trade_flow_count_log_30s"), w30),
            _binance_liq_fs(LiqudationClusterStrength(name="binance_liq_strength_15s"), w15),
            _binance_liq_fs(LiqudationClusterStrength(name="binance_liq_strength_30s"), w30),
            _binance_liq_fs(LiqudationClusterTotalNotional(name="binance_liq_total_notional_30s"), w30),
            _binance_liq_fs(TradeSideLiquidationStrength(name="trade_side_binance_liq_strength_30s"), w30),
            _bybit_liq_fs(LiqudationClusterStrength(name="bybit_liq_strength_15s"), w15),
            _bybit_liq_fs(LiqudationClusterStrength(name="bybit_liq_strength_30s"), w30),
            _bybit_liq_fs(LiqudationClusterTotalNotional(name="bybit_liq_total_notional_30s"), w30),
        ]

    if horizon_sec == 120:
        w60 = 60 * SECOND
        w120 = 120 * SECOND
        return [
            _trade_fs(TradeSide()),
            _trade_fs(TradeSignedNotionalLog()),
            _bbo_fs(BboSpreadBps(), w30),
            _bbo_fs(BboTopDepthLog(), w30),
            _bbo_fs(BboVolumeImbalance(name="bbo_vol_imb_30s"), w30),
            _bbo_fs(BboVolumeImbalance(name="bbo_vol_imb_120s"), w120),
            _bbo_fs(BboDepthImbalanceValue(name="bbo_depth_imb_value_30s"), w30),
            _bbo_fs(BboDepthImbalanceValue(name="bbo_depth_imb_value_120s"), w120),
            _bbo_fs(BboMicroPricePremiumBps(name="bbo_microprice_premium_30s"), w30),
            _bbo_fs(BboMicroPricePremiumBps(name="bbo_microprice_premium_120s"), w120),
            _bbo_fs(TradeBboEdgeBps(name="trade_bbo_edge_30s"), w30),
            _bbo_fs(TradeBboEdgeBps(name="trade_bbo_edge_120s"), w120),
            _bbo_fs(BboMidSmoothDeltaBps(name="bbo_mid_smooth_delta_30s"), w30),
            _bbo_fs(BboMidSmoothDeltaBps(name="bbo_mid_smooth_delta_120s"), w120),
            _bbo_fs(BboMidDeltaBps(name="bbo_mid_delta_bps_15s"), w15),
            _bbo_fs(BboMidDeltaBps(name="bbo_mid_delta_bps_60s"), w60),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_15s"), w15),
            _bbo_fs(BboOrderFlowImbalanceNorm(name="bbo_ofi_norm_15s"), w15),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_60s"), w60),
            _bbo_fs(BboOrderFlowImbalanceNorm(name="bbo_ofi_norm_60s"), w60),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_120s"), w120),
            _trade_fs(TradeFlowImbalance(name="trade_flow_imbalance_15s"), w15),
            _trade_fs(TradeFlowImbalance(name="trade_flow_imbalance_60s"), w60),
            _trade_fs(TradeFlowToxicity(name="trade_flow_toxicity_15s"), w15),
            _trade_fs(TradeFlowToxicity(name="trade_flow_toxicity_60s"), w60),
            _trade_fs(TradeFlowNotionalLog(name="trade_flow_notional_log_60s"), w60),
            _trade_fs(TradeFlowNotionalLog(name="trade_flow_notional_log_120s"), w120),
            _trade_fs(TradeFlowCountLog(name="trade_flow_count_log_60s"), w60),
            _trade_fs(TradeFlowCountLog(name="trade_flow_count_log_120s"), w120),
            _binance_liq_fs(LiqudationClusterStrength(name="binance_liq_strength_30s"), w30),
            _binance_liq_fs(LiqudationClusterStrength(name="binance_liq_strength_120s"), w120),
            _binance_liq_fs(LiqudationClusterTotalNotional(name="binance_liq_total_notional_120s"), w120),
            _binance_liq_fs(TradeSideLiquidationStrength(name="trade_side_binance_liq_strength_120s"), w120),
            _bybit_liq_fs(LiqudationClusterStrength(name="bybit_liq_strength_30s"), w30),
            _bybit_liq_fs(LiqudationClusterStrength(name="bybit_liq_strength_120s"), w120),
            _bybit_liq_fs(LiqudationClusterTotalNotional(name="bybit_liq_total_notional_120s"), w120),
        ]

    if horizon_sec == 300:
        w30 = 30 * SECOND
        w120 = 120 * SECOND
        w300 = 300 * SECOND
        return [
            _trade_fs(TradeSide()),
            _trade_fs(TradeSignedNotionalLog()),
            _bbo_fs(BboSpreadBps(), w30),
            _bbo_fs(BboTopDepthLog(), w30),
            _bbo_fs(BboVolumeImbalance(name="bbo_vol_imb_30s"), w30),
            _bbo_fs(BboVolumeImbalance(name="bbo_vol_imb_300s"), w300),
            _bbo_fs(BboDepthImbalanceValue(name="bbo_depth_imb_value_30s"), w30),
            _bbo_fs(BboDepthImbalanceValue(name="bbo_depth_imb_value_300s"), w300),
            _bbo_fs(BboMicroPricePremiumBps(name="bbo_microprice_premium_30s"), w30),
            _bbo_fs(BboMicroPricePremiumBps(name="bbo_microprice_premium_300s"), w300),
            _bbo_fs(TradeBboEdgeBps(name="trade_bbo_edge_30s"), w30),
            _bbo_fs(TradeBboEdgeBps(name="trade_bbo_edge_300s"), w300),
            _bbo_fs(BboMidSmoothDeltaBps(name="bbo_mid_smooth_delta_30s"), w30),
            _bbo_fs(BboMidSmoothDeltaBps(name="bbo_mid_smooth_delta_300s"), w300),
            _bbo_fs(BboMidDeltaBps(name="bbo_mid_delta_bps_30s"), w30),
            _bbo_fs(BboMidDeltaBps(name="bbo_mid_delta_bps_120s"), w120),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_30s"), w30),
            _bbo_fs(BboOrderFlowImbalanceNorm(name="bbo_ofi_norm_30s"), w30),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_120s"), w120),
            _bbo_fs(BboOrderFlowImbalanceNorm(name="bbo_ofi_norm_120s"), w120),
            _bbo_fs(BboOrderFlowImbalance(name="bbo_ofi_300s"), w300),
            _trade_fs(TradeFlowImbalance(name="trade_flow_imbalance_30s"), w30),
            _trade_fs(TradeFlowImbalance(name="trade_flow_imbalance_120s"), w120),
            _trade_fs(TradeFlowToxicity(name="trade_flow_toxicity_30s"), w30),
            _trade_fs(TradeFlowToxicity(name="trade_flow_toxicity_120s"), w120),
            _trade_fs(TradeFlowNotionalLog(name="trade_flow_notional_log_120s"), w120),
            _trade_fs(TradeFlowNotionalLog(name="trade_flow_notional_log_300s"), w300),
            _trade_fs(TradeFlowCountLog(name="trade_flow_count_log_120s"), w120),
            _trade_fs(TradeFlowCountLog(name="trade_flow_count_log_300s"), w300),
            _binance_liq_fs(LiqudationClusterStrength(name="binance_liq_strength_30s"), w30),
            _binance_liq_fs(LiqudationClusterStrength(name="binance_liq_strength_300s"), w300),
            _binance_liq_fs(LiqudationClusterTotalNotional(name="binance_liq_total_notional_300s"), w300),
            _binance_liq_fs(TradeSideLiquidationStrength(name="trade_side_binance_liq_strength_300s"), w300),
            _bybit_liq_fs(LiqudationClusterStrength(name="bybit_liq_strength_30s"), w30),
            _bybit_liq_fs(LiqudationClusterStrength(name="bybit_liq_strength_300s"), w300),
            _bybit_liq_fs(LiqudationClusterTotalNotional(name="bybit_liq_total_notional_300s"), w300),
        ]

    raise ValueError(f"Unsupported horizon: {horizon_sec}")


def _selected_experiments(selection: str) -> list[Experiment]:
    experiments = [Experiment(name="core")]
    if selection == "all":
        return experiments
    return [experiment for experiment in experiments if experiment.name == selection]


def _horizon_index(horizon_sec: int) -> int:
    try:
        return HORIZONS_SEC.index(horizon_sec)
    except ValueError as exc:
        raise ValueError(f"Unsupported horizon: {horizon_sec}") from exc


def _build_regression_target_builder(horizon_sec: int) -> Callable[[dict[str, pl.DataFrame]], np.ndarray]:
    tau_idx = _horizon_index(horizon_sec)

    def build_target(chunk: dict[str, pl.DataFrame]) -> np.ndarray:
        stats = calculate_model_pnl(
            chunk[BINANCE_BTC_TRADES],
            chunk[BINANCE_BTC_BBO],
            horizons_sec=HORIZONS_SEC,
        )
        target = -np.asarray(stats["trade_pnl"][:, tau_idx], dtype=np.float32)
        valid_mask = np.asarray(stats["valid_mask"][:, tau_idx], dtype=bool)
        return np.where(valid_mask, target, np.nan).astype(np.float32, copy=False)

    return build_target


def _build_sample_weights(chunk: dict[str, pl.DataFrame]) -> np.ndarray:
    trades = chunk[BINANCE_BTC_TRADES]
    notionals = (trades["price"] * trades["amount"]).to_numpy().astype(np.float32)
    return np.minimum(notionals, 100_000.0)


def _to_utc_ts_sec(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _format_utc(ts_us: int) -> str:
    dt = datetime.fromtimestamp(ts_us / SECOND, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


class TimeProgressBar:
    def __init__(self, label: str, start_ts_us: int, end_ts_us: int, chunk_sec: int, width: int = 32):
        self.label = label
        self.start_ts_us = start_ts_us
        self.end_ts_us = end_ts_us
        self.chunk_us = chunk_sec * SECOND
        self.width = width
        self.total_us = max(self.end_ts_us - self.start_ts_us, 1)
        self.total_hours = self.total_us / SECOND / 3600.0
        self.total_chunks = max(1, math.ceil(self.total_us / self.chunk_us))

    def _render(self, chunks_done: int) -> None:
        current_ts_us = min(self.start_ts_us + chunks_done * self.chunk_us, self.end_ts_us)
        elapsed_us = max(0, current_ts_us - self.start_ts_us)
        pct = 100.0 * elapsed_us / self.total_us
        done_width = min(self.width, max(0, int(round(self.width * pct / 100.0))))
        bar = "#" * done_width + "-" * (self.width - done_width)
        elapsed_hours = elapsed_us / SECOND / 3600.0
        msg = (
            f"\r{self.label:<18} [{bar}] {pct:6.2f}% | "
            f"{elapsed_hours:7.1f}/{self.total_hours:7.1f}h | "
            f"chunks {min(chunks_done, self.total_chunks):>4}/{self.total_chunks:<4}"
        )
        print(msg, end="", flush=True)

    def start(self) -> None:
        self._render(0)

    def update(self, chunks_done: int) -> None:
        self._render(chunks_done)

    def finish(self) -> None:
        self._render(self.total_chunks)
        print()


def _fit_with_progress(pipeline: RegressionPipeline, chunk_sec: int, label: str) -> RegressionPipeline:
    data_loader = pipeline._data_loader
    progress = TimeProgressBar(
        label=label,
        start_ts_us=data_loader.data_start_ts,
        end_ts_us=data_loader.data_end_ts,
        chunk_sec=chunk_sec,
    )
    progress.start()
    try:
        return pipeline.fit(log_i_chunk=None, on_chunk=progress.update)
    finally:
        progress.finish()


def _weighted_regression_errors(
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


def _collect_validation_predictions(
    pipeline: RegressionPipeline,
    val_loader: ParquetDataLoader,
    tau_idx: int,
    on_chunk: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores_parts: list[np.ndarray] = []
    pnl_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []

    val_loader.reset()
    try:
        for chunk_idx, chunk in enumerate(val_loader, start=1):
            scores = np.asarray(pipeline.predict_chunk(chunk, proba=False), dtype=np.float32).reshape(-1)
            stats = calculate_model_pnl(
                chunk[BINANCE_BTC_TRADES],
                chunk[BINANCE_BTC_BBO],
                horizons_sec=HORIZONS_SEC,
            )
            scores_parts.append(scores)
            pnl_parts.append(np.asarray(stats["trade_pnl"][:, tau_idx], dtype=np.float32))
            valid_parts.append(np.asarray(stats["valid_mask"][:, tau_idx], dtype=bool))
            weights_parts.append(_build_sample_weights(chunk).astype(np.float32, copy=False))
            if on_chunk is not None:
                on_chunk(chunk_idx)
    finally:
        val_loader.reset()

    return (
        np.concatenate(scores_parts),
        np.concatenate(pnl_parts),
        np.concatenate(valid_parts),
        np.concatenate(weights_parts),
    )


def _eligible_validation_rows(
    scores: np.ndarray,
    pnl_tau: np.ndarray,
    valid_tau: np.ndarray,
    weights: np.ndarray,
    horizon_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(valid_tau)
    if eligible.size == 0:
        raise ValueError(f"No valid rows in validation window for horizon {horizon_sec}s")
    return scores[eligible], pnl_tau[eligible], weights[eligible]


def _worst_first_filter_cumulatives(
    eligible_scores: np.ndarray,
    eligible_pnl: np.ndarray,
    eligible_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    worst_first_idx = np.argsort(eligible_scores)[::-1]
    worst_weights = eligible_weights[worst_first_idx]
    worst_weighted_pnl = (eligible_pnl[worst_first_idx] * worst_weights).astype(np.float64)

    cum_filtered_weight = np.concatenate(([0.0], np.cumsum(worst_weights, dtype=np.float64)))
    cum_filtered_weighted_pnl = np.concatenate(([0.0], np.cumsum(worst_weighted_pnl, dtype=np.float64)))

    total_weight = float(eligible_weights.sum(dtype=np.float64))
    total_weighted_pnl = float((eligible_pnl * eligible_weights).sum(dtype=np.float64))
    pnl_all = np.nan if total_weight == 0.0 else total_weighted_pnl / total_weight

    return cum_filtered_weight, cum_filtered_weighted_pnl, total_weight, total_weighted_pnl, pnl_all


def _build_filter_fraction_rows(
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
) -> list[dict]:
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
    return rows


def _evaluate_with_progress(
    pipeline: RegressionPipeline,
    val_loader: ParquetDataLoader,
    chunk_sec: int,
    max_filter_fraction: float,
    experiment_name: str,
    horizon_sec: int,
) -> pl.DataFrame:
    tau_idx = _horizon_index(horizon_sec)

    progress = TimeProgressBar(
        label=f"Val {experiment_name}/{horizon_sec}s",
        start_ts_us=val_loader.data_start_ts,
        end_ts_us=val_loader.data_end_ts,
        chunk_sec=chunk_sec,
    )
    progress.start()
    try:
        scores, pnl_tau, valid_tau, weights = _collect_validation_predictions(
            pipeline,
            val_loader,
            tau_idx,
            on_chunk=progress.update,
        )
    finally:
        progress.finish()

    eligible_scores, eligible_pnl, eligible_weights = _eligible_validation_rows(
        scores, pnl_tau, valid_tau, weights, horizon_sec
    )
    weighted_rmse, weighted_mae = _weighted_regression_errors(eligible_scores, eligible_pnl, eligible_weights)

    cum_filtered_weight, cum_filtered_weighted_pnl, total_weight, total_weighted_pnl, pnl_all = (
        _worst_first_filter_cumulatives(eligible_scores, eligible_pnl, eligible_weights)
    )

    eval_days = max(1.0, (val_loader.data_end_ts - val_loader.data_start_ts) / SECOND / (24 * 60 * 60))
    rows = _build_filter_fraction_rows(
        experiment_name=experiment_name,
        horizon_sec=horizon_sec,
        eligible_size=eligible_scores.size,
        eval_days=eval_days,
        max_filter_fraction=max_filter_fraction,
        cum_filtered_weight=cum_filtered_weight,
        cum_filtered_weighted_pnl=cum_filtered_weighted_pnl,
        total_weight=total_weight,
        total_weighted_pnl=total_weighted_pnl,
        pnl_all=pnl_all,
        weighted_rmse=weighted_rmse,
        weighted_mae=weighted_mae,
    )
    return pl.DataFrame(rows)


def _make_datafiles(data_root: Path) -> list[ParquetDataLoader.Datafile]:
    parquet_paths = [
        str(data_root / "binance_booktickers" / "perp_btcusdt.parquet"),
        str(data_root / "binance_liquidations" / "perp_btcusdt.parquet"),
        str(data_root / "binance_trades" / "perp_btcusdt.parquet"),
        str(data_root / "bybit_liquidations" / "btcusdt.parquet"),
    ]
    return [
        ParquetDataLoader.Datafile(name, path)
        for name, path in zip(PARQUET_NAMES, parquet_paths, strict=True)
    ]


def _make_loader(
    datafiles: list[ParquetDataLoader.Datafile],
    chunk_sec: int,
    since_ts_sec: int,
    until_ts_sec: int,
) -> ParquetDataLoader:
    return ParquetDataLoader(
        datafiles,
        chunk_sec,
        ParquetDataLoader.InputTimeScale.sec,
        since=since_ts_sec,
        until=until_ts_sec,
        lookback_by_name={
            BINANCE_BTC_BBO: MAX_FEATURE_LOOKBACK_SEC,
            BINANCE_BTC_LIQ: MAX_FEATURE_LOOKBACK_SEC,
            BYBIT_BTC_LIQ: MAX_FEATURE_LOOKBACK_SEC,
        },
        lookahead_by_name={BINANCE_BTC_BBO: TARGET_MAX_HORIZON_SEC},
    )


def _build_catboost_regressor(args: argparse.Namespace):
    return build_model(
        "catboost_chunk",
        "regression",
        {
            "loss_function": args.loss_function,
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "random_seed": args.random_seed,
            "allow_writing_files": False,
            "thread_count": args.thread_count,
            "verbose": False,
        },
    )


def _save_feature_importance(
    pipeline: RegressionPipeline,
    feature_specs: list[FeatureSpec],
    output_dir: Path,
    experiment_name: str,
    horizon_sec: int,
) -> Path:
    model = pipeline.model._model
    importances = model.get_feature_importance(type="FeatureImportance")
    fi_df = pl.DataFrame(
        {
            "feature_name": [spec.feature.name for spec in feature_specs],
            "importance": np.asarray(importances, dtype=np.float64),
        }
    ).sort("importance", descending=True)

    fi_path = output_dir / f"{experiment_name}_{horizon_sec}s_feature_importance.csv"
    fi_df.write_csv(fi_path)
    return fi_path


def _save_pipeline_artifacts(
    output_dir: Path,
    experiment_name: str,
    horizon_sec: int,
    feature_specs: list[FeatureSpec],
    args: argparse.Namespace,
    val_result: pl.DataFrame,
    model_path: Path,
    feature_importance_path: Path,
    train_start_ts_us: int,
    train_end_ts_us: int,
    val_start_ts_us: int,
    val_end_ts_us: int,
) -> tuple[Path, Path]:
    val_results_path = output_dir / f"{experiment_name}_{horizon_sec}s_val.csv"
    metadata_path = output_dir / f"{experiment_name}_{horizon_sec}s.json"

    val_result.write_csv(val_results_path)

    best_row = (
        val_result
        .filter(pl.col("kept_turnover_per_day") >= MIN_TURNOVER_PER_DAY)
        .sort("score_bps", descending=True)
        .head(1)
    )

    best_summary = best_row.to_dicts()[0] if best_row.height > 0 else None

    metadata = {
        "experiment": experiment_name,
        "horizon_sec": horizon_sec,
        "model_path": str(model_path),
        "val_results_path": str(val_results_path),
        "feature_importance_path": str(feature_importance_path),
        "train_window": {
            "start": _format_utc(train_start_ts_us),
            "end": _format_utc(train_end_ts_us),
        },
        "validation_window": {
            "start": _format_utc(val_start_ts_us),
            "end": _format_utc(val_end_ts_us),
        },
        "feature_names": [spec.feature.name for spec in feature_specs],
        "n_features": len(feature_specs),
        "model_params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "loss_function": args.loss_function,
            "random_seed": args.random_seed,
            "thread_count": args.thread_count,
        },
        "best_summary": best_summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return val_results_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate liquidation CatBoostRegressor experiments.")
    parser.add_argument(
        "--train-until",
        default="2026-02-01",
        help="UTC date (YYYY-MM-DD) where training ends and validation starts.",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=31,
        help="Training window size in days.",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=14,
        help="Validation window size in days.",
    )
    parser.add_argument(
        "--train-chunk-hours",
        type=int,
        default=1,
        help="Chunk size for training stream in hours.",
    )
    parser.add_argument(
        "--val-chunk-hours",
        type=int,
        default=1,
        help="Chunk size for validation stream in hours.",
    )
    parser.add_argument(
        "--max-filter-fraction",
        type=float,
        default=0.30,
        help="Maximum fraction of worst predicted trades to filter during evaluation.",
    )
    parser.add_argument(
        "--experiment",
        choices=("all", "core"),
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=80,
        help="CatBoost iterations added on every chunk.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="CatBoost tree depth.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="CatBoost learning rate.",
    )
    parser.add_argument(
        "--loss-function",
        default="RMSE",
        help="CatBoost regression loss function.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="CatBoost random seed.",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=1,
        help="CatBoost thread count.",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/regression",
        help="Directory where trained models will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_until_ts_sec = _to_utc_ts_sec(args.train_until)
    train_from_ts_sec = train_until_ts_sec - args.train_days * 24 * 60 * 60
    valid_from_ts_sec = train_until_ts_sec
    valid_until_ts_sec = valid_from_ts_sec + args.val_days * 24 * 60 * 60

    train_chunk_sec = args.train_chunk_hours * 60 * 60
    val_chunk_sec = args.val_chunk_hours * 60 * 60

    data_root = _resolve_data_root()
    datafiles = _make_datafiles(data_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pl.DataFrame] = []

    for experiment in _selected_experiments(args.experiment):
        for horizon_sec in HORIZONS_SEC:
            train_loader = _make_loader(datafiles, train_chunk_sec, train_from_ts_sec, train_until_ts_sec)
            val_loader = _make_loader(datafiles, val_chunk_sec, valid_from_ts_sec, valid_until_ts_sec)
            feature_specs = _build_feature_specs(horizon_sec)

            print(f"\n=== Experiment={experiment.name} horizon={horizon_sec}s ===")
            print(
                "Training window:",
                _format_utc(train_loader.data_start_ts),
                "->",
                _format_utc(train_loader.data_end_ts),
            )
            print(
                "Validation window:",
                _format_utc(val_loader.data_start_ts),
                "->",
                _format_utc(val_loader.data_end_ts),
            )

            pipeline = RegressionPipeline(
                model=_build_catboost_regressor(args),
                feature_specs=feature_specs,
                data_loader=train_loader,
                target_builder=_build_regression_target_builder(horizon_sec),
                sample_weight_builder=_build_sample_weights,
            )

            _fit_with_progress(
                pipeline,
                chunk_sec=train_chunk_sec,
                label=f"Train {experiment.name}/{horizon_sec}s",
            )

            val_result = _evaluate_with_progress(
                pipeline=pipeline,
                val_loader=val_loader,
                chunk_sec=val_chunk_sec,
                max_filter_fraction=args.max_filter_fraction,
                experiment_name=experiment.name,
                horizon_sec=horizon_sec,
            )
            all_results.append(val_result)

            model_path = model_dir / f"{experiment.name}_{horizon_sec}s.cbm"
            pipeline.model._model.save_model(str(model_path))

            feature_importance_path = _save_feature_importance(
                pipeline=pipeline,
                feature_specs=feature_specs,
                output_dir=model_dir,
                experiment_name=experiment.name,
                horizon_sec=horizon_sec,
            )

            val_results_path, metadata_path = _save_pipeline_artifacts(
                output_dir=model_dir,
                experiment_name=experiment.name,
                horizon_sec=horizon_sec,
                feature_specs=feature_specs,
                args=args,
                val_result=val_result,
                model_path=model_path,
                feature_importance_path=feature_importance_path,
                train_start_ts_us=train_loader.data_start_ts,
                train_end_ts_us=train_loader.data_end_ts,
                val_start_ts_us=val_loader.data_start_ts,
                val_end_ts_us=val_loader.data_end_ts,
            )

            print(f"\nSaved model to: {model_path}")
            print(f"Saved validation results to: {val_results_path}")
            print(f"Saved feature importance to: {feature_importance_path}")
            print(f"Saved pipeline metadata to: {metadata_path}")

    if all_results:
        summary = (
            pl.concat(all_results)
            .filter(pl.col("kept_turnover_per_day") >= MIN_TURNOVER_PER_DAY)
            .sort(["horizon_sec", "score_bps"], descending=[False, True])
        )
        summary_path = model_dir / "summary.csv"
        summary.write_csv(summary_path)
        print(f"\nSaved summary to: {summary_path} ({summary.height} rows)")


if __name__ == "__main__":
    main()