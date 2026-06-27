from __future__ import annotations

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

from .model_pipeline import FeatureSpec

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


def resolve_data_root() -> Path:
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


def to_utc_ts_sec(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def format_utc(ts_us: int) -> str:
    dt = datetime.fromtimestamp(ts_us / SECOND, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def horizon_index(horizon_sec: int) -> int:
    try:
        return HORIZONS_SEC.index(horizon_sec)
    except ValueError as exc:
        raise ValueError(f"Unsupported horizon: {horizon_sec}") from exc


def selected_experiments(selection: str) -> list[Experiment]:
    experiments = [Experiment(name="core")]
    if selection == "all":
        return experiments
    return [experiment for experiment in experiments if experiment.name == selection]


def build_sample_weights(chunk: dict[str, pl.DataFrame]) -> np.ndarray:
    trades = chunk[BINANCE_BTC_TRADES]
    notionals = (trades["price"] * trades["amount"]).to_numpy().astype(np.float32)
    return np.minimum(notionals, 100_000.0)


def build_regression_target_builder(horizon_sec: int) -> Callable[[dict[str, pl.DataFrame]], np.ndarray]:
    tau_idx = horizon_index(horizon_sec)

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


def make_datafiles(data_root: Path) -> list[ParquetDataLoader.Datafile]:
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


def make_loader(
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


def _w(sec: int) -> int:
    return sec * SECOND


HORIZON_WINDOWS: dict[int, dict[str, object]] = {
    30: {
        "short": 15,
        "long": 30,
        "delta": (5, 15),
        "ofi": ((5, 5), (15, 15), 30),
        "flow": (5, 15),
        "amount": (15, 30),
    },
    120: {
        "short": 30,
        "long": 120,
        "delta": (15, 60),
        "ofi": ((15, 15), (60, 60), 120),
        "flow": (15, 60),
        "amount": (60, 120),
    },
    300: {
        "short": 30,
        "long": 300,
        "delta": (30, 120),
        "ofi": ((30, 30), (120, 120), 300),
        "flow": (30, 120),
        "amount": (120, 300),
    },
}


def build_feature_specs(horizon_sec: int) -> list[FeatureSpec]:
    cfg = HORIZON_WINDOWS[horizon_sec]
    ws, wl = _w(cfg["short"]), _w(cfg["long"])
    wd0, wd1 = (_w(d) for d in cfg["delta"])
    (of0, of0n), (of1, of1n), of2 = cfg["ofi"]
    wf0, wf1 = (_w(f) for f in cfg["flow"])
    wa0, wa1 = (_w(a) for a in cfg["amount"])
    wof0, wof0n, wof1, wof1n, wof2 = _w(of0), _w(of0n), _w(of1), _w(of1n), _w(of2)
    h = horizon_sec

    return [
        _trade_fs(TradeSide()),
        _trade_fs(TradeSignedNotionalLog()),
        _bbo_fs(BboSpreadBps(), _w(30)),
        _bbo_fs(BboTopDepthLog(), _w(30)),
        _bbo_fs(BboVolumeImbalance(name=f"bbo_vol_imb_{cfg['short']}s"), ws),
        _bbo_fs(BboVolumeImbalance(name=f"bbo_vol_imb_{h}s"), wl),
        _bbo_fs(BboDepthImbalanceValue(name=f"bbo_depth_imb_value_{cfg['short']}s"), ws),
        _bbo_fs(BboDepthImbalanceValue(name=f"bbo_depth_imb_value_{h}s"), wl),
        _bbo_fs(BboMicroPricePremiumBps(name=f"bbo_microprice_premium_{cfg['short']}s"), ws),
        _bbo_fs(BboMicroPricePremiumBps(name=f"bbo_microprice_premium_{h}s"), wl),
        _bbo_fs(TradeBboEdgeBps(name=f"trade_bbo_edge_{cfg['short']}s"), ws),
        _bbo_fs(TradeBboEdgeBps(name=f"trade_bbo_edge_{h}s"), wl),
        _bbo_fs(BboMidSmoothDeltaBps(name=f"bbo_mid_smooth_delta_{cfg['short']}s"), ws),
        _bbo_fs(BboMidSmoothDeltaBps(name=f"bbo_mid_smooth_delta_{h}s"), wl),
        _bbo_fs(BboMidDeltaBps(name=f"bbo_mid_delta_bps_{cfg['delta'][0]}s"), wd0),
        _bbo_fs(BboMidDeltaBps(name=f"bbo_mid_delta_bps_{cfg['delta'][1]}s"), wd1),
        _bbo_fs(BboOrderFlowImbalance(name=f"bbo_ofi_{of0}s"), wof0),
        _bbo_fs(BboOrderFlowImbalanceNorm(name=f"bbo_ofi_norm_{of0n}s"), wof0n),
        _bbo_fs(BboOrderFlowImbalance(name=f"bbo_ofi_{of1}s"), wof1),
        _bbo_fs(BboOrderFlowImbalanceNorm(name=f"bbo_ofi_norm_{of1n}s"), wof1n),
        _bbo_fs(BboOrderFlowImbalance(name=f"bbo_ofi_{of2}s"), wof2),
        _trade_fs(TradeFlowImbalance(name=f"trade_flow_imbalance_{cfg['flow'][0]}s"), wf0),
        _trade_fs(TradeFlowImbalance(name=f"trade_flow_imbalance_{cfg['flow'][1]}s"), wf1),
        _trade_fs(TradeFlowToxicity(name=f"trade_flow_toxicity_{cfg['flow'][0]}s"), wf0),
        _trade_fs(TradeFlowToxicity(name=f"trade_flow_toxicity_{cfg['flow'][1]}s"), wf1),
        _trade_fs(TradeFlowNotionalLog(name=f"trade_flow_notional_log_{cfg['amount'][0]}s"), wa0),
        _trade_fs(TradeFlowNotionalLog(name=f"trade_flow_notional_log_{cfg['amount'][1]}s"), wa1),
        _trade_fs(TradeFlowCountLog(name=f"trade_flow_count_log_{cfg['amount'][0]}s"), wa0),
        _trade_fs(TradeFlowCountLog(name=f"trade_flow_count_log_{cfg['amount'][1]}s"), wa1),
        _binance_liq_fs(LiqudationClusterStrength(name=f"binance_liq_strength_{cfg['short']}s"), ws),
        _binance_liq_fs(LiqudationClusterStrength(name=f"binance_liq_strength_{h}s"), wl),
        _binance_liq_fs(LiqudationClusterTotalNotional(name=f"binance_liq_total_notional_{h}s"), wl),
        _binance_liq_fs(TradeSideLiquidationStrength(name=f"trade_side_binance_liq_strength_{h}s"), wl),
        _bybit_liq_fs(LiqudationClusterStrength(name=f"bybit_liq_strength_{cfg['short']}s"), ws),
        _bybit_liq_fs(LiqudationClusterStrength(name=f"bybit_liq_strength_{h}s"), wl),
        _bybit_liq_fs(LiqudationClusterTotalNotional(name=f"bybit_liq_total_notional_{h}s"), wl),
    ]
