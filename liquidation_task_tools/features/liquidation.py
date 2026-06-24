from typing import Tuple
import numpy as np

from liquidation_task_tools.base import Feature


def _trade_side_sign(trades) -> np.ndarray:
    side = trades["side"].to_numpy()
    if side.dtype.kind in "OUS":
        if not np.isin(side, ["buy", "sell"]).all():
            raise ValueError("trades['side'] must contain only buy/sell")
        return np.where(side == "buy", 1.0, -1.0)
    sign = np.sign(side.astype(np.float64))
    if (sign == 0).any():
        raise ValueError("trades['side'] must contain only +/-1")
    return sign


class LiquidationFeature(Feature):
    def __init__(self, name: str):
        super().__init__(name)

    def _prepare_window_notionals(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        liquidations = data["liquidations"]
        trades = data["trades"]
        window = data["window_sec"]
        timestamp_shift_us = int(data.get("timestamp_shift_us", 0))

        liq_ts = liquidations["timestamp"].to_numpy().astype(np.int64)
        if timestamp_shift_us != 0:
            liq_ts = liq_ts + timestamp_shift_us

        trades_ts = trades["timestamp"].to_numpy()
        notionals = (liquidations["price"] * liquidations["amount"]).to_numpy().astype(np.float64)

        buy_cumm = np.r_[0.0, np.cumsum(np.where(liquidations["side"] == "buy", notionals, 0.0), dtype=np.float64)]
        sell_cumm = np.r_[0.0, np.cumsum(np.where(liquidations["side"] == "sell", notionals, 0.0), dtype=np.float64)]

        left = np.searchsorted(liq_ts, trades_ts - window, side="left")
        right = np.searchsorted(liq_ts, trades_ts, side="right")

        buy_notional = buy_cumm[right] - buy_cumm[left]
        sell_notional = sell_cumm[right] - sell_cumm[left]
        return buy_notional, sell_notional, trades_ts, liq_ts, right, left

    def calculate_max_used_ts(self, **data) -> np.ndarray:
        trades_ts = data["trades_ts"]
        liq_ts = data["liq_ts"]
        most_recent = data["most_recent"]
        least_recent = data["least_recent"]

        max_used_ts = trades_ts.copy()
        mask = most_recent > least_recent
        max_used_ts[mask] = liq_ts[most_recent[mask] - 1]
        return max_used_ts


class LiqudationClusterImbalance(LiquidationFeature):
    def __init__(self, name: str = "liquidation_notional_imbalance"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        buy_notional, sell_notional, trades_ts, liq_ts, right, left = self._prepare_window_notionals(**data)

        denom = buy_notional + sell_notional
        imbalance = np.divide(
            buy_notional - sell_notional,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
        feature = imbalance.astype(np.float32).reshape((-1, 1))

        max_used_ts = self.calculate_max_used_ts(
            trades_ts=trades_ts,
            liq_ts=liq_ts,
            most_recent=right,
            least_recent=left,
        )
        return feature, trades_ts, max_used_ts


class LiqudationClusterStrength(LiquidationFeature):
    def __init__(self, name: str = "liquidation_notional_strength"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        buy_notional, sell_notional, trades_ts, liq_ts, right, left = self._prepare_window_notionals(**data)

        delta = buy_notional - sell_notional
        signed_log_delta = np.sign(delta) * np.log1p(np.abs(delta))
        feature = signed_log_delta.astype(np.float32).reshape((-1, 1))

        max_used_ts = self.calculate_max_used_ts(
            trades_ts=trades_ts,
            liq_ts=liq_ts,
            most_recent=right,
            least_recent=left,
        )
        return feature, trades_ts, max_used_ts


class LiqudationClusterTotalNotional(LiquidationFeature):
    def __init__(self, name: str = "liquidation_total_notional_log"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        buy_notional, sell_notional, trades_ts, liq_ts, right, left = self._prepare_window_notionals(**data)

        total_notional = buy_notional + sell_notional
        feature = np.log1p(total_notional).astype(np.float32).reshape((-1, 1))

        max_used_ts = self.calculate_max_used_ts(
            trades_ts=trades_ts,
            liq_ts=liq_ts,
            most_recent=right,
            least_recent=left,
        )
        return feature, trades_ts, max_used_ts


class LiqudationClusterCount(LiquidationFeature):
    def __init__(self, name: str = "liquidation_count_log"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, trades_ts, liq_ts, right, left = self._prepare_window_notionals(**data)

        count = right - left
        feature = np.log1p(count.astype(np.float64)).astype(np.float32).reshape((-1, 1))

        max_used_ts = self.calculate_max_used_ts(
            trades_ts=trades_ts,
            liq_ts=liq_ts,
            most_recent=right,
            least_recent=left,
        )
        return feature, trades_ts, max_used_ts


class TradeSideLiquidationImbalance(LiquidationFeature):
    def __init__(self, name: str = "trade_side_liquidation_imbalance"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = LiqudationClusterImbalance().calculate(**data)
        sign = _trade_side_sign(data["trades"]).astype(np.float32)
        return (base[:, 0] * sign).reshape((-1, 1)), trades_ts, max_used_ts


class TradeSideLiquidationStrength(LiquidationFeature):
    def __init__(self, name: str = "trade_side_liquidation_strength"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = LiqudationClusterStrength().calculate(**data)
        sign = _trade_side_sign(data["trades"]).astype(np.float32)
        return (base[:, 0] * sign).reshape((-1, 1)), trades_ts, max_used_ts


class LiqudationMeanNotionalPerEventLog(LiquidationFeature):
    def __init__(self, name: str = "liquidation_mean_notional_per_event_log"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        buy_notional, sell_notional, trades_ts, liq_ts, right, left = self._prepare_window_notionals(**data)
        total_notional = buy_notional + sell_notional
        count = right - left
        mean_notional = np.divide(
            total_notional,
            count,
            out=np.zeros_like(total_notional, dtype=np.float64),
            where=count > 0,
        )
        feature = np.log1p(mean_notional).astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=trades_ts,
            liq_ts=liq_ts,
            most_recent=right,
            least_recent=left,
        )
        return feature, trades_ts, max_used_ts