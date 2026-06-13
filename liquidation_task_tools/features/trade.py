from typing import Tuple

import numpy as np

from liquidation_task_tools.base import Feature


class TradeFeature(Feature):
    def __init__(self, name: str):
        super().__init__(name)

    def _trade_context(self, **data) -> Tuple[object, np.ndarray]:
        trades = data["trades"]
        trades_ts = trades["timestamp"].to_numpy()
        return trades, trades_ts

    def calculate_max_used_ts(self, **data) -> np.ndarray:
        return data["trades_ts"].copy()


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


def _rolling_trade_flow(trades, window_us: int):
    trades_ts = trades["timestamp"].to_numpy()
    price = trades["price"].to_numpy().astype(np.float64)
    amount = trades["amount"].to_numpy().astype(np.float64)
    sign = _trade_side_sign(trades)
    notional = price * amount

    buy_notional = np.where(sign > 0, notional, 0.0)
    sell_notional = np.where(sign < 0, notional, 0.0)
    buy_cumm = np.r_[0.0, np.cumsum(buy_notional, dtype=np.float64)]
    sell_cumm = np.r_[0.0, np.cumsum(sell_notional, dtype=np.float64)]

    left = np.searchsorted(trades_ts, trades_ts - window_us, side="left")
    right = np.arange(1, len(trades_ts) + 1)
    buy_window = buy_cumm[right] - buy_cumm[left]
    sell_window = sell_cumm[right] - sell_cumm[left]
    count = right - left
    return trades_ts, sign, buy_window, sell_window, count


class TradeSide(TradeFeature):
    def __init__(self):
        super().__init__("trade_side")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades, trades_ts = self._trade_context(**data)
        values = _trade_side_sign(trades)

        feature = values.astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeNotionalLog(TradeFeature):
    def __init__(self):
        super().__init__("trade_notional_log")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades, trades_ts = self._trade_context(**data)
        price = trades["price"].to_numpy().astype(np.float64)
        amount = trades["amount"].to_numpy().astype(np.float64)
        values = np.log1p(price * amount)
        feature = values.astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeSignedNotionalLog(TradeFeature):
    def __init__(self):
        super().__init__("trade_signed_notional_log")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades, trades_ts = self._trade_context(**data)
        side_feature, _, _ = TradeSide().calculate(trades=trades)
        price = trades["price"].to_numpy().astype(np.float64)
        amount = trades["amount"].to_numpy().astype(np.float64)
        values = side_feature[:, 0].astype(np.float64) * np.log1p(price * amount)
        feature = values.astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeFlowImbalance(TradeFeature):
    def __init__(self, name: str = "trade_flow_imbalance"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades = data["trades"]
        window_us = int(data["window_sec"])
        trades_ts, _, buy_window, sell_window, _ = _rolling_trade_flow(trades, window_us)

        denom = buy_window + sell_window
        imbalance = np.divide(
            buy_window - sell_window,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
        feature = imbalance.astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeSideFlowImbalance(TradeFeature):
    def __init__(self, name: str = "trade_side_flow_imbalance"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades = data["trades"]
        window_us = int(data["window_sec"])
        trades_ts, sign, buy_window, sell_window, _ = _rolling_trade_flow(trades, window_us)

        denom = buy_window + sell_window
        imbalance = np.divide(
            buy_window - sell_window,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
        feature = (sign * imbalance).astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeFlowToxicity(TradeFeature):
    def __init__(self, name: str = "trade_flow_toxicity"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades = data["trades"]
        window_us = int(data["window_sec"])
        trades_ts, _, buy_window, sell_window, _ = _rolling_trade_flow(trades, window_us)

        denom = buy_window + sell_window
        toxicity = np.divide(
            np.abs(buy_window - sell_window),
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
        feature = toxicity.astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeFlowNotionalLog(TradeFeature):
    def __init__(self, name: str = "trade_flow_notional_log"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades = data["trades"]
        window_us = int(data["window_sec"])
        trades_ts, _, buy_window, sell_window, _ = _rolling_trade_flow(trades, window_us)

        feature = np.log1p(buy_window + sell_window).astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)


class TradeFlowCountLog(TradeFeature):
    def __init__(self, name: str = "trade_flow_count_log"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trades = data["trades"]
        window_us = int(data["window_sec"])
        trades_ts, _, _, _, count = _rolling_trade_flow(trades, window_us)

        feature = np.log1p(count.astype(np.float64)).astype(np.float32).reshape((-1, 1))
        return feature, trades_ts, self.calculate_max_used_ts(trades_ts=trades_ts)
