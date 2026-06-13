from typing import Dict, Tuple
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


class BboFeature(Feature):
    def __init__(self, name: str):
        super().__init__(name)

    def _prepare_bbo_window(self, **data) -> Dict[str, np.ndarray]:
        bbo = data["bbo"]
        trades = data["trades"]
        window = data["window_sec"]

        bbo_ts = bbo["timestamp"].to_numpy()
        trades_ts = trades["timestamp"].to_numpy()

        bid_price = bbo["bid_price"].to_numpy().astype(np.float64)
        ask_price = bbo["ask_price"].to_numpy().astype(np.float64)
        bid_amount = bbo["bid_amount"].to_numpy().astype(np.float64)
        ask_amount = bbo["ask_amount"].to_numpy().astype(np.float64)
        mid_price = (bid_price + ask_price) / 2.0

        left = np.searchsorted(bbo_ts, trades_ts - window, side="left")
        right = np.searchsorted(bbo_ts, trades_ts, side="right")
        snap_idx = right - 1
        snap_valid = snap_idx >= 0

        return {
            "bbo_ts": bbo_ts,
            "trades_ts": trades_ts,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_amount": bid_amount,
            "ask_amount": ask_amount,
            "mid_price": mid_price,
            "left": left,
            "right": right,
            "snap_idx": snap_idx,
            "snap_valid": snap_valid,
        }

        return {
            "bbo_ts": bbo_ts,
            "trades_ts": trades_ts,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_amount": bid_amount,
            "ask_amount": ask_amount,
            "mid_price": mid_price,
            "left": left,
            "right": right,
            "snap_idx": snap_idx,
            "snap_valid": snap_valid,
        }

    def calculate_max_used_ts(self, **data) -> np.ndarray:
        trades_ts = data["trades_ts"]
        bbo_ts = data["bbo_ts"]
        most_recent = data["most_recent"]
        least_recent = data.get("least_recent", most_recent)
        mask = data.get("mask", most_recent > least_recent)

        max_used_ts = trades_ts.copy()
        max_used_ts[mask] = bbo_ts[most_recent[mask] - 1]
        return max_used_ts


class BboSpreadBps(BboFeature):
    def __init__(self):
        super().__init__("bbo_spread_bps")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)

        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        valid = ctx["snap_valid"]
        if np.any(valid):
            idx = ctx["snap_idx"][valid]
            spread = ctx["ask_price"][idx] - ctx["bid_price"][idx]
            mid = ctx["mid_price"][idx]
            spread_bps = np.divide(spread, mid, out=np.zeros_like(spread, dtype=np.float64), where=mid != 0) * 10_000.0
            feature[valid] = spread_bps

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["snap_idx"] + 1,
            least_recent=ctx["snap_idx"],
            mask=valid,
        )
        return feature, ctx["trades_ts"], max_used_ts


class BboVolumeImbalance(BboFeature):
    def __init__(self):
        super().__init__("bbo_volume_imbalance")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)

        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        valid = ctx["snap_valid"]
        if np.any(valid):
            idx = ctx["snap_idx"][valid]
            bid_amt = ctx["bid_amount"][idx]
            ask_amt = ctx["ask_amount"][idx]
            denom = bid_amt + ask_amt
            imbalance = np.divide(bid_amt - ask_amt, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
            feature[valid] = imbalance

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["snap_idx"] + 1,
            least_recent=ctx["snap_idx"],
            mask=valid,
        )
        return feature, ctx["trades_ts"], max_used_ts


class BboMidSmoothDeltaBps(BboFeature):
    def __init__(self):
        super().__init__("bbo_mid_smooth_delta_bps")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)

        mid_cumm = np.r_[0.0, np.cumsum(ctx["mid_price"], dtype=np.float64)]
        counts = ctx["right"] - ctx["left"]
        mid_sum = mid_cumm[ctx["right"]] - mid_cumm[ctx["left"]]
        mid_smooth = np.divide(mid_sum, counts, out=np.zeros_like(mid_sum, dtype=np.float64), where=counts > 0)

        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        valid = ctx["snap_valid"] & (counts > 0)
        if np.any(valid):
            idx = ctx["snap_idx"][valid]
            mid_now = ctx["mid_price"][idx]
            mid_ref = mid_smooth[valid]
            delta_bps = np.divide(mid_now - mid_ref, mid_ref, out=np.zeros_like(mid_now, dtype=np.float64), where=mid_ref != 0) * 10_000.0
            feature[valid] = delta_bps

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["right"],
            least_recent=ctx["left"],
        )
        return feature, ctx["trades_ts"], max_used_ts


class BboTopDepthLog(BboFeature):
    def __init__(self):
        super().__init__("bbo_top_depth_log")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)

        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        valid = ctx["snap_valid"]
        if np.any(valid):
            idx = ctx["snap_idx"][valid]
            depth = (
                ctx["bid_price"][idx] * ctx["bid_amount"][idx]
                + ctx["ask_price"][idx] * ctx["ask_amount"][idx]
            )
            feature[valid] = np.log1p(depth)

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["snap_idx"] + 1,
            least_recent=ctx["snap_idx"],
            mask=valid,
        )
        return feature, ctx["trades_ts"], max_used_ts


class BboMidDeltaBps(BboFeature):
    def __init__(self, name: str = "bbo_mid_delta_bps"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)

        past_idx = ctx["left"] - 1
        valid = ctx["snap_valid"] & (past_idx >= 0)
        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        if np.any(valid):
            mid_now = ctx["mid_price"][ctx["snap_idx"][valid]]
            mid_past = ctx["mid_price"][past_idx[valid]]
            delta_bps = np.divide(
                mid_now - mid_past,
                mid_past,
                out=np.zeros_like(mid_now, dtype=np.float64),
                where=mid_past != 0,
            ) * 10_000.0
            feature[valid] = delta_bps

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["snap_idx"] + 1,
            least_recent=ctx["snap_idx"],
            mask=ctx["snap_valid"],
        )
        return feature, ctx["trades_ts"], max_used_ts


class BboMicroPricePremiumBps(BboFeature):
    def __init__(self):
        super().__init__("bbo_micro_price_premium_bps")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)

        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        valid = ctx["snap_valid"]
        if np.any(valid):
            idx = ctx["snap_idx"][valid]
            bid = ctx["bid_price"][idx]
            ask = ctx["ask_price"][idx]
            bid_amt = ctx["bid_amount"][idx]
            ask_amt = ctx["ask_amount"][idx]
            denom = bid_amt + ask_amt
            micro_price = np.divide(
                ask * bid_amt + bid * ask_amt,
                denom,
                out=(bid + ask) / 2.0,
                where=denom > 0,
            )
            mid = ctx["mid_price"][idx]
            feature[valid] = np.divide(
                micro_price - mid,
                mid,
                out=np.zeros_like(mid, dtype=np.float64),
                where=mid != 0,
            ) * 10_000.0

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["snap_idx"] + 1,
            least_recent=ctx["snap_idx"],
            mask=valid,
        )
        return feature, ctx["trades_ts"], max_used_ts


class BboVolumeImbalanceAbs(BboFeature):
    def __init__(self):
        super().__init__("bbo_volume_imbalance_abs")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = BboVolumeImbalance().calculate(**data)
        return np.abs(base).astype(np.float32, copy=False), trades_ts, max_used_ts


class TradeBboEdgeBps(BboFeature):
    def __init__(self):
        super().__init__("trade_bbo_edge_bps")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self._prepare_bbo_window(**data)
        trades = data["trades"]
        trade_px = trades["price"].to_numpy().astype(np.float64)
        sign = _trade_side_sign(trades)

        feature = np.zeros_like(ctx["trades_ts"], dtype=np.float64)
        valid = ctx["snap_valid"]
        if np.any(valid):
            idx = ctx["snap_idx"][valid]
            mid_now = ctx["mid_price"][idx]
            feature[valid] = -sign[valid] * (mid_now - trade_px[valid]) / trade_px[valid] * 10_000.0

        feature = feature.astype(np.float32).reshape((-1, 1))
        max_used_ts = self.calculate_max_used_ts(
            trades_ts=ctx["trades_ts"],
            bbo_ts=ctx["bbo_ts"],
            most_recent=ctx["snap_idx"] + 1,
            least_recent=ctx["snap_idx"],
            mask=valid,
        )
        return feature, ctx["trades_ts"], max_used_ts


class TradeSideBboVolumeImbalance(BboFeature):
    def __init__(self):
        super().__init__("trade_side_bbo_volume_imbalance")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = BboVolumeImbalance().calculate(**data)
        sign = _trade_side_sign(data["trades"]).astype(np.float32)
        return (base[:, 0] * sign).reshape((-1, 1)), trades_ts, max_used_ts


class TradeSideBboMicroPricePremiumBps(BboFeature):
    def __init__(self):
        super().__init__("trade_side_bbo_micro_price_premium_bps")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = BboMicroPricePremiumBps().calculate(**data)
        sign = _trade_side_sign(data["trades"]).astype(np.float32)
        return (base[:, 0] * sign).reshape((-1, 1)), trades_ts, max_used_ts


class TradeSideBboMidDeltaBps(BboFeature):
    def __init__(self, name: str = "trade_side_bbo_mid_delta_bps"):
        super().__init__(name)

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = BboMidDeltaBps(self.name.replace("trade_side_", "")).calculate(**data)
        sign = _trade_side_sign(data["trades"]).astype(np.float32)
        return (base[:, 0] * sign).reshape((-1, 1)), trades_ts, max_used_ts


class TradeSideBboMidSmoothDeltaBps(BboFeature):
    def __init__(self):
        super().__init__("trade_side_bbo_mid_smooth_delta_bps")

    def calculate(self, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, trades_ts, max_used_ts = BboMidSmoothDeltaBps().calculate(**data)
        sign = _trade_side_sign(data["trades"]).astype(np.float32)
        return (base[:, 0] * sign).reshape((-1, 1)), trades_ts, max_used_ts

