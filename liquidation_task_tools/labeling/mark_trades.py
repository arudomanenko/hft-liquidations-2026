import numpy as np

from liquidation_task_tools.constants import SECOND


def calculate_model_pnl(
    trades,
    bbo,
    filter_mask=None,
    horizons_sec=(30, 120, 300),
    rebate_bps: float = 0.5,
):
    trade_ts = trades["timestamp"].to_numpy()
    price = trades["price"].to_numpy().astype(np.float64)
    side = trades["side"].to_numpy()
    sign = (
        np.where(side == "buy", 1.0, -1.0)
        if side.dtype.kind in "OUS"
        else np.sign(side.astype(np.float64))
    )

    bbo_ts = bbo["timestamp"].to_numpy()
    bid = bbo["bid_price"].to_numpy().astype(np.float64)
    ask = bbo["ask_price"].to_numpy().astype(np.float64)
    mid = (bid + ask) / 2.0

    pnl = np.full((len(trades), len(horizons_sec)), np.nan, dtype=np.float64)
    valid = np.zeros_like(pnl, dtype=bool)

    for col, tau_sec in enumerate(horizons_sec):
        target_ts = trade_ts + tau_sec * SECOND
        idx = np.searchsorted(bbo_ts, target_ts, side="right") - 1
        valid[:, col] = (idx >= 0) & (target_ts <= bbo_ts[-1])

        exit_mid = mid[idx[valid[:, col]]]
        current_pnl = (
            -sign[valid[:, col]]
            * (exit_mid - price[valid[:, col]])
            / price[valid[:, col]]
            * 10_000.0
            + rebate_bps
        )
        pnl[valid[:, col], col] = current_pnl

    if filter_mask is None:
        filter_mask = np.zeros_like(pnl, dtype=bool)
    filter_mask = np.asarray(filter_mask).astype(bool, copy=False)
    
    weights = np.minimum((trades["price"] * trades["amount"]).to_numpy().astype(np.float64), 100_000.0)[:, None]

    def avg(mask):
        active = valid & mask
        active_weights = np.where(active, weights, 0.0)
        denom = active_weights.sum(axis=0)
        return np.divide(
            (active_weights * np.where(active, pnl, 0.0)).sum(axis=0),
            denom,
            out=np.full(len(horizons_sec), np.nan, dtype=np.float64),
            where=denom > 0,
        )

    pnl_all = avg(np.ones_like(filter_mask, dtype=bool))
    pnl_kept = avg(~filter_mask)
    pnl_filtered = avg(filter_mask)

    return {
        "trade_pnl": pnl,
        "valid_mask": valid,
        "pnl_all": pnl_all,
        "pnl_kept": pnl_kept,
        "pnl_filtered": pnl_filtered,
        "score": pnl_kept - pnl_all,
    }


def mark_trades(
    trades,
    bbo,
    horizons_sec=(30, 120, 300),
    rebate_bps: float = 0.5,
):
    stats = calculate_model_pnl(
        trades,
        bbo,
        horizons_sec=horizons_sec,
        rebate_bps=rebate_bps,
    )
    labels = np.zeros_like(stats["trade_pnl"], dtype=np.int8)
    labels[stats["valid_mask"]] = (
        stats["trade_pnl"][stats["valid_mask"]] <= 0
    ).astype(np.int8)

    return labels

