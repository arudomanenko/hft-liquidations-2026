import unittest

import numpy as np
import polars as pl

from liquidation_task_tools.constants import SECOND
from liquidation_task_tools.labeling import calculate_model_pnl, mark_trades


def make_trades(timestamps, sides, amounts=None):
    if amounts is None:
        amounts = np.ones(len(timestamps), dtype=np.float64)
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "ticker": ["perp:btcusdt"] * len(timestamps),
            "side": sides,
            "price": np.full(len(timestamps), 100.0, dtype=np.float64),
            "amount": np.asarray(amounts, dtype=np.float64),
        }
    )


def make_bbo(timestamps, mids):
    mids = np.asarray(mids, dtype=np.float64)
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "ticker": ["perp:btcusdt"] * len(timestamps),
            "bid_price": mids - 0.5,
            "bid_amount": np.ones(len(timestamps), dtype=np.float64),
            "ask_price": mids + 0.5,
            "ask_amount": np.ones(len(timestamps), dtype=np.float64),
        }
    )


class TestCalculateModelPnl(unittest.TestCase):
    def test_forward_fills_bbo_mid_at_horizon(self):
        trades = make_trades(
            timestamps=[31 * SECOND],
            sides=["sell"],
        )
        bbo = make_bbo(
            timestamps=[30 * SECOND, 60 * SECOND, 90 * SECOND],
            mids=[99.0, 101.0, 102.0],
        )

        stats = calculate_model_pnl(trades, bbo, horizons_sec=(30,))

        np.testing.assert_array_equal(stats["valid_mask"], np.array([[True]]))
        np.testing.assert_allclose(stats["trade_pnl"][:, 0], np.array([100.5]))

    def test_reports_weighted_model_pnl_for_kept_and_filtered_trades(self):
        trades = make_trades(
            timestamps=[0, 30 * SECOND, 60 * SECOND],
            sides=["buy", "buy", "sell"],
            amounts=[1.0, 2.0, 1.0],
        )
        bbo = make_bbo(
            timestamps=[30 * SECOND, 60 * SECOND, 90 * SECOND],
            mids=[99.0, 101.0, 101.0],
        )
        filter_mask = np.array([[0], [1], [0]], dtype=np.int8)

        stats = calculate_model_pnl(
            trades,
            bbo,
            filter_mask,
            horizons_sec=(30,),
        )

        np.testing.assert_allclose(stats["trade_pnl"][:, 0], np.array([100.5, -99.5, 100.5]))
        np.testing.assert_allclose(stats["pnl_all"], np.array([0.5]))
        np.testing.assert_allclose(stats["pnl_kept"], np.array([100.5]))
        np.testing.assert_allclose(stats["pnl_filtered"], np.array([-99.5]))
        np.testing.assert_allclose(stats["score"], np.array([100.0]))

    def test_excludes_trades_without_future_bbo_from_pnl_metrics(self):
        trades = make_trades(
            timestamps=[0, 100 * SECOND],
            sides=["buy", "buy"],
        )
        bbo = make_bbo(
            timestamps=[30 * SECOND],
            mids=[99.0],
        )

        stats = calculate_model_pnl(trades, bbo, horizons_sec=(30,))

        np.testing.assert_array_equal(stats["valid_mask"], np.array([[True], [False]]))
        np.testing.assert_allclose(stats["pnl_all"], np.array([100.5]))
        self.assertTrue(np.isnan(stats["trade_pnl"][1, 0]))


class TestMarkTrades(unittest.TestCase):
    def test_returns_filter_labels_for_non_positive_pnl(self):
        trades = make_trades(
            timestamps=[0],
            sides=["buy"],
        )
        bbo = make_bbo(
            timestamps=[30 * SECOND, 120 * SECOND, 300 * SECOND],
            mids=[101.0, 101.0, 99.0],
        )

        labels = mark_trades(
            trades,
            bbo,
        )

        np.testing.assert_array_equal(
            labels,
            np.array([[1, 1, 0]], dtype=np.int8),
        )


if __name__ == "__main__":
    unittest.main()
