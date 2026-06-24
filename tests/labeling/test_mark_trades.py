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
            "timestamp": np.asarray([t * SECOND for t in timestamps], dtype=np.int64),
            "ticker": ["perp:btcusdt"] * len(timestamps),
            "side": sides,
            "price": np.full(len(timestamps), 100.0, dtype=np.float64),
            "amount": np.asarray(amounts, dtype=np.float64),
        }
    )


def make_bbo(timestamps, bids, asks):
    return pl.DataFrame(
        {
            "timestamp": np.asarray([t * SECOND for t in timestamps], dtype=np.int64),
            "ticker": ["perp:btcusdt"] * len(timestamps),
            "bid_price": np.asarray(bids, dtype=np.float32),
            "bid_amount": np.ones(len(timestamps), dtype=np.float64),
            "ask_price": np.asarray(asks, dtype=np.float32),
            "ask_amount": np.ones(len(timestamps), dtype=np.float64),
        }
    )


class TestCalculateModelPnl(unittest.TestCase):
#   Формула: pnl_i(τ) = -s_i * (mid(t_i + τ) - p_i) / p_i * 10_000 + 0.5 bps rebate
#   s_i = +1 если taker buy (мы maker sell), s_i = -1 если taker sell (мы maker buy)
#   w_i = min(price * amount, 100_000)

    DEFAULT_TRADES = make_trades(
        [0, 30, 60, 120, 300],
        ['buy', 'buy', 'sell', 'sell', 'buy'],
        [1, 2, 1, 1, 1],
    )

    DEFAULT_BBO = make_bbo(
        timestamps=[30, 60, 90, 150, 210, 300, 420, 600],
        bids=[9, 14, 24, 99, 99, 200, 85, 145],
        asks=[11, 16, 26, 101, 101, 200, 115, 155],
    )

    def test_average_case(self):
        trade_pnl_expected = np.array(
            [
                [9000.5, 7500.5, -9999.5],
                [8500.5,    0.5, -9999.5],
                [-7499.5,   0.5, 10000.5],
                [0.5,       0.5, 0.5],
                [-9999.5,   0.5, -4999.5],
            ],
            dtype=np.float64,
        )

        valid_mask_expected = np.ones_like(trade_pnl_expected, dtype=bool)

        pnl_all_expected = np.array(
            [1417.16666667, 1250.5, -4166.16666667],
            dtype=np.float64,
        )
        stats = calculate_model_pnl(TestCalculateModelPnl.DEFAULT_TRADES, TestCalculateModelPnl.DEFAULT_BBO)

        np.testing.assert_allclose(
            stats["trade_pnl"],
            trade_pnl_expected
        )
        np.testing.assert_array_equal(
            stats["valid_mask"],
            valid_mask_expected,
        )
        np.testing.assert_allclose(
            stats["pnl_all"],
            pnl_all_expected
        )
        np.testing.assert_allclose(
            stats["pnl_kept"],
            pnl_all_expected,
        )

    def test_with_invalid_horizont(self):
        trades = make_trades(
            [0, 1, 50, 200, 300],
            ['buy', 'buy', 'sell', 'sell', 'buy'],
            [1, 2, 1, 1, 1],
        )
        bbo = make_bbo(
            timestamps=[30, 60, 90, 150, 210, 310],
            bids=[9, 14, 24, 99, 99, 200],
            asks=[11, 16, 26, 101, 101, 200],
        )

        stats = calculate_model_pnl(trades, bbo)

        trade_pnl_expected = np.array(
            [
                [ 9000.5,  7500.5,     0.5   ],  
                [ 9000.5,  7500.5,     0.5   ],   
                [-8499.5,     0.5,     np.nan],   
                [    0.5,     np.nan,  np.nan],   
                [    np.nan,  np.nan,  np.nan],   
            ],
            dtype=np.float64,
        )

        valid_expected = np.array(
            [
                [ True,  True,  True],
                [ True,  True,  True],
                [ True,  True, False],
                [ True, False, False],
                [False, False, False],
            ],
            dtype=bool,
        )

        pnl_all_expected = np.array(
            [3700.5, 5625.5, 0.5],
            dtype=np.float64,
        )

        actual_pnl = stats["trade_pnl"]
        np.testing.assert_array_equal(
            np.isnan(actual_pnl),
            np.isnan(trade_pnl_expected),
        )
        np.testing.assert_allclose(
            np.nan_to_num(actual_pnl, nan=0.0),
            np.nan_to_num(trade_pnl_expected, nan=0.0),
            rtol=0,
            atol=1e-9,
        )

        np.testing.assert_array_equal(
            stats["valid_mask"],
            valid_expected,
        )

        np.testing.assert_allclose(
            stats["pnl_all"],
            pnl_all_expected,
            rtol=0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            stats["pnl_kept"],
            pnl_all_expected,
            rtol=0,
            atol=1e-9,
        )
    
        self.assertTrue(np.all(np.isnan(stats["pnl_filtered"])))


    def test_non_empty_mask(self):
        trades = make_trades(
            [0, 30, 120, 300],
            ['buy', 'buy', 'sell', 'sell'],
            [1, 2, 1, 1],
        )
        bbo = make_bbo(
            timestamps=[0, 30, 120, 300],
            bids=[9, 14, 24, 99],
            asks=[11, 16, 26, 101],
        )
        filter_mask = np.asanyarray([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])

        stats = calculate_model_pnl(trades, bbo, filter_mask)
        print(stats)


if __name__ == "__main__":
    unittest.main()
