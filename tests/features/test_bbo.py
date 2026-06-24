import unittest

import numpy as np
import polars as pl

from liquidation_task_tools.constants import SECOND
from liquidation_task_tools.features import BboSpreadBps


def make_trades(timestamps, sides, price=100.0):
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "side": sides,
            "price": np.full(len(timestamps), price, dtype=np.float64),
            "amount": np.ones(len(timestamps), dtype=np.float64),
        }
    )


def make_bbo(timestamps, bid=99.0, ask=101.0):
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "bid_price": np.full(len(timestamps), bid, dtype=np.float64),
            "bid_amount": np.ones(len(timestamps), dtype=np.float64),
            "ask_price": np.full(len(timestamps), ask, dtype=np.float64),
            "ask_amount": np.ones(len(timestamps), dtype=np.float64),
        }
    )


class TestBboFeatures(unittest.TestCase):
    def test_bbo_spread_bps(self):
        trades = make_trades([2 * SECOND], ["buy"])
        bbo = make_bbo([SECOND])
        window_sec = SECOND

        feature, feature_ts, max_used_ts = BboSpreadBps().calculate(
            trades=trades,
            bbo=bbo,
            window_sec=window_sec,
        )

        self.assertEqual(feature.shape, (1, 1))
        np.testing.assert_array_equal(feature_ts, trades["timestamp"].to_numpy())
        self.assertGreater(feature[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
