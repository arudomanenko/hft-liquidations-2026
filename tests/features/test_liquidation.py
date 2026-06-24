import unittest

import numpy as np
import polars as pl

from liquidation_task_tools.constants import SECOND
from liquidation_task_tools.features import LiqudationClusterImbalance


def make_trades(timestamps, sides):
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "side": sides,
            "price": np.full(len(timestamps), 100.0, dtype=np.float64),
            "amount": np.ones(len(timestamps), dtype=np.float64),
        }
    )


def make_liquidations(timestamps, sides, price=100.0, amount=1.0):
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "side": sides,
            "price": np.full(len(timestamps), price, dtype=np.float64),
            "amount": np.full(len(timestamps), amount, dtype=np.float64),
        }
    )


class TestLiquidationFeatures(unittest.TestCase):
    def test_liquidation_cluster_imbalance(self):
        trades = make_trades([2 * SECOND], ["buy"])
        liquidations = make_liquidations([SECOND], ["buy"], amount=2.0)
        window_sec = SECOND

        feature, feature_ts, max_used_ts = LiqudationClusterImbalance().calculate(
            trades=trades,
            liquidations=liquidations,
            window_sec=window_sec,
        )

        self.assertEqual(feature.shape, (1, 1))
        np.testing.assert_array_equal(feature_ts, trades["timestamp"].to_numpy())
        self.assertEqual(feature[0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
