import unittest

import numpy as np
import polars as pl

from liquidation_task_tools.constants import SECOND
from liquidation_task_tools.features import TradeSide


def make_trades(timestamps, sides, price=100.0, amount=1.0):
    return pl.DataFrame(
        {
            "timestamp": np.asarray(timestamps, dtype=np.int64),
            "side": sides,
            "price": np.full(len(timestamps), price, dtype=np.float64),
            "amount": np.full(len(timestamps), amount, dtype=np.float64),
        }
    )


class TestTradeFeatures(unittest.TestCase):
    def test_trade_side(self):
        trades = make_trades([SECOND, 2 * SECOND], ["buy", "sell"])
        feature, feature_ts, max_used_ts = TradeSide().calculate(trades=trades)

        self.assertEqual(feature.shape, (2, 1))
        np.testing.assert_array_equal(feature_ts, trades["timestamp"].to_numpy())
        np.testing.assert_array_equal(feature[:, 0], np.array([1.0, -1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
