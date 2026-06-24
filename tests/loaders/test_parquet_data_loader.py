import tempfile
import unittest
from pathlib import Path

import polars as pl

from liquidation_task_tools.constants import SECOND
from liquidation_task_tools.loaders import ParquetDataLoader


def write_parquet(path: Path, timestamps) -> None:
    pl.DataFrame({"timestamp": timestamps}).write_parquet(path)


class TestParquetDataLoader(unittest.TestCase):
    def test_window_to_us(self):
        self.assertEqual(
            ParquetDataLoader._window_to_us(1, ParquetDataLoader.InputTimeScale.sec),
            SECOND,
        )

    def test_iterates_over_time_windows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trades.parquet"
            write_parquet(path, [0, SECOND, 2 * SECOND, 3 * SECOND])

            loader = ParquetDataLoader(
                datafiles=[ParquetDataLoader.Datafile(name="trades", path=str(path))],
                window=1,
                timescale=ParquetDataLoader.InputTimeScale.sec,
            )

            chunks = list(loader)

            self.assertEqual(len(chunks), 4)
            self.assertIn("trades", chunks[0])
            self.assertGreaterEqual(chunks[0]["trades"].height, 1)


if __name__ == "__main__":
    unittest.main()
