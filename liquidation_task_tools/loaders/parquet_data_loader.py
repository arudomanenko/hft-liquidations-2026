from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import polars as pl

from liquidation_task_tools.constants import MICROSECOND, MILLISECOND, SECOND, TWO_WEEKS_SEC


class ParquetDataLoader:
    @dataclass
    class Datafile:
        name: str
        path: str

    class InputTimeScale(Enum):
        us = 1
        ms = 2
        sec = 3

    _TIMESCALE_TO_US = {
        InputTimeScale.us.value: MICROSECOND,
        InputTimeScale.ms.value: MILLISECOND,
        InputTimeScale.sec.value: SECOND,
    }

    @staticmethod
    def _normalize_timescale(timescale) -> int:
        if isinstance(timescale, Enum):
            return int(timescale.value)
        return int(timescale)

    @staticmethod
    def _window_to_us(window: int, timescale: InputTimeScale) -> int:
        timescale_key = ParquetDataLoader._normalize_timescale(timescale)
        if timescale_key not in ParquetDataLoader._TIMESCALE_TO_US:
            raise ValueError(f"Unsupported timescale: {timescale}")
        return int(window) * ParquetDataLoader._TIMESCALE_TO_US[timescale_key]

    def _load_part(self, path: str, ts_from: int, ts_to: int) -> pl.DataFrame:
        return (
            self._lf_by_path[path]
            .filter((pl.col("timestamp") >= ts_from) & (pl.col("timestamp") < ts_to))
            .collect()
        )

    @staticmethod
    def _first_ts(lf: pl.LazyFrame) -> int:
        return int(lf.select(pl.col("timestamp").min()).collect()[0, 0])

    @staticmethod
    def _last_ts(lf: pl.LazyFrame) -> int:
        return int(lf.select(pl.col("timestamp").max()).collect()[0, 0])

    def __init__(
        self,
        datafiles: List[Datafile],
        window: int,
        timescale: InputTimeScale,
        since: Optional[int] = None,
        until: Optional[int] = None,
        lookback_by_name: Optional[Dict[str, int]] = None,
        lookahead_by_name: Optional[Dict[str, int]] = None,
    ) -> None:
        self._datafiles = datafiles
        self._time_window_us = self._window_to_us(window, timescale)
        self._lookback_us_by_name = {
            name: self._window_to_us(lookback, timescale)
            for name, lookback in (lookback_by_name or {}).items()
        }
        self._lookahead_us_by_name = {
            name: self._window_to_us(lookahead, timescale)
            for name, lookahead in (lookahead_by_name or {}).items()
        }

        self._lf_by_path = {d.path: pl.scan_parquet(d.path) for d in datafiles}
        self._first_ts_by_name = {d.name: self._first_ts(self._lf_by_path[d.path]) for d in datafiles}
        self._last_ts_by_name = {d.name: self._last_ts(self._lf_by_path[d.path]) for d in datafiles}

        file_start_ts = min(self._first_ts_by_name.values())
        if since is None:
            self._data_start_ts = file_start_ts
        else:
            self._data_start_ts = max(file_start_ts, self._window_to_us(since, timescale))
        file_end_ts = max(self._last_ts_by_name.values()) + 1
        if until is None:
            self._data_end_ts = file_end_ts
        else:
            until_us = self._window_to_us(until, timescale)
            self._data_end_ts = min(file_end_ts, until_us)

        self._ts_from = self._data_start_ts

    def get_particular_time(self, from_ts: int, to_ts: int) -> Dict[str, pl.DataFrame]:
        return {d.name: self._load_part(d.path, from_ts, to_ts) for d in self._datafiles}

    def get_first_two_weeks(self) -> Dict[str, pl.DataFrame]:
        from_ts = self._data_start_ts
        to_ts = from_ts + TWO_WEEKS_SEC * SECOND
        return self.get_particular_time(from_ts, to_ts)

    def get_first_day(self) -> Dict[str, pl.DataFrame]:
        from_ts = self._data_start_ts
        to_ts = from_ts + 24 * 60 * 60 * SECOND
        return self.get_particular_time(from_ts, to_ts)

    def get_first_hour(self) -> Dict[str, pl.DataFrame]:
        from_ts = self._data_start_ts
        to_ts = from_ts + 60 * 60 * SECOND
        return self.get_particular_time(from_ts, to_ts)

    @property
    def data_start_ts(self) -> int:
        return self._data_start_ts

    @property
    def data_end_ts(self) -> int:
        return self._data_end_ts

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, pl.DataFrame]:
        if self._ts_from >= self._data_end_ts:
            raise StopIteration

        ts_to = min(self._ts_from + self._time_window_us, self._data_end_ts)
        chunk = {}
        for d in self._datafiles:
            lookback_us = self._lookback_us_by_name.get(d.name, 0)
            lookahead_us = self._lookahead_us_by_name.get(d.name, 0)
            load_from = max(self._ts_from - lookback_us, self._first_ts_by_name[d.name])
            load_to = min(ts_to + lookahead_us, self._data_end_ts)
            chunk[d.name] = self._load_part(d.path, load_from, load_to)
        self._ts_from = ts_to
        return chunk

    def reset(self):
        self._ts_from = self._data_start_ts
