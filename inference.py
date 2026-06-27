from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from liquidation_task_tools.loaders import ParquetDataLoader
from liquidation_task_tools.training.experiment import (
    BINANCE_BTC_TRADES,
    MIN_TURNOVER_PER_DAY,
    TimeProgressBar,
    build_feature_specs,
    build_sample_weights,
    format_utc,
    horizon_index,
    make_datafiles,
    make_loader,
    resolve_data_root,
    to_utc_ts_sec,
)
from liquidation_task_tools.training.metrics import evaluate_predictions
from liquidation_task_tools.training.model_pipeline import RegressionPipeline
from liquidation_task_tools.training.models import CatBoostChunkRegressor
from liquidation_task_tools.training.postprocess import collect_predictions

MODEL_DIR = Path("artifacts/regression")
SINCE = "2026-02-01"
UNTIL = "2026-03-01"
CHUNK_HOURS = 1
MAX_FILTER_FRACTION = 0.30


def load_model(cbm_path: Path) -> CatBoostChunkRegressor:
    from catboost import CatBoostRegressor

    wrapper = CatBoostChunkRegressor(verbose=False, allow_writing_files=False)
    wrapper._model = CatBoostRegressor()
    wrapper._model.load_model(str(cbm_path))
    wrapper._fitted = True
    return wrapper


def build_pipeline(
    model: CatBoostChunkRegressor,
    feature_specs,
    loader: ParquetDataLoader,
) -> RegressionPipeline:
    return RegressionPipeline(
        model=model,
        feature_specs=feature_specs,
        data_loader=loader,
        target_builder=lambda chunk: np.zeros(len(chunk[BINANCE_BTC_TRADES]), dtype=np.float32),
        sample_weight_builder=build_sample_weights,
    )


def evaluate(
    pipeline: RegressionPipeline,
    loader: ParquetDataLoader,
    *,
    chunk_sec: int,
    horizon_sec: int,
    experiment_name: str,
    max_filter_fraction: float,
    progress_label: str | None = None,
) -> pl.DataFrame:
    tau_idx = horizon_index(horizon_sec)
    label = progress_label or f"Eval {experiment_name}/{horizon_sec}s"
    progress = TimeProgressBar(
        label=label,
        start_ts_us=loader.data_start_ts,
        end_ts_us=loader.data_end_ts,
        chunk_sec=chunk_sec,
    )
    progress.start()
    try:
        scores, pnl, valid_mask, weights = collect_predictions(
            pipeline, loader, tau_idx, on_chunk=progress.update
        )
    finally:
        progress.finish()

    return evaluate_predictions(
        scores, pnl, valid_mask, weights, loader,
        experiment_name=experiment_name,
        horizon_sec=horizon_sec,
        max_filter_fraction=max_filter_fraction,
    )


def parse_model_stem(stem: str) -> tuple[str, int]:
    experiment, horizon_part = stem.rsplit("_", 1)
    return experiment, int(horizon_part.removesuffix("s"))


def evaluate_model_file(
    cbm_path: Path,
    datafiles,
    *,
    since_ts_sec: int,
    until_ts_sec: int,
    chunk_sec: int,
    max_filter_fraction: float,
    output_dir: Path | None = None,
) -> pl.DataFrame:
    experiment_name, horizon_sec = parse_model_stem(cbm_path.stem)
    loader = make_loader(datafiles, chunk_sec, since_ts_sec, until_ts_sec)
    feature_specs = build_feature_specs(horizon_sec)
    pipeline = build_pipeline(load_model(cbm_path), feature_specs, loader)

    print(f"\n=== Inference {experiment_name} horizon={horizon_sec}s ===")
    print("Window:", format_utc(loader.data_start_ts), "->", format_utc(loader.data_end_ts))

    result = evaluate(
        pipeline, loader,
        chunk_sec=chunk_sec,
        horizon_sec=horizon_sec,
        experiment_name=experiment_name,
        max_filter_fraction=max_filter_fraction,
    )

    if output_dir is not None:
        out_path = output_dir / f"{cbm_path.stem}_inference.csv"
        result.write_csv(out_path)
        print(f"Saved results to: {out_path}")

    best = (
        result
        .filter(pl.col("kept_turnover_per_day") >= MIN_TURNOVER_PER_DAY)
        .sort("score_bps", descending=True)
        .head(1)
    )
    if best.height > 0:
        print("Best row:", best.to_dicts()[0])

    return result


def main() -> None:
    since_ts_sec = to_utc_ts_sec(SINCE)
    until_ts_sec = to_utc_ts_sec(UNTIL)
    chunk_sec = CHUNK_HOURS * 60 * 60

    datafiles = make_datafiles(resolve_data_root())
    cbm_paths = sorted(MODEL_DIR.glob("*.cbm"))
    if not cbm_paths:
        raise FileNotFoundError(f"No .cbm models found in {MODEL_DIR}")

    all_results: list[pl.DataFrame] = []
    for cbm_path in cbm_paths:
        all_results.append(
            evaluate_model_file(
                cbm_path, datafiles,
                since_ts_sec=since_ts_sec,
                until_ts_sec=until_ts_sec,
                chunk_sec=chunk_sec,
                max_filter_fraction=MAX_FILTER_FRACTION,
                output_dir=MODEL_DIR,
            )
        )

    if all_results:
        summary = (
            pl.concat(all_results)
            .filter(pl.col("kept_turnover_per_day") >= MIN_TURNOVER_PER_DAY)
            .sort(["horizon_sec", "score_bps"], descending=[False, True])
        )
        summary_path = MODEL_DIR / "inference_summary.csv"
        summary.write_csv(summary_path)
        print(f"\nSaved summary to: {summary_path} ({summary.height} rows)")


if __name__ == "__main__":
    main()
