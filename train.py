from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from inference import evaluate
from liquidation_task_tools.training import FeatureSpec, RegressionPipeline, build_model
from liquidation_task_tools.training.feature_importance import save_feature_importance
from liquidation_task_tools.training.experiment import (
    HORIZONS_SEC,
    MIN_TURNOVER_PER_DAY,
    TimeProgressBar,
    build_feature_specs,
    build_regression_target_builder,
    build_sample_weights,
    format_utc,
    make_datafiles,
    make_loader,
    resolve_data_root,
    to_utc_ts_sec,
)

TRAIN_UNTIL = "2026-02-01"
TRAIN_DAYS = 62
VAL_DAYS = 28
TRAIN_CHUNK_HOURS = 1
VAL_CHUNK_HOURS = 1
MAX_FILTER_FRACTION = 0.30
MODEL_DIR = Path("artifacts/regression")
EXPERIMENT = "core"

ITERATIONS = 300
DEPTH = 5
LEARNING_RATE = 0.01
LOSS_FUNCTION = "RMSE"
RANDOM_SEED = 42
THREAD_COUNT = 3

MODEL_PARAMS = {
    "loss_function": LOSS_FUNCTION,
    "iterations": ITERATIONS,
    "depth": DEPTH,
    "learning_rate": LEARNING_RATE,
    "random_seed": RANDOM_SEED,
    "allow_writing_files": False,
    "thread_count": THREAD_COUNT,
    "verbose": False,
}


def _fit_with_progress(pipeline: RegressionPipeline, chunk_sec: int, label: str) -> RegressionPipeline:
    data_loader = pipeline._data_loader
    progress = TimeProgressBar(
        label=label,
        start_ts_us=data_loader.data_start_ts,
        end_ts_us=data_loader.data_end_ts,
        chunk_sec=chunk_sec,
    )
    progress.start()
    try:
        return pipeline.fit(log_i_chunk=None, on_chunk=progress.update)
    finally:
        progress.finish()


def _save_pipeline_artifacts(
    output_dir: Path,
    experiment_name: str,
    horizon_sec: int,
    feature_specs: list[FeatureSpec],
    val_result: pl.DataFrame,
    model_path: Path,
    feature_importance_path: Path,
    feature_importance_plot_path: Path,
    train_start_ts_us: int,
    train_end_ts_us: int,
    val_start_ts_us: int,
    val_end_ts_us: int,
) -> tuple[Path, Path]:
    val_results_path = output_dir / f"{experiment_name}_{horizon_sec}s_val.csv"
    metadata_path = output_dir / f"{experiment_name}_{horizon_sec}s.json"

    val_result.write_csv(val_results_path)

    best_row = (
        val_result
        .filter(pl.col("kept_turnover_per_day") >= MIN_TURNOVER_PER_DAY)
        .sort("score_bps", descending=True)
        .head(1)
    )
    best_summary = best_row.to_dicts()[0] if best_row.height > 0 else None

    metadata = {
        "experiment": experiment_name,
        "horizon_sec": horizon_sec,
        "model_path": str(model_path),
        "val_results_path": str(val_results_path),
        "feature_importance_path": str(feature_importance_path),
        "feature_importance_plot_path": str(feature_importance_plot_path),
        "train_window": {
            "start": format_utc(train_start_ts_us),
            "end": format_utc(train_end_ts_us),
        },
        "validation_window": {
            "start": format_utc(val_start_ts_us),
            "end": format_utc(val_end_ts_us),
        },
        "feature_names": [spec.feature.name for spec in feature_specs],
        "n_features": len(feature_specs),
        "model_params": MODEL_PARAMS,
        "best_summary": best_summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return val_results_path, metadata_path


def main() -> None:
    train_until_ts_sec = to_utc_ts_sec(TRAIN_UNTIL)
    train_from_ts_sec = train_until_ts_sec - TRAIN_DAYS * 24 * 60 * 60
    valid_from_ts_sec = train_until_ts_sec
    valid_until_ts_sec = valid_from_ts_sec + VAL_DAYS * 24 * 60 * 60

    train_chunk_sec = TRAIN_CHUNK_HOURS * 60 * 60
    val_chunk_sec = VAL_CHUNK_HOURS * 60 * 60

    datafiles = make_datafiles(resolve_data_root())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[pl.DataFrame] = []

    for horizon_sec in HORIZONS_SEC:
        train_loader = make_loader(datafiles, train_chunk_sec, train_from_ts_sec, train_until_ts_sec)
        val_loader = make_loader(datafiles, val_chunk_sec, valid_from_ts_sec, valid_until_ts_sec)
        feature_specs = build_feature_specs(horizon_sec)

        print(f"\n=== Experiment={EXPERIMENT} horizon={horizon_sec}s ===")
        print("Training window:", format_utc(train_loader.data_start_ts), "->", format_utc(train_loader.data_end_ts))
        print("Validation window:", format_utc(val_loader.data_start_ts), "->", format_utc(val_loader.data_end_ts))

        pipeline = RegressionPipeline(
            model=build_model("catboost_chunk", "regression", MODEL_PARAMS),
            feature_specs=feature_specs,
            data_loader=train_loader,
            target_builder=build_regression_target_builder(horizon_sec),
            sample_weight_builder=build_sample_weights,
        )

        _fit_with_progress(pipeline, chunk_sec=train_chunk_sec, label=f"Train {EXPERIMENT}/{horizon_sec}s")

        eval_pipeline = RegressionPipeline(
            model=pipeline.model,
            feature_specs=feature_specs,
            data_loader=val_loader,
            target_builder=build_regression_target_builder(horizon_sec),
            sample_weight_builder=build_sample_weights,
        )
        val_result = evaluate(
            eval_pipeline, val_loader,
            chunk_sec=val_chunk_sec,
            horizon_sec=horizon_sec,
            experiment_name=EXPERIMENT,
            max_filter_fraction=MAX_FILTER_FRACTION,
            progress_label=f"Val {EXPERIMENT}/{horizon_sec}s",
        )
        all_results.append(val_result)

        model_path = MODEL_DIR / f"{EXPERIMENT}_{horizon_sec}s.cbm"
        pipeline.model._model.save_model(str(model_path))

        feature_importance_path, feature_importance_plot_path = save_feature_importance(
            pipeline, feature_specs, MODEL_DIR, EXPERIMENT, horizon_sec
        )
        val_results_path, metadata_path = _save_pipeline_artifacts(
            MODEL_DIR, EXPERIMENT, horizon_sec, feature_specs, val_result,
            model_path, feature_importance_path, feature_importance_plot_path,
            train_loader.data_start_ts, train_loader.data_end_ts,
            val_loader.data_start_ts, val_loader.data_end_ts,
        )

        print(f"\nSaved model to: {model_path}")
        print(f"Saved validation results to: {val_results_path}")
        print(f"Saved feature importance to: {feature_importance_path}")
        print(f"Saved feature importance plot to: {feature_importance_plot_path}")
        print(f"Saved pipeline metadata to: {metadata_path}")

    if all_results:
        summary = (
            pl.concat(all_results)
            .filter(pl.col("kept_turnover_per_day") >= MIN_TURNOVER_PER_DAY)
            .sort(["horizon_sec", "score_bps"], descending=[False, True])
        )
        summary_path = MODEL_DIR / "summary.csv"
        summary.write_csv(summary_path)
        print(f"\nSaved summary to: {summary_path} ({summary.height} rows)")


if __name__ == "__main__":
    main()
