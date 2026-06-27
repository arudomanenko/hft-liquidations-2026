from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .model_pipeline import FeatureSpec, RegressionPipeline


def plot_feature_importance(csv_path: Path) -> Path:
    df = pl.read_csv(csv_path)
    if "feature_name" not in df.columns or "importance" not in df.columns:
        raise ValueError(
            f"{csv_path}: expected columns 'feature_name' and 'importance', got {df.columns}"
        )

    df = df.sort("importance")
    title = csv_path.stem.removesuffix("_feature_importance").replace("_", " ")

    height = max(4.0, 0.28 * df.height + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(df["feature_name"].to_list(), df["importance"].to_list(), color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance — {title}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    output_path = csv_path.with_name(f"{csv_path.stem}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_feature_importance(
    pipeline: RegressionPipeline,
    feature_specs: list[FeatureSpec],
    output_dir: Path,
    experiment_name: str,
    horizon_sec: int,
) -> tuple[Path, Path]:
    model = pipeline.model._model
    importances = model.get_feature_importance(type="FeatureImportance")
    fi_df = pl.DataFrame(
        {
            "feature_name": [spec.feature.name for spec in feature_specs],
            "importance": np.asarray(importances, dtype=np.float64),
        }
    ).sort("importance", descending=True)

    csv_path = output_dir / f"{experiment_name}_{horizon_sec}s_feature_importance.csv"
    fi_df.write_csv(csv_path)
    plot_path = plot_feature_importance(csv_path)
    return csv_path, plot_path
