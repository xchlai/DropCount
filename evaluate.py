from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from baselines import naive_equal_volume_estimate, volume_aware_mle_estimate
from config import ExperimentConfig, load_config
from models import VolumeAwareSetTransformerRegressor
from simulator import DropletDigitalPCRSimulator, FixedSimulationDataset, collate_samples
from train import build_model
from utils import detect_device, ensure_dir, save_metrics_tables, set_global_seed, to_numpy


@torch.no_grad()
def collect_model_predictions(
    model: VolumeAwareSetTransformerRegressor,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, float | str | int | bool]] = []
    for batch in loader:
        outputs = model(
            batch["volume_fractions"].to(device),
            batch["labels"].to(device),
            batch["false_positive_rate"].to(device),
            batch["mask"].to(device),
        )
        pred_copies = to_numpy(outputs["pred_copies"])
        true_copies = to_numpy(batch["true_total_copies"])
        vf = to_numpy(batch["volume_fractions"])
        labels = to_numpy(batch["labels"])
        mask = to_numpy(batch["mask"])
        for i, meta in enumerate(batch["metadata"]):
            n = int(mask[i].sum())
            f = vf[i, :n]
            y = labels[i, :n]
            naive = naive_equal_volume_estimate(y)
            mle = volume_aware_mle_estimate(f, y)
            rows.append(
                {
                    "true_total_copies": float(true_copies[i]),
                    "pred_dl": float(pred_copies[i]),
                    "pred_naive": float(naive.estimate),
                    "pred_mle": float(mle.estimate),
                    "distribution_name": str(meta["distribution_name"]),
                    "cv": float(meta["cv"]),
                    "n_droplets": int(meta["n_droplets"]),
                    "positive_ratio": float(meta["positive_ratio"]),
                    "false_positive_rate": float(meta["false_positive_rate"]),
                    "is_saturated": bool(meta["positive_ratio"] >= 0.99),
                    "naive_saturation": bool(naive.saturation),
                    "mle_saturation": bool(mle.saturation),
                }
            )
    return pd.DataFrame(rows)


def method_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    y_true = df["true_total_copies"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.maximum(y_true, 1.0)
    rel_bias = (y_pred - y_true) / np.maximum(y_true, 1.0)
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    pearson = pearsonr(y_true, y_pred)[0] if len(df) > 1 else np.nan
    spearman = spearmanr(y_true, y_pred).correlation if len(df) > 1 else np.nan
    return {
        "method": pred_col,
        "count": float(len(df)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "rmsle": float(rmsle),
        "median_relative_error": float(np.median(rel_err)),
        "mean_relative_bias": float(np.mean(rel_bias)),
        "r2": float(r2_score(y_true, y_pred)) if len(df) > 1 else np.nan,
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae_log": float(np.mean(np.abs(np.log1p(y_pred) - np.log1p(y_true)))),
    }


def summarize_metrics(df: pd.DataFrame, copy_bins: List[int]) -> Dict[str, pd.DataFrame]:
    methods = ["pred_dl", "pred_naive", "pred_mle"]
    tables: Dict[str, pd.DataFrame] = {}

    overall = pd.DataFrame([method_metrics(df, m) for m in methods])
    tables["overall_metrics"] = overall

    by_cv_rows = []
    for cv, subset in df.groupby("cv"):
        for m in methods:
            row = method_metrics(subset, m)
            row["cv"] = cv
            by_cv_rows.append(row)
    tables["by_cv_metrics"] = pd.DataFrame(by_cv_rows)

    by_dist_rows = []
    for dist, subset in df.groupby("distribution_name"):
        for m in methods:
            row = method_metrics(subset, m)
            row["distribution_name"] = dist
            by_dist_rows.append(row)
    tables["by_distribution_metrics"] = pd.DataFrame(by_dist_rows)

    bin_labels = pd.cut(df["true_total_copies"], bins=copy_bins, include_lowest=True, right=True)
    df_with_bins = df.assign(copy_bin=bin_labels.astype(str))
    by_bin_rows = []
    for copy_bin, subset in df_with_bins.groupby("copy_bin"):
        for m in methods:
            row = method_metrics(subset, m)
            row["copy_bin"] = copy_bin
            by_bin_rows.append(row)
    tables["by_copy_bin_metrics"] = pd.DataFrame(by_bin_rows)

    sat_rows = []
    for sat_name, subset in {
        "near_saturation": df[df["positive_ratio"] >= 0.95],
        "not_near_saturation": df[df["positive_ratio"] < 0.95],
    }.items():
        if subset.empty:
            continue
        for m in methods:
            row = method_metrics(subset, m)
            row["subset"] = sat_name
            sat_rows.append(row)
    tables["by_saturation_metrics"] = pd.DataFrame(sat_rows)
    return tables


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, col in [("DL", "pred_dl"), ("Naive", "pred_naive"), ("MLE", "pred_mle")]:
        ax.scatter(df["true_total_copies"], df[col], s=10, alpha=0.45, label=label)
    lim = max(df[["true_total_copies", "pred_dl", "pred_naive", "pred_mle"]].to_numpy().max(), 1.0)
    ax.plot([1, lim], [1, lim], "k--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True total copies")
    ax.set_ylabel("Predicted total copies")
    ax.set_title("Predicted vs. true copies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_true_loglog.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, col in [("DL", "pred_dl"), ("Naive", "pred_naive"), ("MLE", "pred_mle")]:
        rel_err = (df[col] - df["true_total_copies"]) / np.maximum(df["true_total_copies"], 1.0)
        ax.scatter(df["true_total_copies"], rel_err, s=10, alpha=0.35, label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("True total copies")
    ax.set_ylabel("Relative error")
    ax.set_title("Relative error vs. true copies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "relative_error_vs_true.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        ((df[col] - df["true_total_copies"]) / np.maximum(df["true_total_copies"], 1.0)).to_numpy()
        for col in ["pred_dl", "pred_naive", "pred_mle"]
    ]
    ax.boxplot(data, labels=["DL", "Naive", "MLE"], showfliers=False)
    ax.set_ylabel("Relative error")
    ax.set_title("Error comparison by method")
    fig.tight_layout()
    fig.savefig(out_dir / "error_boxplot.png", dpi=180)
    plt.close(fig)

    grouped_cv = []
    for cv, subset in df.groupby("cv"):
        for label, col in [("DL", "pred_dl"), ("Naive", "pred_naive"), ("MLE", "pred_mle")]:
            grouped_cv.append(
                {
                    "cv": cv,
                    "method": label,
                    "rmsle": np.sqrt(np.mean((np.log1p(subset[col]) - np.log1p(subset["true_total_copies"])) ** 2)),
                }
            )
    grouped_cv_df = pd.DataFrame(grouped_cv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, subset in grouped_cv_df.groupby("method"):
        ax.plot(subset["cv"], subset["rmsle"], marker="o", label=method)
    ax.set_xlabel("CV")
    ax.set_ylabel("RMSLE")
    ax.set_title("Performance across CV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "performance_by_cv.png", dpi=180)
    plt.close(fig)

    dist_rows = []
    for dist, subset in df.groupby("distribution_name"):
        for label, col in [("DL", "pred_dl"), ("Naive", "pred_naive"), ("MLE", "pred_mle")]:
            dist_rows.append(
                {
                    "distribution_name": dist,
                    "method": label,
                    "mae": np.mean(np.abs(subset[col] - subset["true_total_copies"])),
                }
            )
    dist_df = pd.DataFrame(dist_rows)
    pivot = dist_df.pivot(index="distribution_name", columns="method", values="mae")
    pivot.plot(kind="bar", figsize=(8, 4), rot=30)
    plt.ylabel("MAE")
    plt.title("MAE by distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_distribution.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ddPCR quantification methods")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.simulation.random_seed)
    run_dir = Path(args.run_dir) if args.run_dir else config.run_dir
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    eval_dir = ensure_dir(run_dir / "evaluation")
    device = detect_device(config.training.device)

    simulator = DropletDigitalPCRSimulator(config)
    test_samples = args.test_samples or (
        config.simulation.test_samples_per_combo
        * len(config.simulation.distributions.eval_names)
        * len(config.simulation.distributions.cv_values)
        * len(config.simulation.test_n_droplets)
    )
    test_dataset = FixedSimulationDataset(
        simulator=simulator,
        num_samples=test_samples,
        distributions=config.simulation.distributions.eval_names,
        cv_values=config.simulation.distributions.cv_values,
        n_droplets_choices=config.simulation.test_n_droplets,
        seed_offset=2025,
    )
    loader = DataLoader(test_dataset, batch_size=config.training.batch_size, collate_fn=collate_samples)

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    results_df = collect_model_predictions(model, loader, device)
    results_df.to_csv(eval_dir / "predictions.csv", index=False)
    tables = summarize_metrics(results_df, config.simulation.copy_bins)
    save_metrics_tables(tables, eval_dir)
    make_plots(results_df, eval_dir)
    print(tables["overall_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
