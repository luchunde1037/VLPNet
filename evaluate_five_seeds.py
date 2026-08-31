"""Evaluate five independently trained VLPNet checkpoints."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluate import DEFAULT_WINDOWS, choose_device, evaluate_checkpoint, resolve_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent

# PyCharm configuration.
DATA_ROOT = PROJECT_ROOT / "data_full"
CHECKPOINT_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "results" / "five_seeds"
SEEDS = (10, 20, 30, 40, 50)
BATCH_SIZE = 32
DEVICE = "auto"  # "auto", "cuda", or "cpu"


def summarize(metrics):
    metric_names = ("R2", "correlation", "MAE", "RMSE", "bias")
    rows = []
    for window_sec, group in metrics.groupby("window_sec"):
        row = {
            "window_sec": window_sec,
            "n_seeds": group["seed"].nunique(),
            "n_records": int(group["n"].iloc[0]),
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("window_sec").reset_index(drop=True)


def plot_summary(summary, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
    settings = (("R2", r"$R^2$"), ("MAE", r"MAE of $\log_{10}A$"), ("RMSE", r"RMSE of $\log_{10}A$"))
    x = summary["window_sec"].to_numpy()
    for axis, (metric, label) in zip(axes, settings):
        mean = summary[f"{metric}_mean"].to_numpy()
        standard_deviation = summary[f"{metric}_std"].to_numpy()
        axis.plot(x, mean, marker="o", linewidth=1.8)
        axis.fill_between(x, mean - standard_deviation, mean + standard_deviation, alpha=0.2)
        axis.set_xlabel("P-wave observation duration (s)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    device = choose_device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for seed in SEEDS:
        checkpoint = resolve_checkpoint(CHECKPOINT_DIR, seed)
        print(f"Seed {seed}: {checkpoint.name}")
        metrics = evaluate_checkpoint(
            checkpoint, DATA_ROOT, device, BATCH_SIZE, windows=DEFAULT_WINDOWS
        )
        metrics.insert(0, "seed", seed)
        all_metrics.append(metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize(metrics)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_seed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_mean_std.csv", index=False)
    plot_summary(summary, OUTPUT_DIR / "performance_mean_std.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
