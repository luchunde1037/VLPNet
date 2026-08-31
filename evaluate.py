"""Evaluate a trained VLPNet checkpoint across observation durations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import SAMPLE_RATE, VariableLengthPWaveDataset
from models import VLPNet
from utils import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parent

# PyCharm configuration: change SEED only when evaluating another model.
DATA_ROOT = PROJECT_ROOT / "data_full"
MODEL_DIR = PROJECT_ROOT / "model"
SEED = 50
OUTPUT_DIR = PROJECT_ROOT / "results" / f"evaluation_seed{SEED}"
BATCH_SIZE = 32
DEVICE = "auto"  # "auto", "cuda", or "cpu"

DEFAULT_WINDOWS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)


def choose_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def resolve_checkpoint(model_dir, seed):
    """Find a new or legacy checkpoint for the requested seed."""
    preferred = (
        model_dir / f"VLPNet_seed{seed}_best.chkpt",
        model_dir / f"VLPNet_seed{seed}_best.pt",
    )
    for path in preferred:
        if path.is_file():
            return path

    matches = sorted(model_dir.glob(f"*seed{seed}*best*.chkpt"))
    if not matches:
        matches = sorted(model_dir.glob(f"*seed{seed}*.chkpt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(f"  - {path.name}" for path in matches)
        raise RuntimeError(
            f"Multiple checkpoints were found for seed {seed}:\n{names}\n"
            "Keep only the intended checkpoint or rename it to "
            f"VLPNet_seed{seed}_best.chkpt."
        )

    if seed == 50:
        fallback = sorted(model_dir.glob("*.chkpt"))
        if len(fallback) == 1:
            return fallback[0]

    raise FileNotFoundError(
        f"No checkpoint was found for seed {seed} in {model_dir}. "
        f"Expected VLPNet_seed{seed}_best.chkpt or a legacy filename "
        f"containing 'seed{seed}'."
    )


def convert_legacy_state_dict(state_dict):
    """Convert parameter keys from the earlier implementation to VLPNet keys."""
    prefix_map = {
        "conv1.": "feature_extractor.0.",
        "bn1.": "feature_extractor.1.",
        "conv2.": "feature_extractor.3.",
        "bn2.": "feature_extractor.4.",
        "lstm.": "sequence_encoder.",
        "fc.": "regression_head.",
    }
    converted = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        new_key = clean_key
        for old_prefix, new_prefix in prefix_map.items():
            if clean_key.startswith(old_prefix):
                new_key = new_prefix + clean_key[len(old_prefix):]
                break
        converted[new_key] = value
    return converted


def load_model(checkpoint_path, device):
    model = VLPNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(convert_legacy_state_dict(state_dict))
    model.eval()
    return model


def predict_at_window(model, dataset, window_sec, device, batch_size):
    crop_samples = int(round(window_sec * SAMPLE_RATE))
    indices = np.flatnonzero(dataset.valid_lengths >= crop_samples)
    predictions, observations = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            waveforms = dataset.waveforms[batch_indices].copy()
            waveforms[:, crop_samples:] = 0.0
            waveform_tensor = torch.from_numpy(waveforms).unsqueeze(-1).float().to(device)
            lengths = torch.full((len(batch_indices),), crop_samples, dtype=torch.long, device=device)
            predictions.append(model(waveform_tensor, lengths).cpu().numpy())
            observations.append(dataset.targets[batch_indices])
    return np.concatenate(predictions), np.concatenate(observations)


def evaluate_checkpoint(checkpoint_path, data_root, device, batch_size, windows=DEFAULT_WINDOWS):
    dataset = VariableLengthPWaveDataset(data_root, split="test", mode="valid_length")
    model = load_model(checkpoint_path, device)
    metric_rows = []
    for window_sec in windows:
        predictions, observations = predict_at_window(
            model, dataset, window_sec, device, batch_size
        )
        metrics = compute_metrics(predictions, observations)
        metric_rows.append({"window_sec": window_sec, "n": len(observations), **metrics})
    return pd.DataFrame(metric_rows)


def plot_metrics(metrics, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
    settings = (("R2", r"$R^2$"), ("MAE", r"MAE of $\log_{10}A$"), ("RMSE", r"RMSE of $\log_{10}A$"))
    for axis, (column, label) in zip(axes, settings):
        axis.plot(metrics["window_sec"], metrics[column], marker="o", linewidth=1.8)
        axis.set_xlabel("P-wave observation duration (s)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    device = choose_device(DEVICE)
    checkpoint = resolve_checkpoint(MODEL_DIR, SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_checkpoint(
        checkpoint, DATA_ROOT, device, BATCH_SIZE
    )
    metrics_path = OUTPUT_DIR / "metrics_by_window.csv"
    figure_path = OUTPUT_DIR / "performance_by_window.png"
    metrics.to_csv(metrics_path, index=False)
    plot_metrics(metrics, figure_path)
    print(f"Checkpoint: {checkpoint}")
    print(metrics.to_string(index=False))
    print(f"Metrics: {metrics_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
