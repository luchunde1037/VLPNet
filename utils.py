"""Shared training and evaluation utilities."""

import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, settings, save_path, best_loss):
    """Save a model checkpoint and its training settings."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_loss,
            "settings": settings,
        },
        save_path,
    )


def append_csv(path, row, header=None):
    """Append one row to a CSV file and create its header when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file and header is not None:
            writer.writerow(header)
        writer.writerow(row)


def count_parameters(model):
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_metrics(predictions, observations):
    """Compute regression metrics using prediction minus observation as error."""
    predictions = np.asarray(predictions)
    observations = np.asarray(observations)
    errors = predictions - observations
    denominator = np.sum((observations - observations.mean()) ** 2)
    r2 = np.nan if denominator <= 0 else 1.0 - np.sum(errors ** 2) / denominator
    correlation = (
        np.corrcoef(predictions, observations)[0, 1]
        if predictions.std() > 0 and observations.std() > 0
        else np.nan
    )
    return {
        "R2": r2,
        "correlation": correlation,
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "bias": np.mean(errors),
    }

