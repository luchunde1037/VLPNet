"""Train and validate VLPNet on variable-length P-wave records."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from data_loader import SAMPLE_RATE, prepare_dataloaders
from models import VLPNet
from utils import append_csv, count_parameters, save_checkpoint, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent

# PyCharm configuration: change SEED to train each independent model.
DATA_ROOT = PROJECT_ROOT / "data_full"
MODEL_DIR = PROJECT_ROOT / "model"
LOG_DIR = PROJECT_ROOT / "log"
SEED = 50
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
DEVICE = "auto"  # "auto", "cuda", or "cpu"

EVALUATION_WINDOWS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def choose_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_squared_error = 0.0
    total_records = 0
    for waveforms, lengths, targets in loader:
        waveforms = waveforms.float().to(device)
        lengths = lengths.to(device)
        targets = targets.float().to(device)
        optimizer.zero_grad()
        predictions = model(waveforms, lengths)
        loss = functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        total_squared_error += loss.item() * len(targets)
        total_records += len(targets)
    return total_squared_error / total_records


def evaluate_windows(model, dataset, device, windows=EVALUATION_WINDOWS, batch_size=32):
    """Evaluate mean squared error independently at each observation duration."""
    model.eval()
    losses, counts = [], []
    with torch.no_grad():
        for seconds in windows:
            crop_samples = int(round(seconds * SAMPLE_RATE))
            indices = np.flatnonzero(dataset.valid_lengths >= crop_samples)
            counts.append(len(indices))
            if len(indices) == 0:
                losses.append(np.nan)
                continue
            squared_error = 0.0
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                waveforms = dataset.waveforms[batch_indices].copy()
                waveforms[:, crop_samples:] = 0.0
                targets = torch.from_numpy(dataset.targets[batch_indices]).float().to(device)
                waveform_tensor = torch.from_numpy(waveforms).unsqueeze(-1).float().to(device)
                lengths = torch.full((len(batch_indices),), crop_samples, dtype=torch.long, device=device)
                predictions = model(waveform_tensor, lengths)
                squared_error += torch.sum((predictions - targets) ** 2).item()
            losses.append(squared_error / len(indices))
    finite_losses = [loss for loss in losses if np.isfinite(loss)]
    return float(np.mean(finite_losses)), losses, counts


def main():
    set_seed(SEED)
    device = choose_device(DEVICE)
    run_name = f"VLPNet_seed{SEED}"
    checkpoint_path = MODEL_DIR / f"{run_name}_best.chkpt"
    log_path = LOG_DIR / f"{run_name}.csv"

    train_loader, validation_loader, test_loader = prepare_dataloaders(
        DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    model = VLPNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float("inf")
    header = ["epoch", "train_loss", "validation_mean_loss"] + [
        f"validation_{seconds:g}s" for seconds in EVALUATION_WINDOWS
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(",".join(header) + "\n", encoding="utf-8")

    settings = {
        "data_root": str(DATA_ROOT),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "device": DEVICE,
    }

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation_loss, window_losses, counts = evaluate_windows(
            model, validation_loader.dataset, device, batch_size=BATCH_SIZE
        )
        scheduler.step()
        append_csv(
            log_path,
            [epoch + 1, train_loss, validation_loss, *window_losses],
        )
        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS}: train={train_loss:.6f}, "
            f"validation={validation_loss:.6f}, elapsed={time.time() - start:.1f}s"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                settings,
                checkpoint_path,
                best_loss,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_window_losses, test_counts = evaluate_windows(
        model, test_loader.dataset, device, batch_size=BATCH_SIZE
    )
    print(f"Test mean loss: {test_loss:.6f}")
    for seconds, loss, count in zip(EVALUATION_WINDOWS, test_window_losses, test_counts):
        print(f"  {seconds:>3.1f} s: loss={loss:.6f}, n={count}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
