"""Dataset and data-loader utilities for VLPNet."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 100
MAX_SECONDS = 6.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)
SPLIT_DIRECTORIES = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


class VariableLengthPWaveDataset(Dataset):
    """Load preprocessed P-wave records from one dataset split."""

    def __init__(
        self,
        data_root,
        split="train",
        mode="random_crop",
        fixed_window_sec=None,
        min_window_sec=0.5,
        normalize=None,
    ):
        if split not in SPLIT_DIRECTORIES:
            raise ValueError(f"Unknown split: {split}")
        if mode == "fixed" and fixed_window_sec is None:
            raise ValueError("fixed_window_sec is required when mode='fixed'.")

        self.split_dir = Path(data_root) / SPLIT_DIRECTORIES[split]
        metadata_path = self.split_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        required = {"waveform_file", "log_A", "valid_len"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")

        self.mode = mode
        self.fixed_window_sec = fixed_window_sec
        self.min_samples = int(round(min_window_sec * SAMPLE_RATE))
        self.normalize = normalize

        waveforms, valid_lengths, targets, record_ids = [], [], [], []
        for row in metadata.itertuples(index=False):
            waveform_path = self.split_dir / str(row.waveform_file)
            if not waveform_path.exists():
                raise FileNotFoundError(f"Waveform file not found: {waveform_path}")
            with np.load(waveform_path, allow_pickle=False) as item:
                waveform = item["ud2"].astype(np.float32).reshape(-1)

            valid_length = min(int(row.valid_len), len(waveform), MAX_SAMPLES)
            if valid_length < self.min_samples:
                continue
            waveform = waveform[:MAX_SAMPLES]
            waveform = np.pad(waveform, (0, MAX_SAMPLES - len(waveform)))

            waveforms.append(waveform)
            valid_lengths.append(valid_length)
            targets.append(float(row.log_A))
            record_ids.append(str(getattr(row, "record_id", waveform_path.stem)))

        if not waveforms:
            raise ValueError(f"No usable records were found in {self.split_dir}")

        self.waveforms = np.stack(waveforms)
        self.valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.record_ids = record_ids

        # Backward-compatible attribute names used by the evaluation routines.
        self.waveform = self.waveforms
        self.valid_len = self.valid_lengths
        self.log_A = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        waveform = self.waveforms[index].copy()
        valid_length = int(self.valid_lengths[index])

        if self.mode == "random_crop":
            crop_length = np.random.randint(self.min_samples, valid_length + 1)
        elif self.mode == "fixed":
            requested = int(round(self.fixed_window_sec * SAMPLE_RATE))
            crop_length = min(requested, valid_length)
        else:
            crop_length = valid_length

        waveform[crop_length:] = 0.0
        if self.normalize == "peak":
            scale = np.max(np.abs(waveform[:crop_length])) + 1e-8
            waveform /= scale

        return (
            torch.from_numpy(waveform).unsqueeze(-1),
            torch.tensor(crop_length, dtype=torch.long),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )


def prepare_dataloaders(
    data_root,
    batch_size=32,
    normalize=None,
    num_workers=0,
    min_window_sec=0.5,
):
    """Create training, validation, and test data loaders."""
    train_set = VariableLengthPWaveDataset(
        data_root,
        split="train",
        mode="random_crop",
        normalize=normalize,
        min_window_sec=min_window_sec,
    )
    validation_set = VariableLengthPWaveDataset(
        data_root,
        split="validation",
        mode="valid_length",
        normalize=normalize,
        min_window_sec=min_window_sec,
    )
    test_set = VariableLengthPWaveDataset(
        data_root,
        split="test",
        mode="valid_length",
        normalize=normalize,
        min_window_sec=min_window_sec,
    )
    kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(validation_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )

