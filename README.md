# VLPNet:A Unified Model for Variable-Length P-Wave Records

VLPNet is a unified deep-learning framework for magnitude-related regression from variable-length initial P-wave observations. A single trained model accepts observation durations from 0.5 to 6.0 s and uses the valid sequence length to exclude zero-padded samples from temporal encoding.

## Repository structure

```text
VLPNet/
├── data/                       # Public 100/30/30 workflow subset
├── data_full/                  # Complete-dataset metadata 
├── log/                        # Training logs
├── model/                      # Best checkpoint
├── results/                    # Evaluation outputs
├── data_loader.py              # Variable-length data loading
├── evaluate.py                 # Evaluation of one checkpoint
├── evaluate_five_seeds.py      # Aggregation across five checkpoints
├── export_requirements.py      # Export dependencies from the active environment
├── LICENSE
├── models.py                   # VLPNet architecture
├── README.md
├── requirements.txt
├── train.py                    # Model training and validation
└── utils.py                    # Shared utilities and metrics
```


## Environment

The code was developed and tested using the following environment:

| Component | Version or configuration |
|---|---|
| Operating system | Windows 10, version 10.0.19041 SP0 |
| Python | 3.8.0 |
| PyTorch | 2.3.1 |
| PyTorch CUDA build | 11.8 |
| CUDA available | Yes |
| CUDA device count | 1 |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| NumPy | 1.24.3 |
| pandas | 2.0.3 |
| Matplotlib | 3.7.5 |

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data source

The strong-motion records used in this study were obtained from the KiK-net database maintained by the National Research Institute for Earth Science and Disaster Resilience (NIED), Japan:

- [NIED K-NET and KiK-net](https://www.kyoshin.bosai.go.jp/en/)
- [Download by data condition](https://www.kyoshin.bosai.go.jp/en/dtdownload/)
- [Download via HTTPS](https://www.kyoshin.bosai.go.jp/en/https_download/)

User registration is required by NIED to download waveform data. Users must comply with the data-use requirements provided by NIED. The recommended citation is:

> National Research Institute for Earth Science and Disaster Resilience (2019). NIED K-NET, KiK-net. https://doi.org/10.17598/NIED.0004

## Data organization

### Public workflow subset

The `data/` directory contains a limited subset for verifying the code workflow:

```text
data/
├── train/          # 100 records
│   ├── metadata.csv
│   └── *.npz
├── validation/     # 30 records
│   ├── metadata.csv
│   └── *.npz
└── test/           # 30 records
    ├── metadata.csv
    └── *.npz
```

This subset is provided solely to verify data loading, model training, checkpoint saving, and evaluation. The complete waveform dataset is not redistributed because the original KiK-net records are third-party data maintained by NIED and are subject to its user-registration and data-use requirements.

The `data_full/metadata.csv` file provides the record-level information and dataset split assignments used in this study:

| Column | Description |
|---|---|
| `record_id` | Unique identifier assigned to each station–event record |
| `waveform_file` | Filename of the corresponding processed waveform file |
| `station_code` | KiK-net station code used to identify and download the original record |
| `origin_time` | Earthquake origin time |
| `record_time` | Recording time associated with the waveform |
| `mag` | Earthquake magnitude provided in the source metadata |
| `split` | Dataset assignment: `train`, `validation`, or `test` |

The station codes, earthquake origin times, and record times allow registered users to identify and download the corresponding original records from the official NIED K-NET and KiK-net website. Together with the dataset split assignments and the preprocessing procedure described in the paper, this metadata can be used to reconstruct the complete dataset employed in this study. The `waveform_file` column specifies the expected filename of each processed waveform file after reconstruction.

## Training

The main settings are grouped at the top of `train.py`:

```python
DATA_ROOT = PROJECT_ROOT / "data_full"
MODEL_DIR = PROJECT_ROOT / "model"
LOG_DIR = PROJECT_ROOT / "log"
SEED = 50
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
DEVICE = "auto"
```

Run `train.py` directly in PyCharm. To repeat the five independent experiments, set `SEED` successively to 10, 20, 30, 40, and 50.

For a short workflow test with the public subset, temporarily use:

```python
DATA_ROOT = PROJECT_ROOT / "data"
EPOCHS = 1
```

The training outputs are:

```text
log/
├── VLPNet_seed10.csv
├── VLPNet_seed20.csv
├── VLPNet_seed30.csv
├── VLPNet_seed40.csv
└── VLPNet_seed50.csv

model/
├── VLPNet_seed10_best.chkpt
├── VLPNet_seed20_best.chkpt
├── VLPNet_seed30_best.chkpt
├── VLPNet_seed40_best.chkpt
└── VLPNet_seed50_best.chkpt
```

The best checkpoint is selected using the mean validation loss across the predefined P-wave observation durations.

## Evaluation of one checkpoint

The main settings are grouped at the top of `evaluate.py`:

```python
DATA_ROOT = PROJECT_ROOT / "data_full"
MODEL_DIR = PROJECT_ROOT / "model"
SEED = 50
BATCH_SIZE = 32
DEVICE = "auto"
```

Run `evaluate.py` directly in PyCharm. The outputs for seed 50 are:

```text
results/evaluation_seed50/
├── metrics_by_window.csv
└── performance_by_window.png
```


## Reproducing the reported results

1. Register with NIED and download the KiK-net records identified by the released metadata.
2. Apply the preprocessing procedure described in the paper.
3. Save the processed `ud2` arrays using the filenames listed in `data_full`.
4. Run `train.py` for seeds 10, 20, 30, 40, and 50.
5. Run `evaluate_five_seeds.py`.
6. Compare `results/five_seeds/metrics_mean_std.csv` with the corresponding results reported in the paper.

Exact numerical reproduction requires the same records, temporal split, preprocessing, model settings, and random seeds used in the study.

## License and data terms

The source code is released under the MIT License. This license does not apply to K-NET or KiK-net records. The waveform data remain subject to the terms of NIED.

