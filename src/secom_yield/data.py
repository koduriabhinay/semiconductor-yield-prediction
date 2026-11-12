"""Loading utilities for the SECOM semiconductor manufacturing dataset.

The SECOM dataset (UCI Machine Learning Repository, McCann & Johnston) contains
sensor measurements from a semiconductor wafer fabrication process:

- ``secom.data``: 1,567 production runs x 590 columns. Column 0 is a timestamp;
  columns 1..589 are sensor signals (with many missing values).
- ``secom_labels.data``: one ``-1`` (pass) / ``+1`` (fail) label per run plus a
  human-readable timestamp (104 fails out of 1,567 runs, ~6.6% fail rate).

Fail is mapped to the positive class (1) so that model outputs read naturally
as *fail probability*.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SECOM_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
)
SECOM_LABELS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
)

N_RUNS_EXPECTED = 1567
N_COLS_EXPECTED = 590  # column 0 = timestamp, columns 1..589 = sensor signals
N_SIGNALS = N_COLS_EXPECTED - 1

RANDOM_STATE = 42


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def download_secom(dest_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Download secom.data and secom_labels.data from the UCI archive."""
    dest = Path(dest_dir) if dest_dir else default_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    data_path = dest / "secom.data"
    labels_path = dest / "secom_labels.data"
    if not data_path.exists():
        urllib.request.urlretrieve(SECOM_DATA_URL, data_path)
    if not labels_path.exists():
        urllib.request.urlretrieve(SECOM_LABELS_URL, labels_path)
    return data_path, labels_path


def load_secom(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the SECOM dataset.

    Returns
    -------
    X : DataFrame, shape (1567, 590)
        Columns ``time`` (column 0, used for run ordering) and
        ``s1``..``s589`` (sensor signals, may contain NaN).
    y : Series of int, shape (1567,)
        1 = fail, 0 = pass.
    run_time : Series of float
        The raw timestamp column, for time-ordered excursion analysis.
    """
    ddir = Path(data_dir) if data_dir else default_data_dir()
    data_path = ddir / "secom.data"
    labels_path = ddir / "secom_labels.data"
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"SECOM files not found in {ddir}. "
            "Run `python scripts/download_data.py` first."
        )

    X = pd.read_csv(
        data_path, sep=r"\s+", header=None, na_values=["NaN", "nan", "NULL"]
    )
    if X.shape != (N_RUNS_EXPECTED, N_COLS_EXPECTED):
        raise ValueError(f"Unexpected secom.data shape {X.shape}")
    X.columns = ["time"] + [f"s{i}" for i in range(1, N_COLS_EXPECTED)]

    labels = pd.read_csv(
        labels_path, sep=r"\s+", header=None, usecols=[0], names=["label"]
    )["label"]
    if len(labels) != N_RUNS_EXPECTED:
        raise ValueError(f"Unexpected label count {len(labels)}")
    y = (labels == 1).astype(int)  # +1 = fail -> 1, -1 = pass -> 0
    y.name = "fail"

    return X, y, X["time"]
