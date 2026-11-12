"""Download the SECOM dataset from the UCI ML Repository into data/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secom_yield.data import download_secom

if __name__ == "__main__":
    data_path, labels_path = download_secom()
    print(f"Downloaded:\n  {data_path}\n  {labels_path}")
