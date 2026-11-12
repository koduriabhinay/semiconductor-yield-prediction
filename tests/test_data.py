"""Tests for SECOM loading."""

import numpy as np


def test_load_secom_shape(secom):
    X, y, run_time = secom
    assert X.shape == (1567, 590)
    assert len(y) == 1567
    assert list(X.columns[:3]) == ["time", "s1", "s2"]
    assert X.columns[-1] == "s589"


def test_label_mapping_binary(secom):
    X, y, run_time = secom
    assert set(np.unique(y)) == {0, 1}
    # SECOM is imbalanced: ~104 fails out of 1567
    assert 50 < int(y.sum()) < 200
