from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd

from data_processor import ureg
from data_processor.calc.compression_ratio import CompressionRatio
from data_processor.data_set import DataSet

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def calculator():
    return CompressionRatio(Mock(spec=Path))

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "host": ["host1"] * 6,
        "tool": ["tool1", "tool1", "tool2", "tool2", "tool3", "tool3"],
        "dataset": ["image"] * 6,
        "mode": ["compress"] * 6,
        "strength": ["default"] * 6,
        "threading": ["single", "single", "single", "single", "multi", "multi"],
        "run": [1] * 6,
        "size": [100, 50, 80, 40, 90, 45],
    })
    df["size"] = df["size"].astype("pint[byte]")
    return df


def test_calculate_compression_ratio_basic(calculator, sample_df):
    result = calculator._calculate_compression_ratio(sample_df)

    expected_ratio = DataSet.IMAGE.value / (100 * ureg.byte)
    row = result[result["tool"] == "tool1"].iloc[0]
    assert row["compression_ratio"] == expected_ratio
