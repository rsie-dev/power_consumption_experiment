from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd

from data_processor import ureg
from data_processor.calc.throughput import Throughput
from data_processor.data_set import DataSet

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def calculator():
    return Throughput(Mock(spec=Path))


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "host": ["host"] * 6,
        "tool": ["tool1", "tool1", "tool2", "tool2", "tool3", "tool3"],
        "dataset": ["image"] * 6,
        "mode": ["compress"] * 6,
        "strength": ["default"] * 6,
        "threading": ["single", "single", "single", "single", "multi", "multi"],
        "run": [1, 2, 1, 2, 1, 2],
        "duration": [1.0, 1.1, 0.9, 1.0, 1.2, 1.3],
    })
    df["duration"] = df["duration"].astype("pint[second]")
    return df


def test_calculate_throughput(calculator, sample_df):
    result = calculator._calculate_throughput(sample_df)

    row = result[result["tool"] == "tool1"].iloc[0]
    expected_tp = DataSet.IMAGE.size / (1 * ureg.second)
    assert row["throughput"] == expected_tp


def test_aggregate_throughput(calculator, sample_df):
    df = calculator._calculate_throughput(sample_df)

    result = calculator._aggregate_throughput(df)

    expected_throughput = (DataSet.IMAGE.size / (1.0 * ureg.second) + DataSet.IMAGE.size / (1.1 * ureg.second)) / 2
    row = result[result["tool"] == "tool1"].iloc[0]
    assert row["average_throughput"] == expected_throughput
