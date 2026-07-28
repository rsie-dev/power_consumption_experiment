import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

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
        "host": ["host1"] * 6,
        "tool": ["tool1", "tool1", "tool2", "tool2", "tool3", "tool3"],
        "dataset": ["image"] * 6,
        "mode": ["compress"] * 6,
        "strength": ["default"] * 6,
        "threading": ["single", "single", "single", "single", "multi", "multi"],
        "run": [1, 2, 1, 2, 1, 2],
        "real": [1.0, 1.1, 0.9, 1.0, 1.2, 1.3],
    })
    df["real"] = df["real"].astype("pint[second]")
    return df


def test_calculate_throughput_basic(calculator, sample_df):
    result = calculator._calculate_throughput(sample_df)

    expected_throughput = (DataSet.IMAGE.value / (1.0 * ureg.second) + DataSet.IMAGE.value / (1.1 * ureg.second)) / 2
    row = result[result["tool"] == "tool1"].iloc[0]
    assert row["average_throughput"] == expected_throughput
