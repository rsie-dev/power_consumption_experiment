from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd

from data_processor import ureg
from data_processor.calc.power import Power

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def power_calculator():
    return Power(Mock(spec=Path))


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "host": ["host1", "host1", "host2", "host2"],
        "tool": ["gzip", "gzip", "gzip", "gzip"],
        "dataset": ["image", "image", "image", "image"],
        "mode": ["compress", "compress", "compress", "compress"],
        "strength": ["default", "default", "default", "default"],
        "threading": ["single", "single", "single", "single"],
        "run": [1, 2, 1, 2],
        "energy": [100.0, 150.0, 200.0, 250.0],
        "duration": [10.0, 15.0, 15.0, 21.0],
    })
    df["energy"] = df["energy"].astype("pint[joule]")
    df["duration"] = df["duration"].astype("pint[second]")
    return df


def test_calculate_power_basic(power_calculator, sample_df):
    result = power_calculator._calculate_power(sample_df)

    row = result[result["host"] == "host1"].iloc[0]
    assert row["num_runs"] == 2
    expected = (100 * ureg.joule + 150 * ureg.joule) / (10 * ureg.second + 15 * ureg.second)
    assert row["average_power"] == expected
    row = result[result["host"] == "host2"].iloc[0]
    assert row["num_runs"] == 2
    expected = (200 * ureg.joule + 250 * ureg.joule) / (15 * ureg.second + 21 * ureg.second)
    assert row["average_power"] == expected

def test_calculate_power_grouping(power_calculator, sample_df):
    result = power_calculator._calculate_power(sample_df)

    assert len(result) == 2
    host1_row = result.iloc[0]
    assert host1_row["num_runs"] == 2
    assert host1_row["average_power"] == 10.0 * ureg.watt
    host2_row = result.iloc[1]
    assert host2_row["num_runs"] == 2
    assert host2_row["average_power"] == 12.5 * ureg.watt
