from pathlib import Path
from io import StringIO

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from .average_calculator import AverageCalculator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def run_data():
    data = """
run,power
No Unit,ampere·second·volt
1,4.428333128232399
2,4.506464898589998
3,4.5322596144951985"""
    return data


@pytest.fixture
def run_data_frame(run_data):
    return _as_dataframe(run_data, times=False)


def _as_dataframe(data: str, times=True) -> pd.DataFrame:
    df = pd.read_csv(StringIO(data), header=[0, 1])

    names = df.columns.get_level_values(0)
    units = df.columns.get_level_values(1)

    df.columns = names  # flatten
    if times:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col, unit in zip(names, units):  # apply units
        if unit != "No Unit":
            df[col] = df[col].astype(f"pint[{unit}]")
    return df


@pytest.fixture
def calculator():
    return AverageCalculator(Path())


def test_calculate_average(calculator, run_data_frame):
    data = """
runs,power_average,power_std,power_var
No Unit,ampere·second·volt,ampere·second·volt,ampere²·second²·volt²
3,4.48901921377253,0.05411508639737879,0.002928442575795771
    """
    df_expected = _as_dataframe(data, times=False)

    df_actual = calculator._calculate_averages(run_data_frame)

    for column in ["power_average", "power_std", "power_var"]:
        df_expected[column] = df_expected[column].astype("float")
        df_actual[column] = df_actual[column].astype("float")
    assert_frame_equal(df_actual, df_expected, rtol=1e-7, atol=1e-9)
