from io import StringIO

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from .energy_calculator import EnergyCalculator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def run_data_single():
    data = """
run,timestamp,voltage,current,power
No Unit,No Unit,volt,ampere,ampere·volt
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775
"""
    return data


@pytest.fixture
def run_data_two(run_data_single):
    data = run_data_single + """
2,2026-04-07 07:41:05.513,5.12585,0.49288,2.526428948
2,2026-04-07 07:41:05.523,5.12585,0.49189,2.5213543565
2,2026-04-07 07:41:05.533,5.12585,0.51322,2.630688737
"""
    return data


@pytest.fixture
def single_run_data_frame(run_data_single):
    return _as_dataframe(run_data_single)


@pytest.fixture
def run_data_frame_two(run_data_two):
    return _as_dataframe(run_data_two)


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
    return EnergyCalculator()


def test_calculate_power_single(calculator, single_run_data_frame):
    data = """
run,timestamp,voltage,current,power,power_duration,energy_used
No Unit,No Unit,volt,ampere,ampere·volt,second,ampere·second·volt
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125,,
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316,0.01,0.025064607316
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775,0.01,0.024467147749999998
"""
    df_expected = _as_dataframe(data)

    df_actual = calculator.calculate_energy(single_run_data_frame)

    for column in ["power", "energy_used"]:
        df_expected[column] = df_expected[column].astype("float")
        df_actual[column] = df_actual[column].astype("float")
    assert_frame_equal(df_actual, df_expected, rtol=1e-7, atol=1e-9)


def test_calculate_power_double(calculator, run_data_frame_two):
    data = """
run,timestamp,voltage,current,power,power_duration,energy_used
No Unit,No Unit,volt,ampere,ampere·volt,second,ampere·second·volt
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125,,
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316,0.01,0.025064607316
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775,0.01,0.024467147749999998
2,2026-04-07 07:41:05.513,5.12585,0.49288,2.526428948,,
2,2026-04-07 07:41:05.523,5.12585,0.49189,2.5213543565,0.01,0.025213543565
2,2026-04-07 07:41:05.533,5.12585,0.51322,2.630688737,0.01,0.02630688737
"""
    df_expected = _as_dataframe(data)

    df_actual = calculator.calculate_energy(run_data_frame_two)

    for column in ["power", "energy_used"]:
        df_expected[column] = df_expected[column].astype("float")
        df_actual[column] = df_actual[column].astype("float")
    assert_frame_equal(df_actual, df_expected, rtol=1e-7, atol=1e-9)
