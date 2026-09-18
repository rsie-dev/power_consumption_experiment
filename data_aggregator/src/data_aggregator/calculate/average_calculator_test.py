from pathlib import Path
from io import StringIO

import pytest
import pandas as pd
from pint.testing import assert_allclose

from data_aggregator.util.frame_io import adjust_pint_columns

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
    return _as_dataframe(run_data)


def _as_dataframe(data: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(data), header=[0, 1])
    return adjust_pint_columns(df)


@pytest.fixture
def calculator():
    return AverageCalculator(Path())


def test_calculate_average(calculator, run_data_frame):
    data = """
runs,power_average,power_std,power_var
No Unit,joule,joule,joule^2
3,4.48901921377253,0.05411508639737879,0.002928442575795771
    """
    df_expected = _as_dataframe(data)

    df_actual = calculator._calculate_averages(run_data_frame)

    for column in ["power_average", "power_std", "power_var"]:
        assert_allclose(
            df_actual[column].pint.quantity,
            df_expected[column].pint.quantity,
            rtol=1e-7,
            atol=1e-9,
        )
