from pathlib import Path

import pytest
import pandas as pd
from pint.testing import assert_allclose

from data_aggregator.util.frame_io import FrameIO

from .average_power_calculator import AveragePowerCalculator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def energy_data() -> pd.DataFrame:
    data = """
run,power
No Unit,joule
1,4.428333128232399
2,4.506464898589998
3,4.5322596144951985"""
    return _as_dataframe(data)


def _as_dataframe(data: str) -> pd.DataFrame:
    io = FrameIO()
    return io.load_str(data)


@pytest.fixture
def calculator():
    return AveragePowerCalculator(Path())


def test_calculate_power_averages(calculator, energy_data):
    data = """
runs,power_average,power_std,power_var
No Unit,joule,joule,joule^2
3,4.48901921377253,0.05411508639737879,0.002928442575795771
    """
    df_expected = _as_dataframe(data)

    df_actual = calculator._calculate_power_averages(energy_data)

    for column in ["power_average", "power_std", "power_var"]:
        assert_allclose(
            df_actual[column].pint.quantity,
            df_expected[column].pint.quantity,
            rtol=1e-7,
            atol=1e-9,
        )
