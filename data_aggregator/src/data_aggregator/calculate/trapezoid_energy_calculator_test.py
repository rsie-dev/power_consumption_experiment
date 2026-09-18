import pytest
import pandas as pd
from pint.testing import assert_allclose

from data_aggregator.util.frame_io import FrameIO

from .trapezoid_energy_calculator import TrapezoidEnergyCalculator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


def _as_dataframe(data: str) -> pd.DataFrame:
    io = FrameIO()
    return io.load_str(data)


@pytest.fixture
def calculator():
    return TrapezoidEnergyCalculator()


def test_calculate_energy_single(calculator, power_data_one):
    data = """
run,timestamp,voltage,current,power,power_duration,energy_used
No Unit,No Unit,volt,ampere,watt,second,joule
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125,,
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316,0.01,0.025262221720499998
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775,0.01,0.024765877532999997
"""
    df_expected = _as_dataframe(data)

    df_actual = calculator.calculate_energy(power_data_one)

    for column in ["voltage", "current", "power", "power_duration", "energy_used"]:
        assert_allclose(
            df_actual[column].pint.quantity,
            df_expected[column].pint.quantity,
            rtol=1e-7,
            atol=1e-9,
        )


def test_calculate_energy_double(calculator, power_data_two):
    data = """
run,timestamp,voltage,current,power,power_duration,energy_used
No Unit,No Unit,volt,ampere,watt,second,joule
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125,,
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316,0.01,0.025262221720499998
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775,0.01,0.024765877532999997
2,2026-04-07 07:41:05.513,5.12585,0.49288,2.526428948,,
2,2026-04-07 07:41:05.523,5.12585,0.49189,2.5213543565,0.01,0.025238916522499998
2,2026-04-07 07:41:05.533,5.12585,0.51322,2.630688737,0.01,0.0257602154675
"""
    df_expected = _as_dataframe(data)

    df_actual = calculator.calculate_energy(power_data_two)

    for column in ["voltage", "current", "power", "power_duration", "energy_used"]:
        assert_allclose(
            df_actual[column].pint.quantity,
            df_expected[column].pint.quantity,
            rtol=1e-7,
            atol=1e-9,
        )
