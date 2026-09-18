from pathlib import Path
from unittest.mock import Mock

import pytest

from data_processor import ureg
from data_processor.calc.average_power import AveragePower

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def power_calculator():
    return AveragePower(Mock(spec=Path))


def test_calculate_power_one(power_calculator, sample_energy_df):
    result = power_calculator._calculate_power(sample_energy_df)

    row = result[result["host"] == "host1"].iloc[0]
    assert row["num_runs"] == 2
    expected = ((100 + 150) * ureg.joule) / ((10 + 15) * ureg.second)
    assert row["average_power"] == expected
    row = result[result["host"] == "host2"].iloc[0]
    assert row["num_runs"] == 2
    expected = ((200 + 250) * ureg.joule) / ((15 + 21) * ureg.second)
    assert row["average_power"] == expected

def test_calculate_power_grouping(power_calculator, sample_energy_df):
    result = power_calculator._calculate_power(sample_energy_df)

    assert len(result) == 2
    host1_row = result.iloc[0]
    assert host1_row["num_runs"] == 2
    assert host1_row["average_power"] == 10.0 * ureg.watt
    host2_row = result.iloc[1]
    assert host2_row["num_runs"] == 2
    assert host2_row["average_power"] == 12.5 * ureg.watt
