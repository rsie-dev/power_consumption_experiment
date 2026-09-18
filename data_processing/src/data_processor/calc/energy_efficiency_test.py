from unittest.mock import Mock
from pathlib import Path

import pytest


from data_processor import ureg
from data_processor.data_set import DataSet
from .energy_efficiency import EnergyEfficiency

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def energy_efficiency():
    return EnergyEfficiency(Mock(spec=Path))


def test_energy_total(energy_efficiency, sample_energy_df, sample_idle_power_df):
    result = energy_efficiency._calculate_energy_efficiency(sample_energy_df, sample_idle_power_df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    expected = ((DataSet.IMAGE.value / (100.0 * ureg.joule)) +
                (DataSet.IMAGE.value / (120.0 * ureg.joule))) / 2
    assert host1_row["average_energy_efficiency_total"] == expected
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    expected = ((DataSet.IMAGE.value / (50.0 * ureg.joule)) +
                (DataSet.IMAGE.value / (70.0 * ureg.joule))) / 2
    assert host2_row ["average_energy_efficiency_total"] == expected

def test_energy_net(energy_efficiency, sample_energy_df, sample_idle_power_df):
    result = energy_efficiency._calculate_energy_efficiency(sample_energy_df, sample_idle_power_df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    expected = ((DataSet.IMAGE.value / (100.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt)) +
                (DataSet.IMAGE.value / (120.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt))) / 2
    assert host1_row["average_energy_efficiency_net"] == expected
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    expected = ((DataSet.IMAGE.value / (50.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt)) +
                (DataSet.IMAGE.value / (70.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt))) / 2
    assert host2_row ["average_energy_efficiency_net"] == expected
