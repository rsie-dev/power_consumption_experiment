from unittest.mock import Mock
from pathlib import Path

import pytest


from data_processor import ureg
from .energy_consumption import EnergyConsumption

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def energy_consumption():
    return EnergyConsumption(Mock(spec=Path))


def test_energy_total(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    expected = ((100.0 + 150.0) * ureg.joule) / 2
    assert host1_row["average_energy_consumption_total"] == expected
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    expected = ((200.0 + 250.0) * ureg.joule) / 2
    assert host2_row ["average_energy_consumption_total"] == expected

def test_energy_net(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    expected = ((100.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt) +
                (150.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt)) / 2
    assert host1_row["average_energy_consumption_net"] == expected
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    expected = ((200.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt) +
                (250.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt)) / 2
    assert host2_row ["average_energy_consumption_net"] == expected


def test_energy_norm(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    assert host1_row["average_energy_consumption_norm_0_watt"] == 75.0 * ureg.joule
    assert host1_row["average_energy_consumption_norm_1_watt"] == 75.0 * ureg.joule + 10 * ureg.second * 1 * ureg.watt
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    assert host2_row ["average_energy_consumption_norm_0_watt"] == 209.0 * ureg.joule
    assert host2_row ["average_energy_consumption_norm_1_watt"] == 209.0 * ureg.joule + 8 * ureg.second * 1 * ureg.watt
