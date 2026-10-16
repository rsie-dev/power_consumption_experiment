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
    assert host1_row["run"] == 1
    expected_ee_total = DataSet.IMAGE.size / (100 * ureg.joule)
    assert host1_row["energy_efficiency_total"] == expected_ee_total

def test_energy_net(energy_efficiency, sample_energy_df, sample_idle_power_df):
    result = energy_efficiency._calculate_energy_efficiency(sample_energy_df, sample_idle_power_df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["run"] == 1
    net_consumption = (100 * ureg.joule) - (5 * ureg.watt) * (10 * ureg.second)
    expected_ee_net = DataSet.IMAGE.size / net_consumption
    assert host1_row["energy_efficiency_net"] == expected_ee_net


def test_aggregate_energy_efficiency_total(energy_efficiency, sample_energy_df, sample_idle_power_df):
    df = energy_efficiency._calculate_energy_efficiency(sample_energy_df, sample_idle_power_df)

    result = energy_efficiency._aggregate_energy_efficiency(df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    expected = ((DataSet.IMAGE.size / (100.0 * ureg.joule)) +
                (DataSet.IMAGE.size / (150.0 * ureg.joule))) / 2
    assert host1_row["average_energy_efficiency_total"] == expected
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    expected = ((DataSet.IMAGE.size / (200.0 * ureg.joule)) +
                (DataSet.IMAGE.size / (250.0 * ureg.joule))) / 2
    assert host2_row ["average_energy_efficiency_total"] == expected


def test_aggregate_energy_efficiency_net(energy_efficiency, sample_energy_df, sample_idle_power_df):
    df = energy_efficiency._calculate_energy_efficiency(sample_energy_df, sample_idle_power_df)

    result = energy_efficiency._aggregate_energy_efficiency(df)

    host1_row = result[result["host"] == "host1"].iloc[0]
    assert host1_row["num_runs"] == 2
    expected = ((DataSet.IMAGE.size / (100.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt)) +
                (DataSet.IMAGE.size / (150.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt))) / 2
    assert host1_row["average_energy_efficiency_net"] == expected
    host2_row = result[result["host"] == "host2"].iloc[0]
    assert host2_row ["num_runs"] == 2
    expected = ((DataSet.IMAGE.size / (200.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt)) +
                (DataSet.IMAGE.size / (250.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt))) / 2
    assert host2_row ["average_energy_efficiency_net"] == expected
