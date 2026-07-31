from unittest.mock import Mock
from pathlib import Path

import pytest
import pandas as pd


from data_processor import ureg
from .energy_consumption import EnergyConsumption

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def sample_energy_df():
    df = pd.DataFrame({
        "host": ["radxax4", "radxax4", "raspi5", "raspi5"],
        "tool": ["gzip"] * 4,
        "dataset": ["image"] * 4,
        "mode": ["compress"] * 4,
        "strength": ["default"] * 4,
        "threading": ["single"] * 4,
        "run": [1, 2, 1, 2],
        "energy": [100.0, 120.0, 50.0, 70.0],
        "real": [10.0, 10.0, 8.0, 8.0],
        "power": [10.0, 12.0, 6.0, 8.0],
    })
    df["energy"] = df["energy"].astype("pint[joule]")
    df["real"] = df["real"].astype("pint[second]")
    df["power"] = df["power"].astype("pint[watt]")
    return df


@pytest.fixture
def sample_idle_power_df():
    df = pd.DataFrame({
        "host": ["radxax4", "raspi5"],
        "average_power": [5.0, 2.0],
    })
    df["average_power"] = df["average_power"].astype("pint[watt]")
    return df


@pytest.fixture
def energy_consumption():
    return EnergyConsumption(Mock(spec=Path))


def test_energy_total(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    radxax4_row = result[result["host"] == "radxax4"].iloc[0]
    assert radxax4_row["num_runs"] == 2
    expected = (100.0 * ureg.joule + 120.0 * ureg.joule) / 2
    assert radxax4_row["average_energy_consumption_total"] == expected
    raspi5_row = result[result["host"] == "raspi5"].iloc[0]
    assert raspi5_row ["num_runs"] == 2
    expected = (50.0 * ureg.joule + 70.0 * ureg.joule) / 2
    assert raspi5_row ["average_energy_consumption_total"] == expected

def test_energy_net(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    radxax4_row = result[result["host"] == "radxax4"].iloc[0]
    assert radxax4_row["num_runs"] == 2
    expected = ((100.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt) +
                (120.0 * ureg.joule - 10 * ureg.second * 5 * ureg.watt)) / 2
    assert radxax4_row["average_energy_consumption_net"] == expected
    raspi5_row = result[result["host"] == "raspi5"].iloc[0]
    assert raspi5_row ["num_runs"] == 2
    expected = ((50.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt) +
                (70.0 * ureg.joule - 8 * ureg.second * 2 * ureg.watt)) / 2
    assert raspi5_row ["average_energy_consumption_net"] == expected


def test_energy_norm(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    radxax4_row = result[result["host"] == "radxax4"].iloc[0]
    assert radxax4_row["num_runs"] == 2
    assert radxax4_row["average_energy_consumption_norm_0_watt"] == 60.0 * ureg.joule
    assert radxax4_row["average_energy_consumption_norm_1_watt"] == 60.0 * ureg.joule + 10 * ureg.second * 1 * ureg.watt
    raspi5_row = result[result["host"] == "raspi5"].iloc[0]
    assert raspi5_row ["num_runs"] == 2
    assert raspi5_row ["average_energy_consumption_norm_0_watt"] == 44.0 * ureg.joule
    assert raspi5_row ["average_energy_consumption_norm_1_watt"] == 44.0 * ureg.joule + 8 * ureg.second * 1 * ureg.watt
