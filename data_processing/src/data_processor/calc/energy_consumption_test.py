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
        "host": ["radxax4", "radxax4"],
        "tool": ["gzip", "gzip"],
        "dataset": ["image", "image"],
        "mode": ["compress", "compress"],
        "strength": ["default", "default"],
        "threading": ["single", "single"],
        "run": [1, 2],
        "energy": [100.0, 120.0],
        "real": [10.0, 10.0],
        "power": [10.0, 12.0],
    })
    df["energy"] = df["energy"].astype("pint[joule]")
    df["real"] = df["real"].astype("pint[second]")
    df["power"] = df["power"].astype("pint[watt]")
    return df


@pytest.fixture
def sample_idle_power_df():
    df = pd.DataFrame({
        "host": ["radxax4"],
        "average_power": [5.0],
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
    assert radxax4_row["average_energy_total"] == 110.0 * ureg.joule


def test_energy_net(energy_consumption, sample_energy_df, sample_idle_power_df):
    result = energy_consumption._calculate_energy_consumption(sample_energy_df, sample_idle_power_df)

    radxax4_row = result[result["host"] == "radxax4"].iloc[0]
    assert radxax4_row["num_runs"] == 2
    assert radxax4_row["average_energy_net"] == 60.0 * ureg.joule
