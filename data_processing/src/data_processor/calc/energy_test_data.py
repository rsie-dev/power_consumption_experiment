import pytest
import pandas as pd


@pytest.fixture
def sample_energy_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "host": ["host1", "host1", "host2", "host2"],
        "tool": ["gzip"] * 4,
        "dataset": ["image"] * 4,
        "mode": ["compress"] * 4,
        "strength": ["default"] * 4,
        "threading": ["single"] * 4,
        "run": [1, 2, 1, 2],
        "energy": [100.0, 150.0, 200.0, 250.0], # power
        "real": [10.0, 10.0, 8.0, 8.0],
        "power": [10.0, 12.0, 6.0, 8.0],
        "duration": [10.0, 15.0, 15.0, 21.0],
    })
    df["energy"] = df["energy"].astype("pint[joule]")
    df["real"] = df["real"].astype("pint[second]")
    df["power"] = df["power"].astype("pint[watt]")
    df["duration"] = df["duration"].astype("pint[second]")
    return df
