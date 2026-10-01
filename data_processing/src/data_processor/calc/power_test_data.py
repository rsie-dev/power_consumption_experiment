import pytest
import pandas as pd


@pytest.fixture
def sample_idle_power_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "host": ["host1", "host2"],
        "duration": [2, 2],
        "energy": [10, 4],
    })
    df["duration"] = df["duration"].astype("pint[second]")
    df["energy"] = df["energy"].astype("pint[joule]")
    return df
