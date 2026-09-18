import pytest
import pandas as pd


@pytest.fixture
def sample_idle_power_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "host": ["radxax4", "raspi5"],
        "average_power": [5.0, 2.0],
    })
    df["average_power"] = df["average_power"].astype("pint[watt]")
    return df
