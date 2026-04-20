from pathlib import Path
from io import StringIO

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from .power_aggregator import PowerAggregator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def run_data_single():
    data = """
run,timestamp,voltage,current,energy,real,size,power_duration,power
No Unit,No Unit,volt,ampere,ampere·volt,second,byte,second,ampere·second·volt
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125,13.96,1.0,,
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316,13.96,1.0,0.01,0.025064607316
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775,13.96,1.0,0.01,0.024467147749999998
1,2026-04-07 07:40:28.301,5.12401,0.4775,2.446714775,13.96,1.0,0.01,0.024467147749999998
1,2026-04-07 07:40:28.311,5.10017,0.64122,3.2703310074000003,13.96,1.0,0.01,0.032703310074
1,2026-04-07 07:40:28.321,5.10267,0.64941,3.3137249247,13.96,1.0,0.01,0.033137249247
1,2026-04-07 07:40:28.331,5.12401,0.66851,3.4254519251000004,13.96,1.0,0.01,0.034254519251
1,2026-04-07 07:40:28.341,5.12401,0.76748,3.932575194800001,13.96,1.0,0.01,0.03932575194800001"""
    return data


@pytest.fixture
def single_run_data_frame(run_data_single):
    return _as_dataframe(run_data_single)


def _as_dataframe(data: str, times=True) -> pd.DataFrame:
    df = pd.read_csv(StringIO(data), header=[0, 1])

    names = df.columns.get_level_values(0)
    units = df.columns.get_level_values(1)

    df.columns = names  # flatten
    if times:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col, unit in zip(names, units):  # apply units
        if unit != "No Unit":
            df[col] = df[col].astype(f"pint[{unit}]")
    return df


@pytest.fixture
def aggregator():
    return PowerAggregator(Path())


def test_aggregate_power_single(aggregator, single_run_data_frame):
    data = """
run,power,real,size
No Unit,ampere·second·volt,second,byte
1,0.21341973333599978,13.96,1.0"""
    df_expected = _as_dataframe(data, times=False)

    df_actual = aggregator._aggregate_power(single_run_data_frame)

    df_expected["power"] = df_expected["power"].astype("float")
    df_actual["power"] = df_actual["power"].astype("float")
    assert_frame_equal(df_actual, df_expected, rtol=1e-7, atol=1e-9)
