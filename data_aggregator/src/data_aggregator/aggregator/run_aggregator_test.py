from pathlib import Path
import datetime as dt

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from data_aggregator.util.frame_io import FrameIO
from data_aggregator.common import Measurement
from data_aggregator.common import RunInfo
from .run_aggregator import RunAggregator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def readings():
    data = """
timestamp,voltage,current
No Unit,volt,ampere
2026-09-04T09:29:01.000+02:00,12.05609,0.74541
2026-09-04T09:29:02.000+02:00,12.11117,0.47726
2026-09-04T09:29:03.000+02:00,12.09992,0.46857
2026-09-04T09:29:04.000+02:00,12.12872,0.46684
2026-09-04T09:29:05.000+02:00,12.12622,0.51521
"""
    return data


@pytest.fixture
def readings_df(readings):
    return _as_dataframe(readings)


def _as_dataframe(data: str) -> pd.DataFrame:
    io = FrameIO()
    return io.load_str(data)


@pytest.fixture
def aggregator():
    return RunAggregator(Path())


def test_cut_lead_tail(aggregator, readings_df):
    measurement = Measurement(
        start=dt.datetime.fromisoformat("2026-09-04T09:29:01.000+02:00"),
        end = dt.datetime.fromisoformat("2026-09-04T09:29:05.000+02:00"),
        readings=readings_df,
        timings=None,
        count=None,
    )
    run_info = RunInfo(run=0, measurement=measurement)

    df_actual = aggregator._cut_lead_tail(run_info)

    #df_expected = readings_df.iloc[1:-1].copy()
    df_expected = readings_df.copy()
    df_expected = df_expected.reset_index(drop=True)
    assert_frame_equal(df_actual, df_expected, rtol=1e-7, atol=1e-9)
