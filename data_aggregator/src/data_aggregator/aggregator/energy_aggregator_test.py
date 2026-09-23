from pathlib import Path

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from data_aggregator.util.frame_io import FrameIO
from .energy_aggregator import EnergyAggregator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def run_data_single():
    # pylint: disable=line-too-long
    data = """
host,tool,dataset,mode,strength,threading,run,timestamp,voltage,current,power,real,size,power_duration,energy_used
No Unit,No Unit,No Unit,No Unit,No Unit,No Unit,No Unit,No Unit,volt,ampere,ampere·volt,second,byte,second,ampere·second·volt
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125,13.96,1.0,,
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316,13.96,1.0,0.01,0.025064607316
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775,13.96,1.0,0.01,0.024467147749999998
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.301,5.12401,0.4775,2.446714775,13.96,1.0,0.01,0.024467147749999998
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.311,5.10017,0.64122,3.2703310074000003,13.96,1.0,0.01,0.032703310074
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.321,5.10267,0.64941,3.3137249247,13.96,1.0,0.01,0.033137249247
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.331,5.12401,0.66851,3.4254519251000004,13.96,1.0,0.01,0.034254519251
raspi5,bzip2,sensor,compress,default,single,1,2026-04-07 07:40:28.341,5.12401,0.76748,3.932575194800001,13.96,1.0,0.01,0.03932575194800001"""
    return data


@pytest.fixture
def single_run_data_frame(run_data_single):
    return _as_dataframe(run_data_single)


def _as_dataframe(data: str) -> pd.DataFrame:
    io = FrameIO()
    return io.load_str(data)


@pytest.fixture
def aggregator():
    return EnergyAggregator(Path())


def test_aggregate_energy_single(aggregator, single_run_data_frame):
    data = """
host,tool,dataset,mode,strength,threading,run,sample_duration,energy,size,average_run_power
No Unit,No Unit,No Unit,No Unit,No Unit,No Unit,No Unit,second,joules,byte,watt
raspi5,bzip2,sensor,compress,default,single,1,0.07,0.21341973333599978,1.0,3.0488533333714254"""
    df_expected = _as_dataframe(data)

    df_actual = aggregator.aggregate_energy(single_run_data_frame)

    for key in ["sample_duration", "energy", "average_run_power"]:
        df_expected[key] = df_expected[key].astype("float")
        df_actual[key] = df_actual[key].astype("float")
    assert_frame_equal(df_actual, df_expected, rtol=1e-7, atol=1e-9)
