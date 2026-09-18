import pytest
import pandas as pd

from data_aggregator.util.frame_io import FrameIO

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def power_data_one() -> pd.DataFrame:
    data = """
run,timestamp,voltage,current,power
No Unit,No Unit,volt,ampere,watt
1,2026-04-07 07:40:28.271,5.12425,0.49685,2.5459836125
1,2026-04-07 07:40:28.281,5.12401,0.48916,2.5064607316
1,2026-04-07 07:40:28.291,5.12401,0.4775,2.446714775
"""
    return _as_dataframe(data)


@pytest.fixture
def power_data_two(power_data_one) -> pd.DataFrame:
    data = """
run,timestamp,voltage,current,power
No Unit,No Unit,volt,ampere,watt
2,2026-04-07 07:41:05.513,5.12585,0.49288,2.526428948
2,2026-04-07 07:41:05.523,5.12585,0.49189,2.5213543565
2,2026-04-07 07:41:05.533,5.12585,0.51322,2.630688737
"""
    two = _as_dataframe(data)
    return pd.concat([power_data_one, two])


def _as_dataframe(data: str) -> pd.DataFrame:
    io = FrameIO()
    return io.load_str(data)
