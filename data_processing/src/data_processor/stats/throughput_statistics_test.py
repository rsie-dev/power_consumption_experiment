from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd

from .throughput_statistics import ThroughputStatistics

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

@pytest.fixture
def sample_throughput() -> pd.DataFrame:
    df = pd.DataFrame({
        "host": ["host1", "host2"],
        "throughput": [36992000.0, 35111050.0],
    })
    df["throughput"] = df["throughput"].astype("pint[byte/second]")
    return df


@pytest.fixture
def calculator():
    return ThroughputStatistics(Mock(spec=Path), 1, None, "single", "compress", False)


def test_calculate_base(calculator, sample_throughput):
    actual_df = calculator._calculate_base(sample_throughput)

    actual = actual_df["log_throughput"].iloc[0]
    assert actual == pytest.approx(17.426212231014187)


def test_extract_factors_one(calculator):
    actual_relevant_factors = calculator._extract_factors("C(tool, Sum)")

    assert actual_relevant_factors == "tool"


def test_extract_factors_two(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum)")

    assert actual_relevant_factors == ("host", "tool")


def test_extract_factors_three(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum):C(dataset, Sum)")

    assert actual_relevant_factors == ("host", "tool", "dataset")
