from pathlib import Path
from unittest.mock import Mock

import pytest

from .throughput_statistics import ThroughputStatistics

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def calculator():
    return ThroughputStatistics(Mock(spec=Path))


def test_extract_factors_one(calculator):
    actual_relevant_factors = calculator._extract_factors("C(tool, Sum)")

    assert actual_relevant_factors == "tool"


def test_extract_factors_two(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum)")

    assert actual_relevant_factors == ("host", "tool")


def test_extract_factors_three(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum):C(dataset, Sum)")

    assert actual_relevant_factors == ("host", "tool", "dataset")