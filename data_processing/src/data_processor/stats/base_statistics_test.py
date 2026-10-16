from pathlib import Path
from unittest.mock import Mock

import pytest

from .base_statistics import BaseStatistics

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def calculator():
    return BaseStatistics(Mock(spec=Path), 1, None, "single", "compress", False)

def test_extract_factors_one(calculator):
    actual_relevant_factors = calculator._extract_factors("C(tool, Sum)")

    assert actual_relevant_factors == "tool"


def test_extract_factors_two(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum)")

    assert actual_relevant_factors == ("host", "tool")


def test_extract_factors_three(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum):C(dataset, Sum)")

    assert actual_relevant_factors == ("host", "tool", "dataset")
