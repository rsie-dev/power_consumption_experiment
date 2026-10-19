from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from .base_statistics import BaseStatistics

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def calculator():
    with patch.object(BaseStatistics, "__abstractmethods__", set()):
        yield BaseStatistics(Mock(spec=Path), False, "single", "compress", False)


def test_build_interaction_formula(calculator):
    actual_formula = calculator._build_interaction_formula("response", 2, ["host", "tool"])

    expected_formula = """
response ~ (
    C(host, Sum)
    + C(tool, Sum)
)**2
"""
    assert actual_formula == expected_formula


def test_build_full_formula(calculator):
    actual_formula = calculator._build_full_formula("response", ["host", "tool"])

    expected_formula = """
response ~
    C(host, Sum) *
    C(tool, Sum)
"""
    assert actual_formula == expected_formula


def test_extract_factors_one(calculator):
    actual_relevant_factors = calculator._extract_factors("C(tool, Sum)")

    assert actual_relevant_factors == "tool"


def test_extract_factors_two(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum)")

    assert actual_relevant_factors == ("host", "tool")


def test_extract_factors_three(calculator):
    actual_relevant_factors = calculator._extract_factors("C(host, Sum):C(tool, Sum):C(dataset, Sum)")

    assert actual_relevant_factors == ("host", "tool", "dataset")
