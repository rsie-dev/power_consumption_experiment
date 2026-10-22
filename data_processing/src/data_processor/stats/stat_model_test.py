import pytest

from .stat_model import StatModel

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def stat_model():
    return StatModel()


def test_build_interaction_formula(stat_model):
    actual_formula = stat_model._build_interaction_formula("response", 2, ["host", "tool"])

    expected_formula = """
response ~ (
    C(host, Sum)
    + C(tool, Sum)
)**2
"""
    assert actual_formula == expected_formula


def test_build_full_formula(stat_model):
    actual_formula = stat_model._build_full_formula("response", ["host", "tool"])

    expected_formula = """
response ~
    C(host, Sum) *
    C(tool, Sum)
"""
    assert actual_formula == expected_formula
