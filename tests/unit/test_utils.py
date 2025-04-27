"""
Unit tests for the utils module.
"""

import pytest

from src.utils import filter_list, safe_get, sample_function


def test_sample_function():
    """Test the sample_function in utils.py."""
    result = sample_function(1, 2)
    assert result == 3


@pytest.mark.parametrize(
    "input_a, input_b, expected",
    [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
    ],
)
def test_sample_function_parametrized(input_a, input_b, expected):
    """Test the sample_function with multiple inputs using parametrize."""
    result = sample_function(input_a, input_b)
    assert result == expected


def test_safe_get():
    """Test the safe_get function."""
    data = {"a": 1, "b": 2}
    assert safe_get(data, "a") == 1
    assert safe_get(data, "c") is None
    assert safe_get(data, "c", "default") == "default"


def test_filter_list():
    """Test the filter_list function."""
    items = [1, 2, 3, 4, 5]

    # Filter even numbers
    result = filter_list(items, lambda x: x % 2 == 0)
    assert result == [2, 4]

    # Filter numbers greater than 3
    result = filter_list(items, lambda x: x > 3)
    assert result == [4, 5]
