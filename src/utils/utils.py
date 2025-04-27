"""
Utility functions for the Enron Email Analysis Pipeline.
"""

from typing import Any, Callable, Dict, List


def sample_function(a: int, b: int) -> int:
    """
    A sample function that adds two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.

    Args:
        data: Dictionary to get value from
        key: Key to look up
        default: Default value to return if key is not found

    Returns:
        Value from dictionary or default if key not found
    """
    return data.get(key, default)


def filter_list(items: List[Any], condition: Callable) -> List[Any]:
    """
    Filter a list based on a condition function.

    Args:
        items: List of items to filter
        condition: Function that takes an item and returns True/False

    Returns:
        Filtered list containing only items where condition(item) is True
    """
    return [item for item in items if condition(item)]
