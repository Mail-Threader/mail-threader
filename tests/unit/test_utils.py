"""
Unit tests for the utils module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.utils import filter_list, safe_get, sample_function, save_to_postgresql


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


def test_save_to_postgresql_success():
    """Test the save_to_postgresql function with a successful database operation."""
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "data": [{"key": "value"}, [1, 2, 3], "string"],
        }
    )

    # Mock the create_engine function
    mock_engine = MagicMock()

    # Create a mock for the DataFrame copy with a to_sql method
    df_copy_mock = MagicMock()
    df_copy_mock.to_sql = MagicMock()

    with patch(
        "src.utils.utils.create_engine", return_value=mock_engine
    ) as mock_create_engine, patch("pandas.DataFrame.copy", return_value=df_copy_mock), patch(
        "src.utils.utils.logger"
    ) as mock_logger:
        # Call the function with a provided database URL
        save_to_postgresql(
            df, db_url="postgresql://user:password@localhost/testdb", table_name="test_table"
        )

        # Assert that create_engine was called with the correct URL
        mock_create_engine.assert_called_once_with("postgresql://user:password@localhost/testdb")

        # Assert that to_sql was called with the correct parameters
        df_copy_mock.to_sql.assert_called_once_with(
            "test_table",
            mock_engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

        # Assert that the success message was logged
        mock_logger.info.assert_called_once_with(
            f"Saved {len(df)} rows to PostgreSQL table: test_table"
        )


def test_save_to_postgresql_no_db_url():
    """Test the save_to_postgresql function when no database URL is provided."""
    # Create a sample DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Mock the environment variable
    with patch(
        "os.environ.get", return_value="postgresql://user:password@localhost/testdb"
    ) as mock_env_get, patch("src.utils.utils.create_engine") as mock_create_engine, patch(
        "pandas.DataFrame.copy", return_value=df.copy()
    ), patch(
        "pandas.DataFrame.to_sql"
    ), patch(
        "src.utils.utils.logger"
    ) as mock_logger:
        # Call the function without a database URL
        save_to_postgresql(df, db_url=None, table_name="test_table")

        # Assert that the environment variable was checked
        mock_env_get.assert_called_once_with("DATABASE_URL")

        # Assert that create_engine was called with the URL from the environment
        mock_create_engine.assert_called_once_with("postgresql://user:password@localhost/testdb")

        # Assert that the success message was logged
        # Note: We're checking for any_call instead of assert_called_once_with because
        # the function now logs multiple messages
        mock_logger.info.assert_any_call(f"Saved {len(df)} rows to PostgreSQL table: test_table")


def test_save_to_postgresql_no_db_url_no_env():
    """Test the save_to_postgresql function when no database URL is provided and no environment variables are set."""
    # Create a sample DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Mock the environment variable to return None for all variables
    def mock_environ_get(key, default=None):
        return default

    with patch("os.environ.get", side_effect=mock_environ_get) as mock_env_get, patch(
        "src.utils.utils.create_engine"
    ) as mock_create_engine, patch("pandas.DataFrame.copy", return_value=df.copy()), patch(
        "src.utils.utils.logger"
    ) as mock_logger:
        # Call the function without a database URL
        save_to_postgresql(df, db_url=None, table_name="test_table")

        # Assert that create_engine was not called
        mock_create_engine.assert_not_called()

        # Assert that the error message was logged
        mock_logger.error.assert_called_once_with(
            "No database URL provided and environment variables for connection are not properly set"
        )


def test_save_to_postgresql_with_individual_env_vars():
    """Test the save_to_postgresql function when using individual environment variables."""
    # Create a sample DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Mock the environment variables
    def mock_environ_get(key, default=None):
        env_vars = {
            "DATABASE_URL": None,  # Ensure DATABASE_URL is not set
            "PG_HOST": "localhost",
            "PG_PORT": "5432",
            "PG_USER": "testuser",
            "PG_PASSWORD": "testpass",
            "PG_DATABASE": "testdb",
        }
        return env_vars.get(key, default)

    with patch("os.environ.get", side_effect=mock_environ_get) as mock_env_get, patch(
        "src.utils.utils.create_engine"
    ) as mock_create_engine, patch("pandas.DataFrame.copy", return_value=df.copy()), patch(
        "pandas.DataFrame.to_sql"
    ), patch(
        "src.utils.utils.logger"
    ) as mock_logger:
        # Call the function without a database URL
        save_to_postgresql(df, db_url=None, table_name="test_table")

        # Assert that create_engine was called with the constructed URL
        expected_url = "postgresql://testuser:testpass@localhost:5432/testdb"
        mock_create_engine.assert_called_once_with(expected_url)

        # Assert that the info message about constructing URL was logged
        mock_logger.info.assert_any_call("Constructed database URL from environment variables")

        # Assert that the success message was logged
        mock_logger.info.assert_any_call(f"Saved {len(df)} rows to PostgreSQL table: test_table")


def test_save_to_postgresql_error_handling():
    """Test the error handling in the save_to_postgresql function."""
    # Create a sample DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Mock the create_engine function to raise an exception
    with patch(
        "src.utils.utils.create_engine", side_effect=Exception("Database connection error")
    ) as mock_create_engine, patch("pandas.DataFrame.copy", return_value=df.copy()), patch(
        "src.utils.utils.logger"
    ) as mock_logger:
        # Call the function with a provided database URL
        save_to_postgresql(
            df, db_url="postgresql://user:password@localhost/testdb", table_name="test_table"
        )

        # Assert that create_engine was called
        mock_create_engine.assert_called_once()

        # Assert that the error message was logged
        mock_logger.error.assert_called_once_with(
            "Error saving to PostgreSQL: Database connection error"
        )


def test_save_to_postgresql_custom_message():
    """Test the save_to_postgresql function with a custom success message."""
    # Create a sample DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Mock the necessary components
    with patch("src.utils.utils.create_engine"), patch(
        "pandas.DataFrame.copy", return_value=df.copy()
    ), patch("pandas.DataFrame.to_sql"), patch("src.utils.utils.logger") as mock_logger:
        # Call the function with a custom success message
        custom_message = "Custom success message for test"
        save_to_postgresql(
            df,
            db_url="postgresql://user:password@localhost/testdb",
            table_name="test_table",
            success_message=custom_message,
        )

        # Assert that the custom success message was logged
        mock_logger.info.assert_called_once_with(custom_message)


def test_save_to_postgresql_with_dotenv():
    """Test the save_to_postgresql function with environment variables loaded from .env file."""
    # Create a sample DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Mock the environment variables that would be loaded from .env
    env_vars = {
        "DATABASE_URL": "postgresql://dotenv_user:dotenv_password@dotenv_host/dotenv_db",
        "PG_HOST": "dotenv_host",
        "PG_PORT": "5432",
        "PG_USER": "dotenv_user",
        "PG_PASSWORD": "dotenv_password",
        "PG_DATABASE": "dotenv_db",
    }

    # Mock the necessary components
    with patch(
        "os.environ.get", side_effect=lambda key, default=None: env_vars.get(key, default)
    ), patch("src.utils.utils.create_engine") as mock_create_engine, patch(
        "pandas.DataFrame.copy", return_value=df.copy()
    ), patch(
        "pandas.DataFrame.to_sql"
    ), patch(
        "src.utils.utils.logger"
    ) as mock_logger:
        # Call the function without a database URL
        save_to_postgresql(df, db_url=None, table_name="test_table")

        # Assert that create_engine was called with the URL from the environment
        mock_create_engine.assert_called_once_with(
            "postgresql://dotenv_user:dotenv_password@dotenv_host/dotenv_db"
        )

        # Assert that the success message was logged
        mock_logger.info.assert_any_call(f"Saved {len(df)} rows to PostgreSQL table: test_table")
