"""
Utility functions for the Enron Email Analysis Pipeline.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine
from supabase import Client, create_client

# Load environment variables from .env file
load_dotenv()


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


def save_to_postgresql(
    df: pd.DataFrame,
    db_url: Optional[str] = None,
    table_name: str = "emails",
    if_exists: str = "replace",
    success_message: Optional[str] = None,
) -> None:
    """
    Save a DataFrame to a PostgreSQL database.

    Args:
        df: DataFrame to save
        db_url: Database connection URL (if None, will use environment variable DATABASE_URL)
        table_name: Name of the table to create/replace
        if_exists: How to behave if the table already exists ('fail', 'replace', or 'append')
        success_message: Custom success message (if None, will use a default message)
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Convert complex data types to strings for PostgreSQL compatibility
        for col in df_copy.columns:
            if df_copy[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df_copy[col] = df_copy[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict)) else x
                )

        # Get database URL from environment variables if not provided
        if db_url is None:
            # First try to get the complete DATABASE_URL

            logger.info("Trying to get database URL from environment variable DATABASE_URL")

            db_url = os.environ.get("DATABASE_URL")

            # If DATABASE_URL is not set, try to construct it from individual environment variables
            if not db_url:
                pg_host = os.environ.get("PG_HOST")
                pg_port = os.environ.get("PG_PORT", "5432")
                pg_user = os.environ.get("PG_USER")
                pg_password = os.environ.get("PG_PASSWORD")
                pg_database = os.environ.get("PG_DATABASE")

                # Check if all required parameters are available
                if pg_host and pg_user and pg_password and pg_database:
                    db_url = (
                        f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
                    )
                    logger.info(f"Constructed database URL from environment variables")
                else:
                    logger.error(
                        "No database URL provided and environment variables for connection are not properly set"
                    )
                    return

        # Create SQLAlchemy engine
        engine = create_engine(db_url)

        # Save DataFrame to PostgreSQL
        df_copy.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            method="multi",  # Use multi-row insert for better performance
            chunksize=1000,  # Insert in chunks to avoid memory issues
        )

        # Log success message
        if success_message:
            logger.info(success_message)
        else:
            logger.info(f"Saved {len(df)} rows to PostgreSQL table: {table_name}")
    except Exception as e:
        logger.error(f"Error saving to PostgreSQL: {e}")


def upload_to_supabase(
    file_path: str,
    bucket_name: str = "visualizations",
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    public_access: bool = True,
) -> Union[str, None]:
    """
    Upload a file to a Supabase storage bucket.

    Args:
        file_path: Path to the file to upload
        bucket_name: Name of the Supabase bucket to upload to
        supabase_url: Supabase URL (if None, will use environment variable SUPABASE_URL)
        supabase_key: Supabase service role key (if None, will use environment variable SUPABASE_SERVICE_KEY)
        public_access: Whether the file should be publicly accessible

    Returns:
        URL of the uploaded file if successful, None otherwise
    """
    try:
        # Get Supabase credentials from environment variables if not provided
        if supabase_url is None:
            supabase_url = os.environ.get("SUPABASE_URL")
            if not supabase_url:
                logger.error(
                    "No Supabase URL provided and SUPABASE_URL environment variable is not set"
                )
                return None

        if supabase_key is None:
            supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
            if not supabase_key:
                logger.error(
                    "No Supabase key provided and SUPABASE_SERVICE_KEY environment variable is not set"
                )
                return None

        # Initialize Supabase client
        supabase_client: Client = create_client(supabase_url, supabase_key)

        # Get the file name from the path
        file_name = os.path.basename(file_path)

        # Read the file
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Upload the file to the bucket
        response = supabase_client.storage.from_(bucket_name).upload(
            path=file_name, file=file_content, file_options={"content-type": "auto"}
        )

        # Get the public URL if public access is enabled
        if public_access:
            file_url = supabase_client.storage.from_(bucket_name).get_public_url(file_name)
            logger.info(f"Uploaded file to Supabase bucket '{bucket_name}': {file_url}")
            return file_url
        else:
            logger.info(f"Uploaded file to Supabase bucket '{bucket_name}': {file_name}")
            return file_name

    except Exception as e:
        logger.error(f"Error uploading file to Supabase: {e}")
        return None
