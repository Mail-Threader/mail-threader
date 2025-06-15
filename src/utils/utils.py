"""
Utility functions for the Enron Email Analysis Pipeline.
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Union

import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine
from supabase import Client, create_client

# Load environment variables from .env file
load_dotenv()


def initialize_nltk(resources: Optional[List[str]] = None) -> None:
    """
    Initialize NLTK resources. Downloads required NLTK data if not already present.

    Args:
        resources (List[str], optional): List of NLTK resources to download.
            If None, downloads default resources: ['punkt', 'stopwords', 'wordnet'].

    Raises:
        Exception: If there's an error downloading NLTK resources.
    """
    if resources is None:
        resources = ["punkt", "stopwords", "wordnet"]

    try:
        for resource in resources:
            nltk.download(resource, quiet=True)
        logger.info(f"Successfully initialized NLTK resources: {', '.join(resources)}")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise Exception(f"Failed to initialize NLTK resources: {e}")


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


def load_processed_df(search_dir: str, search_file_name: str, db_table: str = None) -> pd.DataFrame:
    """

    Load the latest processed data file from the specified directory.

    Args:
        search_dir (str): Path to the directory containing the processed data files.
        search_file_name (str): File name to search for.
        db_table (str | None): None or Name of the database table to load data from.
    """
    pkl_files = [
        f for f in os.listdir(search_dir) if f.startswith(search_file_name) and f.endswith(".pkl")
    ]

    json_files = [
        f for f in os.listdir(search_dir) if f.startswith(search_file_name) and f.endswith(".json")
    ]

    if pkl_files:
        pkl_files.sort(reverse=True)
        file_path = os.path.join(search_dir, pkl_files[0])

        try:
            df = pd.read_pickle(file_path)
            logger.info(f"Loaded data from {file_path}: {len(df)} emails")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()

    if json_files:
        json_files.sort(reverse=True)
        file_path = os.path.join(search_dir, json_files[0])

        try:
            df = pd.read_json(file_path)
            logger.info(f"Loaded data from {file_path}: {len(df)} emails")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()

    if not pkl_files and not json_files:
        logger.warning(f"No processed data files found in {search_dir}")

        if db_table is None:
            logger.warning("No database table specified. Skipping database load.")
            return pd.DataFrame()

        logger.warning("Fetching data from database...")

        try:
            engine = create_engine(os.environ.get("DATABASE_URL"))
            df = pd.read_sql(f"SELECT * FROM {db_table}", engine)
            logger.info(f"Loaded data from database: {len(df)} emails")
            return df
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")

    return pd.DataFrame()


def calculate_influence_score(metrics: Dict[str, float]) -> float:
    """
    Calculate influence score from actor metrics.

    Args:
        metrics (Dict[str, float]): Dictionary containing actor metrics

    Returns:
        float: Calculated influence score
    """
    try:
        degree_centrality = metrics.get("degree_centrality", 0)
        betweenness_centrality = metrics.get("betweenness_centrality", 0)
        pagerank = metrics.get("pagerank", 0)
        return (degree_centrality + betweenness_centrality + pagerank) / 3
    except Exception as e:
        logger.error(f"Error calculating influence score: {e}")
        return 0.0


def calculate_communication_patterns(emails: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate communication patterns from email data.

    Args:
        emails (pd.DataFrame): DataFrame containing email data

    Returns:
        Dict[str, Any]: Dictionary containing communication patterns
    """
    try:
        daily_patterns = emails.groupby(emails["date"].dt.day_name()).size()
        busiest_day = daily_patterns.idxmax()
        busiest_day_count = int(daily_patterns.max())

        response_times = []
        for _, email in emails.iterrows():
            if pd.notna(email["date"]):
                replies = emails[
                    (emails["subject"].str.contains(email["subject"], na=False))
                    & (emails["date"] > email["date"])
                ]
                if not replies.empty:
                    response_time = (replies["date"].min() - email["date"]).total_seconds() / 3600
                    response_times.append(response_time)

        avg_response_time = np.mean(response_times) if response_times else None

        return {
            "busiest_day": busiest_day,
            "busiest_day_count": busiest_day_count,
            "avg_response_time": f"{avg_response_time:.1f} hours" if avg_response_time else "N/A",
        }
    except Exception as e:
        logger.error(f"Error calculating communication patterns: {e}")
        return {}


def calculate_event_metrics(event_emails: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate metrics for significant events.

    Args:
        event_emails (pd.DataFrame): DataFrame containing event emails

    Returns:
        Dict[str, Any]: Dictionary containing event metrics
    """
    try:
        participants = set()
        for _, email in event_emails.iterrows():
            if pd.notna(email["from"]):
                participants.update(re.findall(r"[\w\.-]+@[\w\.-]+", email["from"]))
            if pd.notna(email["to"]):
                participants.update(re.findall(r"[\w\.-]+@[\w\.-]+", email["to"]))

        return {
            "participant_count": len(participants),
            "avg_email_length": float(event_emails["body"].str.len().mean()),
            "reply_rate": float(
                len(event_emails[event_emails["subject"].str.contains("Re:", na=False)])
                / len(event_emails)
            ),
        }
    except Exception as e:
        logger.error(f"Error calculating event metrics: {e}")
        return {}


def calculate_thread_metrics(thread: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for email threads.

    Args:
        thread (List[Dict[str, Any]]): List of emails in thread

    Returns:
        Dict[str, Any]: Dictionary containing thread metrics
    """
    try:
        participants = set()
        for email in thread:
            if "from" in email and email["from"]:
                sender_emails = re.findall(r"[\w\.-]+@[\w\.-]+", email["from"])
                participants.update(sender_emails)

        thread_duration = None
        if thread[0]["date"] and thread[-1]["date"]:
            thread_duration = (thread[-1]["date"] - thread[0]["date"]).total_seconds() / 3600

        response_times = []
        for i in range(len(thread) - 1):
            if thread[i]["date"] and thread[i + 1]["date"]:
                response_time = (thread[i + 1]["date"] - thread[i]["date"]).total_seconds() / 3600
                response_times.append(response_time)

        avg_response_time = np.mean(response_times) if response_times else None

        return {
            "duration_hours": f"{thread_duration:.1f}" if thread_duration else "N/A",
            "avg_response_time": f"{avg_response_time:.1f} hours" if avg_response_time else "N/A",
            "participant_count": len(participants),
        }
    except Exception as e:
        logger.error(f"Error calculating thread metrics: {e}")
        return {}


def analyze_topic_trend(topic_counts: Dict[str, Dict[int, int]], topic_num: int) -> Dict[str, Any]:
    """
    Analyze the trend of a topic over time.

    Args:
        topic_counts (Dict[str, Dict[int, int]]): Dictionary containing topic counts over time
        topic_num (int): Topic number to analyze

    Returns:
        Dict[str, Any]: Dictionary containing trend analysis
    """
    try:
        if not topic_counts:
            return {"trend": "unknown", "peak_period": None, "peak_count": 0}

        topic_data = {period: counts.get(topic_num, 0) for period, counts in topic_counts.items()}

        if not topic_data:
            return {"trend": "unknown", "peak_period": None, "peak_count": 0}

        peak_period = max(topic_data.items(), key=lambda x: x[1])

        values = list(topic_data.values())
        if len(values) < 2:
            trend = "stable"
        else:
            slope = np.polyfit(range(len(values)), values, 1)[0]
            if slope > 0.1:
                trend = "increasing"
            elif slope < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"

        return {"trend": trend, "peak_period": peak_period[0], "peak_count": peak_period[1]}
    except Exception as e:
        logger.error(f"Error analyzing topic trend: {e}")
        return {"trend": "unknown", "peak_period": None, "peak_count": 0}
