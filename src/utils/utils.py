"""
Utility functions for the Enron Email Analysis Pipeline.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union

import nltk
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from supabase import Client, create_client


def initialize_nltk(resources: Optional[List[str]] = None) -> None:
	"""
    Initialize NLTK resources. Downloads required NLTK data if not already present.

    Args:
        resources (List[str], optional): List of NLTK resources to download.
            If None, downloads default resources: ['punkt', 'stopwords', 'wordnet'].

    Raises:
        Exception: If there's an error downloading NLTK resources.
    """
	try:
		# First, ensure NLTK data directory exists
		nltk_data_dir = os.path.expanduser("~/nltk_data")

		logger.info(f"NLTK data directory: {nltk_data_dir}")

		if not os.path.exists(nltk_data_dir):
			os.makedirs(nltk_data_dir)

		# Download required NLTK resources
		required_packages = resources or [
			"punkt",
			"stopwords",
			"wordnet",
			"omw-1.4",
			"punkt_tab",
			"averaged_perceptron_tagger_eng",
		]

		for package in required_packages:
			try:
				nltk.download(package, download_dir=nltk_data_dir, quiet=True)
			except Exception as e:
				logger.error(f"Error downloading NLTK package {package}: {e}")

	except Exception as e:
		save_error_log(f"Error initializing NLTK: {e}")


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
					"No Supabase URL provided and SUPABASE_URL environment variable is not set")
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
		response = supabase_client.storage.from_(bucket_name).upload(path=file_name,
			file=file_content,
			file_options={"content-type": "auto"})

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


def load_processed_df(
	search_dir: str,
	search_file_name: str,
	db_table: Optional[str] = None,
	limit: Optional[int] = None,
) -> pd.DataFrame | None:
	"""

    Load the latest processed data file from the specified directory.

    Args:
        search_dir (str): Path to the directory containing the processed data files.
        search_file_name (str): File name to search for.
        db_table (str | None): None or Name of the database table to load data from.
    """
	try:
		pkl_files = [
			f for f in os.listdir(search_dir)
			if f.startswith(search_file_name) and f.endswith(".pkl")
		]

		json_files = [
			f for f in os.listdir(search_dir)
			if f.startswith(search_file_name) and f.endswith(".json")
		]

		if pkl_files:
			pkl_files.sort(reverse=True)
			file_path = os.path.join(search_dir, pkl_files[0])

			try:
				df = pd.read_pickle(file_path)
				if limit is not None:
					df = df.head(limit)
				logger.info(f"Loaded data from {file_path}: {len(df)} emails")
				return df
			except Exception as e:
				logger.error(f"Error loading data from {file_path}: {e}")
				save_error_log(f"Error loading data from {file_path}: {e}", )
				return None

		if json_files:
			json_files.sort(reverse=True)
			file_path = os.path.join(search_dir, json_files[0])

			try:
				df = pd.read_json(file_path)
				if limit is not None:
					df = df.head(limit)
				logger.info(f"Loaded data from {file_path}: {len(df)} emails")
				return df
			except Exception as e:
				logger.error(f"Error loading data from {file_path}: {e}")
				save_error_log(f"Error loading data from {file_path}: {e}", )
				return None

		if not pkl_files and not json_files:
			logger.warning(f"No processed data files found in {search_dir}")

			if db_table is None:
				logger.warning("No database table specified. Skipping database load.")
				save_error_log("No processed data files found and no database table specified.", )
				return None

			logger.warning("Fetching data from database...")

			try:
				engine = create_engine(os.environ.get("DATABASE_URL") or "")
				df = pd.read_sql(f"SELECT * FROM {db_table}", engine)
				logger.info(f"Loaded data from database: {len(df)} emails")
				return df
			except Exception as e:
				logger.error(f"Error loading data from database: {e}")

		return None
	except Exception as e:
		logger.error(f"Error loading processed data: {e}")
		save_error_log(f"Error loading processed data: {e}")
		return None


def save_error_log(
	error_message: str | int | Exception | None | tuple | bool | list | dict,
	error_dir: str = "./error_logs/",
	file_name: Optional[str] = None,
) -> None:
	"""
    Save an error message to a log file.

    Args:
        error_message (str | int | Exception | None | tuple | bool | list | dict): The error message to log.
        error_dir (str): Directory to save the error log file.
        file_name (str, optional): Name of the log file. If None, uses a timestamped name.
    """
	if not os.path.exists(error_dir):
		os.makedirs(error_dir)

	curr_timestamp = pd.Timestamp.now()

	if file_name is None:
		# search for most recent log file within 1 hour
		recent_files = [f for f in os.listdir(error_dir) if f.endswith(".log")]
		if recent_files:
			recent_files.sort(reverse=True)
			temp_file_name = recent_files[0]

			file_timestamp = pd.Timestamp(temp_file_name.split(".")[0])
			if (curr_timestamp - file_timestamp).total_seconds() < 3600:
				# If the most recent log file is within the last hour, use it
				file_name = recent_files[0]

			if file_name is None:
				# If no recent log file found, create a new one with current timestamp
				file_name = f"{pd.Timestamp.now()}.log"
		else:
			# If no recent log file found, create a new one with current timestamp
			file_name = f"{pd.Timestamp.now()}.log"

	file_path = os.path.join(error_dir, file_name)
	with open(file_path, "a") as f:
		f.write(f"{pd.Timestamp.now()}: {error_message}\n")


custom_stop_words = set([
	"enron",
	"ect",
	"corp",
	"com",
	"recipient",
	"subject",
	"email",
	"message",
	"cc",
	"to",
	"from",
	"sent",
	"pm",
	"am",
	"forwarded",
	"original",
	"attached",
	"http",
	"https",
	"www",
	"energy",
	"deal",
	"trading",
	"enron.com",
	"enron.net",
	"enronxgate",
	"e-mail",
	"mail",
	"contact",
	"address",
	"phone",
	"fax",
	"please",
	"thanks",
	"regards",
	"attached",
	"forward",
	"re",
	"fw",
	"fwd",
])


def sort_emails_by_date(df: pd.DataFrame):
	"""
        Sort emails by date in ascending order.

        Args:
            df (pd.DataFrame): DataFrame containing email data.

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
	logger.info("Sorting emails by date...")

	date_formats = [
		"%d/%m/%Y %H:%M:%S",
		"%Y-%m-%d %H:%M:%S",
		"%m/%d/%Y %H:%M:%S",
		"%Y/%m/%d %H:%M:%S",
	]

	for date_format in date_formats:
		try:
			df["date"] = pd.to_datetime(
				df["date"],
				format=date_format,
			)
			# If we successfully parsed any dates, break the loop
			if not df["date"].isna().all():
				break
		except Exception:
			continue

	df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

	return df
