import json
import os
from datetime import datetime
from sys import getsizeof

import pandas as pd
from firebase_admin import credentials, firestore_async, initialize_app
from google.cloud.firestore import AsyncClient
from loguru import logger

from utils import save_error_log


class DatabaseManager:
	"""Database manager class to handle database initialization and operations."""

	def __init__(self):
		"""
        Initialize the database manager using environment variables.
        """
		# Check if firestore project config exists
		firestore_service_key = os.path.join(os.getcwd(), "serviceAccountKey.json")

		if not os.path.exists(firestore_service_key):
			raise FileNotFoundError(
				f"The service account key file {firestore_service_key} does not exist.")

		try:
			# Application Default credentials are automatically created.
			cred = credentials.Certificate(firestore_service_key)
			initialize_app(cred)

		except Exception as e:
			logger.error(f"Failed to initialize Firestore: {e}")
			save_error_log(f"Failed to initialize Firestore: {e}", )
			return None

	def get_db(self):
		"""
        Get the Firestore database instance.

        Returns:
                firestore_async.AsyncClient: Firestore database instance.
        """
		try:
			db = firestore_async.client()
			self.db = db
			logger.info("Firestore database instance created successfully.")
			return db
		except Exception as e:
			logger.error(f"Failed to get Firestore database instance: {e}")
			save_error_log(f"Failed to get Firestore database instance: {e}", )
			return None


async def save_df(
	db: AsyncClient,
	df: pd.DataFrame,
	result_type="processed_data",
	columns: list[str] = [],
):
	"""
    Save processed data to the Firestore database.

    Args:
            db (AsyncClient): Firestore database client.
            df (pd.DataFrame): DataFrame containing processed data.
            result_type (str): Type of result being saved (default: "processed_data") options are "processed_data", "analysis_results", "visualizations", "stories".
            columns (list[str]): List of columns to save. If empty, all columns will be saved.

    Returns:
            bool: True if data is saved successfully, False otherwise.
    """
	if not columns:
		logger.warning("No columns specified for saving. Saving all columns.")

	if df.empty:
		logger.warning("DataFrame is empty. No data to save.")
		return

	if not all(col in df.columns for col in columns):
		missing_cols = [col for col in columns if col not in df.columns]
		logger.error(f"Missing columns in DataFrame: {missing_cols}")
		raise ValueError(f"Missing columns: {missing_cols}")

	try:
		df_copy = df[columns].copy() if columns else df.copy()

		data = df_copy.to_dict(orient="records")

		output_col_ref = db.collection("output")
		result_type_doc_ref = output_col_ref.document(result_type)
		result_type_coll_ref = result_type_doc_ref.collection(
			f"{result_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

		results = {
			"success_count": 0,
			"failure_count": 0,
			"links": [],
		}

		batch = db.batch()

		for record in data:
			try:
				# Save each record as a document
				new_doc_ref = result_type_coll_ref.document()
				batch.set(new_doc_ref, record)
				results["success_count"] += 1
			except Exception as e:
				results["failure_count"] += 1
				logger.error(f"Failed to save record {record}: {e}")

		metadata = {
			"rows": len(data),
			"columns": len(df_copy.columns),
		}

		# Save metadata
		metadata_doc_ref = result_type_coll_ref.document("metadata")
		batch.set(metadata_doc_ref, metadata)

		logger.info(f"Saved {results['success_count']} records successfully, "
			f"{results['failure_count']} records failed.")

		return True

	except Exception as e:
		logger.error(f"Failed to save processed data: {e}")
		save_error_log(f"Failed to save processed data to Firestore: {e}", )
		return False
