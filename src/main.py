import argparse
import os
from typing import Optional
import time
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from data_preparation import DataPreparation
from database import DatabaseManager
from story_development import StoryDevelopment
from summarization_classification import Summarization
from utils import initialize_nltk
from visualization import Visualization

default_data_dir = os.path.join(os.getcwd(), "data")
default_output_dir = os.path.join(os.getcwd(), "output")
default_error_dir = os.path.join(os.getcwd(), "error_logs")

LIMIT = 50


def parse_arguments():
	"""
	Parse command line arguments.

	Returns:
			argparse.Namespace: Parsed arguments
	"""
	parser = argparse.ArgumentParser(description="Enron Email Analysis Pipeline")

	# Add arguments for each step of the pipeline
	parser.add_argument(
		"--data-dir",
		type=str,
		default=default_data_dir,
		help="Directory containing the email data files",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=default_output_dir,
		help="Directory to store all output files",
	)
	parser.add_argument(
		"--error-dir",
		type=str,
		default=default_error_dir,
		help="Directory to store error logs",
	)

	# Add mutually exclusive group for skip/run options
	skip_run_group = parser.add_mutually_exclusive_group()
	skip_run_group.add_argument(
		"--skip",
		choices=["data-prep", "analysis", "vis", "story"],
		nargs="+",
		help="Steps to skip (can specify multiple steps)",
	)
	skip_run_group.add_argument(
		"--run",
		choices=["data-prep", "analysis", "vis", "story"],
		nargs="+",
		help="Steps to run (can specify multiple steps)",
	)

	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit the number of emails to process",
	)

	return parser.parse_args()


def setup_directories(args):
	"""
	Set up the directory structure for the pipeline.

	Args:
			args (argparse.Namespace): Parsed command line arguments

	Returns:
			dict: Dictionary containing paths to all directories
	"""

	# Create the main output directory
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Create the error directory if it doesn't exist
	if not os.path.exists(args.error_dir):
		os.makedirs(args.error_dir)

	# Create subdirectories for each step
	processed_data_dir = os.path.join(args.output_dir, "processed_data")
	analysis_results_dir = os.path.join(args.output_dir, "analysis_results")
	visualizations_dir = os.path.join(args.output_dir, "visualizations")
	stories_dir = os.path.join(args.output_dir, "stories")

	# Create directories if they don't exist
	for directory in [
		processed_data_dir,
		analysis_results_dir,
		visualizations_dir,
		stories_dir,
	]:
		if not os.path.exists(directory):
			os.makedirs(directory)

	return {
		"data_dir": args.data_dir,
		"output_dir": args.output_dir,
		"processed_data_dir": processed_data_dir,
		"analysis_results_dir": analysis_results_dir,
		"visualizations_dir": visualizations_dir,
		"stories_dir": stories_dir,
	}


def run_data_preparation(
	dirs: dict,
	db: DatabaseManager,
	skip=False,
	limit: Optional[int] = None,
):
	"""
	Run the data preparation step.

	Args:
			dirs (dict): Dictionary containing directory paths
			db (DatabaseManager): Database manager instance
			skip (bool): Whether to skip this step
			limit (int): Limit the number of emails to process. If None, process all emails.

	Returns:
			pandas.DataFrame: Processed email data
	"""
	data_prep = DataPreparation(input_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])

	if skip:
		logger.info("Skipping data preparation step...")
		return None

	logger.info("Running data preparation step...")
	df = data_prep.process_all_emails(limit=limit)
	data_prep.save_to_pickle(df)
	logger.info(f"Processed {len(df)} emails")
	data_prep.save_to_database(df, db)
	return df


def run_summarization_classification(
	processed_email_data: Optional[pd.DataFrame],
	dirs: dict,
	db: DatabaseManager,
	skip=False,
	limit: Optional[int] = None,
):
	"""
	Run the summarization and classification step.

	Args:
			processed_email_data (pandas.DataFrame): Processed email data or None
			dirs (dict): Dictionary containing directory paths.
			db (DatabaseManager): Database manager instance
			skip (bool): Whether to skip this step.
			limit (int): Limit the number of emails to process. If None, process all emails.

	Returns:
			dict: Dictionary containing analysis results.
	"""

	analyzer = Summarization(
		input_dir=dirs["processed_data_dir"],
		output_dir=dirs["analysis_results_dir"],
		skip=skip,
	)

	if skip:
		logger.info("Skipping summarization and classification step...")
		return None

	logger.info("Running summarization and classification step...")
	analysis_df = analyzer.analyze_emails(processed_emails_df=processed_email_data, limit=limit)

	if analysis_df is None or analysis_df.empty:
		logger.error("No analysis results found. Exiting summarization step.")
		return None

	analyzer.save_to_pkl(analysis_df)

	return analysis_df


def run_visualization(processed_data: Optional[pd.DataFrame],
	analysis_results: Optional[pd.DataFrame],
	dirs: dict,
	db: DatabaseManager,
	skip=False):
	"""
	Run the visualization step.

	Args:
		df (pandas.DataFrame): Processed email data or None
		analysis_results (pandas.DataFrame): Analysis results or None
		dirs (dict): Dictionary containing directory paths
		db (DatabaseManager): Database manager instance
		skip (bool): Whether to skip this step

	Returns:
			dict: Paths to generated visualizations
	"""
	if skip:
		logger.info("Skipping visualization step...")
		return {}

	logger.info("Running visualization step...")

	visualizer = Visualization(
		input_dir=dirs["processed_data_dir"],
		analysis_dir=dirs["analysis_results_dir"],
		output_dir=dirs["visualizations_dir"],
	)
	visualization_paths = visualizer.visualize_all(processed_data, analysis_results)
	return visualization_paths


def run_story_development(
	processed_data: Optional[pd.DataFrame],
	analysis_results: Optional[pd.DataFrame],
	dirs: dict,
	db: DatabaseManager,
	skip=False,
	limit: Optional[int] = None,
):
	"""
	Run the story development step.

	Args:
		analysis_results (pd.DataFrame|None): Analysis results
		dirs (dict): Dictionary containing directory paths
		db (DatabaseManager): Database manager instance
		skip (bool): Whether to skip this step
		limit (int): Limit the number of emails to process. If None, process all emails.

	Returns:
		dict: Generated stories
	"""
	if skip:
		logger.info("Skipping story development step...")
		return None

	logger.info("Running story development step...")

	story_developer = StoryDevelopment(
		processed_data_dir=dirs["processed_data_dir"],
		analysis_results_dir=dirs["analysis_results_dir"],
		output_dir=dirs["stories_dir"],
	)
	stories = story_developer.develop_story(processed_data,
		analysis_results,
		limit=limit,
		db_manager=db)
	return stories


def main():
	"""
	Main function to run the email analysis pipeline.
	"""
	# Parse command line arguments
	args = parse_arguments()

	# Load environment variables from .env file
	load_dotenv()

	# Initialize database
	logger.info("Initializing database...")
	db = DatabaseManager()
	db.connect()
	db.create_tables()

	# Initializing NLTK
	initialize_nltk()

	# Set up the directory structure
	dirs = setup_directories(args)

	# Determine which steps to run
	steps_to_run = ["data-prep", "analysis", "vis", "story"]
	if args.run:
		steps_to_run = args.run
	elif args.skip:
		steps_to_run = [step for step in steps_to_run if step not in args.skip]

	logger.info(f"Running steps: {steps_to_run}")

	limit = args.limit

	processed_df = run_data_preparation(dirs,
		skip=("data-prep" not in steps_to_run),
		limit=limit,
		db=db)
	analysis_results = run_summarization_classification(processed_df,
		dirs,
		skip=("analysis" not in steps_to_run),
		limit=limit,
		db=db)
	visualization_paths = run_visualization(processed_df,
		analysis_results,
		dirs,
		skip=("vis" not in steps_to_run),
		db=db)
	story_results = run_story_development(
		processed_data=processed_df,
		analysis_results=analysis_results,
		dirs=dirs,
		skip=("story" not in steps_to_run),
		limit=limit,
		db=db,
	)


if __name__ == "__main__":
	main()
