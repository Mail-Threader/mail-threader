import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from utils import DatabaseManager, initialize_nltk

default_data_dir = os.path.join(os.getcwd(), "data")
default_output_dir = os.path.join(os.getcwd(), "output")


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

    # Add database initialization option
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset the database",
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


def run_data_preparation(dirs, db: DatabaseManager, skip=False, limit: int = None):
    """
    Run the data preparation step.

    Args:
        dirs (dict): Dictionary containing directory paths
        db (DatabaseManager): Database manager
        skip (bool): Whether to skip this step
        limit (int): Limit the number of emails to process. If None, process all emails.

    Returns:
        pandas.DataFrame: Processed email data
    """

    from data_preparation import DataPreparation

    data_prep = DataPreparation(input_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])
    if skip:
        logger.info("Skipping data preparation step...")
        try:
            df = data_prep.load_data(limit=limit)
            logger.info(f"Loaded processed data with {len(df)} emails")
            return df
        except FileNotFoundError:
            logger.warning("No processed data found. Running data preparation step...")

    logger.info("Running data preparation step...")
    df = data_prep.process_all_emails(limit=limit)
    data_prep.save_to_pickle(df)
    data_prep.save_to_json(df)
    db_res = data_prep.save_to_db(df, db)
    logger.info(f"Processed {len(df)} emails and saved to database: {db_res}")
    return df


def run_summarization_classification(df, dirs, skip=False, limit: int = None):
    """
    Run the summarization and classification step.

    Args:
        df (pandas.DataFrame): Processed email data.
        dirs (dict): Dictionary containing directory paths.
        skip (bool): Whether to skip this step.

    Returns:
        dict: Dictionary containing analysis results.
    """
    from summarization_classification import SummarizationClassification

    analyzer = SummarizationClassification(
        input_dir=dirs["processed_data_dir"],
        output_dir=dirs["analysis_results_dir"],
        skip=skip,
        save_to_db=True,
    )
    if skip:
        logger.info("Skipping summarization and classification step...")
        df = analyzer.load_data()
        return df

    logger.info("Running summarization and classification step...")
    df, res = analyzer.analyze_emails(df=df)
    analyzer.save_to_json(res, is_dataframe=False)
    analyzer.save_to_json(df)
    analyzer.save_to_pickle(df)
    logger.info(f"Processed {len(df)} emails")
    return df, res


def run_visualization(df, analysis_results, dirs, skip=False):
    """
    Run the visualization step.

    Args:
        df (pandas.DataFrame): Processed email data
        analysis_results (dict): Analysis results
        dirs (dict): Dictionary containing directory paths
        skip (bool): Whether to skip this step

    Returns:
        dict: Paths to generated visualizations
    """
    if skip:
        logger.info("Skipping visualization step...")
        return {}

    logger.info("Running visualization step...")
    from visualization import Visualization

    visualizer = Visualization(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["visualizations_dir"],
    )
    visualization_paths = visualizer.visualize_all(df, analysis_results)
    return visualization_paths


def run_story_development(df, analysis_results, dirs, skip=False):
    """
    Run the story development step.

    Args:
        df (pandas.DataFrame): Processed email data
        analysis_results (dict): Analysis results
        dirs (dict): Dictionary containing directory paths
        skip (bool): Whether to skip this step

    Returns:
        dict: Generated stories
    """
    if skip:
        logger.info("Skipping story development step...")
        return {}

    logger.info("Running story development step...")
    from story_development import StoryDevelopment

    story_developer = StoryDevelopment(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["stories_dir"],
    )
    stories = story_developer.develop_stories(df, analysis_results)
    logger.info(f"Generated {len(stories)} stories")
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

    # Initializing NLTK
    initialize_nltk()

    if args.reset_db:
        db.reset_db()

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

    df = run_data_preparation(dirs, skip=("data-prep" not in steps_to_run), limit=limit, db=db)
    processed_df = df
    df, analysis_results = run_summarization_classification(
        df, dirs, skip=("analysis" not in steps_to_run), limit=limit
    )
    visualization_paths = run_visualization(
        processed_df, analysis_results, dirs, skip=("vis" not in steps_to_run)
    )
    story_results = run_story_development(
        processed_df, analysis_results, dirs, skip=("story" not in steps_to_run)
    )


if __name__ == "__main__":
    main()
