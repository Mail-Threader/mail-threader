import argparse
import os
import sys

import pandas
from loguru import logger

# Import the four main modules
from data_preparation import DataPreparation
from story_development import StoryDevelopment
from summarization_classification import SummarizationClassification
from visualization import Visualization

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


def run_data_preparation(dirs, skip=False):
    """
    Run the data preparation step.

    Args:
        dirs (dict): Dictionary containing directory paths
        skip (bool): Whether to skip this step

    Returns:
        pandas.DataFrame: Processed email data
    """

    # data_prep = DataPreparation(input_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"], skip=skip)
    data_prep = DataPreparation(input_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])
    if skip:
        logger.info("Skipping data preparation step...")
        try:
            df = data_prep.load_data()
            logger.info(f"Loaded processed data with {len(df)} emails")
            return df
        except FileNotFoundError:
            logger.warning("No processed data found. Running data preparation step...")

    logger.info("Running data preparation step...")
    df = data_prep.process_all_emails()
    # data_prep.save_to_pickle(df)
    # data_prep.save_to_json(df)
    logger.info(f"Processed {len(df)} emails")
    return df


def run_summarization_classification(df, dirs, skip=False):
    """
    Run the summarization and classification step.

    Args:
        df (pandas.DataFrame): Processed email data.
        dirs (dict): Dictionary containing directory paths.
        skip (bool): Whether to skip this step.

    Returns:
        dict: Dictionary containing analysis results.
    """

    analyzer = SummarizationClassification(
        input_dir=dirs["processed_data_dir"], output_dir=dirs["analysis_results_dir"], skip=skip
    )
    if skip:
        logger.info("Skipping summarization and classification step...")
        df = analyzer.load_data()
        return df

    logger.info("Running summarization and classification step...")
    df, res = analyzer.analyze_emails(df=df)
    analyzer.save_to_json(res, is_dataframe=False)
    # analyzer.save_to_json(df)
    analyzer.save_to_pickle(df)
    logger.info(f"Processed {len(df)} emails")
    return df


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
    story_developer = StoryDevelopment(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["stories_dir"],
    )
    stories = story_developer.develop_stories(df, analysis_results)
    logger.info(f"Generated {len(stories)} stories")
    return stories


def generate_report(dirs, data_results, analysis_results, visualization_paths, story_results):
    """
    Generate a final HTML report.

    Args:
        dirs (dict): Dictionary containing directory paths
        data_results (pandas.DataFrame): Processed email data
        analysis_results (dict): Analysis results
        visualization_paths (dict): Paths to generated visualizations
        story_results (dict): Generated stories

    Returns:
        str: Path to the generated report
    """
    logger.info("Generating final report...")
    report_path = os.path.join(dirs["output_dir"], "report.html")

    logger.info(f"Report generated at {report_path}")
    return report_path


def main():
    """
    Main function to run the entire pipeline.
    """
    try:
        args = parse_arguments()
        dirs = setup_directories(args)

        # Determine which steps to run based on skip/run arguments
        all_steps = ["data-prep", "analysis", "vis", "story"]
        steps_to_run = []

        if args.skip:
            # If skip is specified, run all steps except those in skip
            steps_to_run = [step for step in all_steps if step not in args.skip]
        elif args.run:
            # If run is specified, only run those steps
            steps_to_run = args.run
        else:
            # If neither skip nor run is specified, run all steps
            steps_to_run = all_steps

        if steps_to_run == []:
            logger.info("No steps to run. Exiting program...")
            sys.exit(0)

        logger.info(f"Running steps: {steps_to_run}")

        # Run each step of the pipeline based on steps_to_run
        df = None
        analysis_results = None
        visualization_paths = {}
        story_results = {}

        df = run_data_preparation(dirs, skip=("data-prep" not in steps_to_run))
        analysis_results = run_summarization_classification(
            df, dirs, skip=("analysis" not in steps_to_run)
        )
        visualization_paths = run_visualization(
            df, analysis_results, dirs, skip=("vis" not in steps_to_run)
        )
        story_results = run_story_development(
            df, analysis_results, dirs, skip=("story" not in steps_to_run)
        )

        # Generate a final report
        report_path = generate_report(
            dirs, df, analysis_results, visualization_paths, story_results
        )
        logger.info(f"Pipeline completed. Final report available at {report_path}")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        logger.info("Exiting program...")
        sys.exit(0)


if __name__ == "__main__":
    main()
