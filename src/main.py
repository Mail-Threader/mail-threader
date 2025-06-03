import argparse
import os

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
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation step (use existing processed data)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip summarization and classification (use existing analysis results)",
    )
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization step")
    parser.add_argument("--skip-stories", action="store_true", help="Skip story development step")

    return parser.parse_args()

def setup_directories(args):
    root = os.getcwd()

    processed_data_dir = os.path.join(root, "src", "data_preparation")  # point to where the file actually is
    analysis_results_dir = os.path.join(args.output_dir, "summarization_classification")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    stories_dir = os.path.join(args.output_dir, "stories")

    for directory in [analysis_results_dir, visualizations_dir, stories_dir]:
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

#def setup_directories(args):
    """
    Set up the directory structure for the pipeline.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        dict: Dictionary containing paths to all directories
    """
    # # Create timestamp for output directories
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    if skip:
        logger.info("Skipping data preparation step...")
        # Try to load existing processed data
        DataPreparation(email_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])
        # try:
        #     df = data_prep.load_data()
        #     logger.info(f"Loaded processed data with {len(df)} emails")
        #     return df
        # except FileNotFoundError:
        #     logger.warning("No processed data found. Running data preparation step...")

    logger.info("Running data preparation step...")
    DataPreparation(email_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])
    # df = data_prep.process_emails()
    # logger.info(f"Processed {len(df)} emails")
    # return df


def run_summarization_classification(_, dirs, skip=False, limit=None):
    """
    Run the summarization and classification step.

    Args:
        _ (pandas.DataFrame): Unused input to match interface.
        dirs (dict): Dictionary containing directory paths.
        skip (bool): Whether to skip this step.

    Returns:
        dict: Dictionary containing analysis results.
    """
    if skip:
        logger.info("Skipping summarization and classification step...")
        return

    logger.info("Running summarization and classification step...")

    analyzer = SummarizationClassification(
        input_dir=dirs["processed_data_dir"],
        output_dir=dirs["analysis_results_dir"]
    )

    df = analyzer.load_json_emails("clean_emails.json")
    if limit:
        df = df.head(limit)

    df = analyzer.clean_text_column(df)
    df = analyzer.tokenize_column(df, text_column="clean_body")
    df["clean_body"] = df["clean_body"].apply(lambda x: analyzer.preprocess_text(x))
    X, vectorizer = analyzer.vectorize_document(df["clean_body"])
    model, labels = analyzer.cluster_documents(X, method="kmeans", n_clusters=5)
    df["cluster"] = labels
    topics = analyzer.generate_cluster_topics(df["clean_body"], labels, X, vectorizer)
    print("Cluster Topics:", topics)
    top_words = analyzer.extract_top_words(X, vectorizer)
    print("Top overall words:", top_words)
    df = analyzer.extract_entities(df)
    df = analyzer.analyze_sentiment(df)
    summary = analyzer.summarize_corpus(df)
    print("Corpus Summary:\n", summary)
    #analyzer.save_to_csv(df, "email_summarization_results.csv")
    analyzer.save_to_sqlite(df)
    analyzer.save_to_json(df, "email_summarization_results.json")

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
    Visualization(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["visualizations_dir"],
    )
    # visualization_paths = visualizer.visualize_all(df, analysis_results)
    # logger.info(f"Generated {len(visualization_paths)} visualizations")
    # return visualization_paths

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
    StoryDevelopment(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["stories_dir"],
    )
    # stories = story_developer.develop_stories(df, analysis_results)
    # logger.info(f"Generated {len(stories)} stories")
    # return stories


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
    args = parse_arguments()
    dirs = setup_directories(args)

    # Run each step of the pipeline
    #df = run_data_preparation(dirs, args.skip_data_prep)
    analysis_results = run_summarization_classification(None, dirs, args.skip_analysis, limit=50)
    #visualization_paths = run_visualization(df, analysis_results, dirs, args.skip_visualization)
    #story_results = run_story_development(df, analysis_results, dirs, args.skip_stories)
    # Generate a final report
    #report_path = generate_report(dirs, df, analysis_results, visualization_paths, story_results)
    #logger.info(f"Pipeline completed. Final report available at {report_path}")

if __name__ == "_main_":
    main()
