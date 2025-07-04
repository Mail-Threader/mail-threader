import argparse
import os
import time

from loguru import logger
from tqdm import tqdm
from data_preparation import DataPreparation
from story_development import StoryDevelopment
from summarization_classification import SummarizationClassification
from visualization import Visualization

default_data_dir = os.path.join(os.getcwd(), "data")
default_output_dir = os.path.join(os.getcwd(), "output")

LIMIT = 50

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enron Email Analysis Pipeline")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Directory containing the email data files")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Directory to store all output files")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip summarization and classification")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization step")
    parser.add_argument("--skip-stories", action="store_true", help="Skip story development step")
    return parser.parse_args()

def setup_directories(args):
    root = os.getcwd()
    processed_data_dir = os.path.join(root, "src", "data_preparation")
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

def run_data_preparation(dirs, skip=False):
    if skip:
        logger.info("Skipping data preparation step...")
        DataPreparation(email_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])
        return

    logger.info("Running data preparation step...")
    DataPreparation(email_dir=dirs["data_dir"], output_dir=dirs["processed_data_dir"])

def run_summarization_classification(_, dirs, skip=False, limit=None):
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
    logger.info(f"Cluster Topics: {topics}")
    top_words = analyzer.extract_top_words(X, vectorizer)
    logger.info(f"Top overall words: {top_words}")

    start = time.time()
    df = analyzer.extract_entities(df)
    end = time.time()
    logger.info(f"NER extraction took: {end - start:.2f} seconds")

    start = time.time()
    df = analyzer.analyze_sentiment(df)
    end = time.time()
    logger.info(f"Sentiment analysis took: {end - start:.2f} seconds")

    start = time.time()
    df = analyzer.summarize_abstractive(df)
    end = time.time()
    logger.info(f"Abstractive summarization took: {end - start:.2f} seconds")

    analyzer.save_to_json(df, os.path.join(dirs["analysis_results_dir"], "email_summarization_results.json"))
    return df

def run_visualization(df, analysis_results, dirs, skip=False):
    if skip:
        logger.info("Skipping visualization step...")
        return {}

    logger.info("Running visualization step...")
    Visualization(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["visualizations_dir"]
    )

def run_story_development(df, analysis_results, dirs, skip=False):
    if skip:
        logger.info("Skipping story development step...")
        return {}

    logger.info("Running story development step...")
    StoryDevelopment(
        input_dir=dirs["processed_data_dir"],
        analysis_dir=dirs["analysis_results_dir"],
        output_dir=dirs["stories_dir"]
    )

def generate_report(dirs, data_results, analysis_results, visualization_paths, story_results):
    logger.info("Generating final report...")
    report_path = os.path.join(dirs["output_dir"], "report.html")
    logger.info(f"Report generated at {report_path}")
    return report_path

def main():
    args = parse_arguments()
    dirs = setup_directories(args)

    df = run_data_preparation(dirs, args.skip_data_prep)
    analysis_results = run_summarization_classification(None, dirs, args.skip_analysis, limit=LIMIT)
    visualization_paths = run_visualization(df, analysis_results, dirs, args.skip_visualization)
    story_results = run_story_development(df, analysis_results, dirs, args.skip_stories)
    report_path = generate_report(dirs, df, analysis_results, visualization_paths, story_results)
    logger.info(f"Pipeline completed. Final report available at {report_path}")

if __name__ == "__main__":
    main()
