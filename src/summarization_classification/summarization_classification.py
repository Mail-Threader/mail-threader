from loguru import logger
import os

class SummarizationClassification:
    """
    Class responsible for summarization, classification, and categorization tasks:
    - Text preprocessing (tokenization, lemmatization)
    - Topic modeling
    - Email clustering
    - Entity recognition
    - Sentiment analysis
    - Text summarization
    """

    def __init__(self, input_dir="./processed_data/", output_dir="./analysis_results/"):
        """
        Initialize the SummarizationClassification class.

        Args:
            input_dir (str): Directory containing processed email data
            output_dir (str): Directory to store analysis results
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


if __name__ == "__main__":

    # Create SummarizationClassification instance
    analyzer = SummarizationClassification()

    # Analyze emails
    results = analyzer.analyze_emails()

    print("Done!")
