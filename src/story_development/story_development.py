from loguru import logger
import os

class StoryDevelopment:
    """
    Class responsible for story development tasks:
    - Identifying key actors and their relationships
    - Tracking topics over time
    - Detecting significant events
    - Constructing narratives from email threads
    - Generating story summaries
    """

    def __init__(
        self,
        input_dir="./processed_data/",
        analysis_dir="./analysis_results/",
        output_dir="./stories/",
    ):
        """
        Initialize the StoryDevelopment class.

        Args:
            input_dir (str): Directory containing processed email data
            analysis_dir (str): Directory containing analysis results
            output_dir (str): Directory to store generated stories
        """
        self.input_dir = input_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


if __name__ == "__main__":
    # Create StoryDevelopment instance
    story_dev = StoryDevelopment()

    # Load data
    df = story_dev.load_data()

    # Load analysis results
    analysis_results = story_dev.load_analysis_results()

    # Develop stories
    results = story_dev.develop_stories(df, analysis_results)

    print("Done!")
