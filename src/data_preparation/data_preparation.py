# from loguru import logger
import os


class DataPreparation:
    """
    Class responsible for data preparation and storage tasks:
    - Loading email data
    - Cleaning and preprocessing text
    - Extracting metadata (sender, recipient, date, etc.)
    - Storing processed data
    """

    def __init__(self, email_dir="./data/", output_dir="./processed_data/"):
        """
        Initialize the DataPreparation class.

        Args:
            email_dir (str): Directory containing the email files
            output_dir (str): Directory to store processed data
        """
        self.email_dir = email_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


if __name__ == "__main__":
    # Create a DataPreparation instance
    data_prep = DataPreparation()

    # Process emails
    # df = data_prep.process_emails()

    print("Done!")
