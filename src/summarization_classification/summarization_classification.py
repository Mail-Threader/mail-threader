import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
import pandas as pd
import spacy
from loguru import logger
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from transformers.pipelines import pipeline

from src.utils.utils import save_to_postgresql

# Download necessary NLTK resources
try:
    logger.info("Downloading NLTK resources")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")


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
            input_dir: Directory containing processed email data
            output_dir: Directory to store analysis results
        """
        # Load NLP models
        try:
            self.ner_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy NER model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.ner_model = spacy.load("en_core_web_sm")

        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            logger.info("Initialized sentiment analysis pipeline")
        except Exception as e:
            logger.error(f"Error initializing sentiment pipeline: {e}")
            self.sentiment_pipeline = None

        # Set directories
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Create an output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

    def load_data(self, skip: bool = False, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads email data from a JSON or pickle file.

        Args:
            skip: Whether to skip loading data from the input directory (default is False).
            file_path: Full path to the file to load. If None, finds the most recent file.

        Returns:
            DataFrame of emails or empty DataFrame if no data is found
        """
        # If file_path is provided, use it directly
        if file_path is not None:
            return self._load_file(file_path)

        # Otherwise, find the most recent file
        search_dir = self.output_dir if skip else self.input_dir
        search_file_pattern = "analysis_results_" if skip else "processed_data_"

        # Find all matching files
        try:
            all_files = os.listdir(search_dir)
            pkl_files = [
                f for f in all_files if f.startswith(search_file_pattern) and f.endswith(".pkl")
            ]
            json_files = [
                f for f in all_files if f.startswith(search_file_pattern) and f.endswith(".json")
            ]

            logger.info(
                f"Found {len(pkl_files)} pkl and {len(json_files)} json files in {search_dir}"
            )

            # No files found
            if not pkl_files and not json_files:
                logger.error(f"No data files found in {search_dir}")
                return pd.DataFrame()

            # Prefer pickle files (faster loading)
            if pkl_files:
                pkl_files.sort(reverse=True)  # Sort by timestamp (newest first)
                return self._load_file(os.path.join(search_dir, pkl_files[0]))

            # Fall back to JSON files
            json_files.sort(reverse=True)
            return self._load_file(os.path.join(search_dir, json_files[0]))

        except Exception as e:
            logger.error(f"Error finding data files: {e}")
            return pd.DataFrame()

    @staticmethod
    def _load_file(file_path: str) -> pd.DataFrame:
        """
        Helper method to load a file based on its extension.

        Args:
            file_path: Path to the file to load

        Returns:
            DataFrame with the loaded data or empty DataFrame on error
        """
        try:
            if file_path.endswith(".pkl"):
                df = pd.read_pickle(file_path)
                logger.info(f"Loaded pickle data from {file_path}: {len(df)} records")
                return df
            elif file_path.endswith(".json"):
                df = pd.read_json(file_path, orient="records")
                logger.info(f"Loaded JSON data from {file_path}: {len(df)} records")
                return df
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def clean_text_column(
        df: pd.DataFrame, column: str = "body", new_column: str = "clean_body"
    ) -> pd.DataFrame:
        """
        Clean a text column by removing excessive whitespace and normalizing text.

        Args:
            df: Input DataFrame
            column: Name of the column containing text to clean
            new_column: Name of the new column to store cleaned text

        Returns:
            DataFrame with an additional column containing cleaned text
        """
        df = df.copy()
        # Fill NA values and normalize whitespace
        df[new_column] = df[column].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
        return df

    @staticmethod
    def tokenize_column(
        df: pd.DataFrame, text_column: str = "body", new_column: str = "tokens"
    ) -> pd.DataFrame:
        """
        Tokenizes the specified text column into a list of word tokens.

        Args:
            df: Input DataFrame
            text_column: Column in the DataFrame that contains text to tokenize
            new_column: Name of the column where tokens will be stored

        Returns:
            DataFrame with an additional column containing tokens
        """
        df = df.copy()
        df[new_column] = df[text_column].fillna("").apply(word_tokenize)
        return df

    @staticmethod
    def vectorize_document(
        documents: Union[List[str], pd.Series], max_features: int = 5000
    ) -> Tuple[Any, TfidfVectorizer]:
        """
        Vectorizes a list of text documents using TF-IDF.

        Args:
            documents: List or Series of strings (documents)
            max_features: Max number of features to keep in the vocabulary

        Returns:
            A tuple containing:
                - tfidf_matrix: Sparse matrix of TF-IDF features
                - vectorizer: The fitted TfidfVectorizer instance
        """
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix, vectorizer

    @staticmethod
    def create_vectorizer_model_pipeline(num_clusters: int = 5) -> Pipeline:
        """
        Creates a pipeline for TF-IDF vectorization followed by KMeans clustering.

        Args:
            num_clusters: Number of clusters for KMeans

        Returns:
            A scikit-learn Pipeline object
        """
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(stop_words="english", max_features=5000, min_df=2, max_df=0.95),
                ),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=num_clusters,
                        random_state=42,
                        n_init=10,
                        max_iter=300,
                    ),
                ),
            ]
        )
        return pipeline

    @staticmethod
    def generate_cluster_topics(
        labels: np.ndarray,
        tfidf_matrix: Any,
        vectorizer: TfidfVectorizer,
        top_n: int = 6,
    ) -> Dict[int, List[str]]:
        """
        Generates top keywords per cluster/topic.

        Args:
            labels: Cluster labels assigned to each document
            tfidf_matrix: TF-IDF feature matrix
            vectorizer: Fitted TfidfVectorizer
            top_n: Number of top words to return per cluster

        Returns:
            Dictionary of cluster_id -> a list of top keywords
        """
        # Get feature names from vectorizer
        terms = np.array(vectorizer.get_feature_names_out())
        topics: Dict[int, List[str]] = {}

        # For each cluster, find the top terms
        for cluster_id in sorted(set(labels)):
            # Create a mask for documents in this cluster
            mask = np.array(labels) == cluster_id

            # Skip empty clusters
            if not np.any(mask):
                topics[cluster_id] = []
                continue

            # Calculate mean TF-IDF scores for this cluster
            cluster_matrix = tfidf_matrix[mask].mean(axis=0)

            # Get indices of top terms
            top_indices = cluster_matrix.A1.argsort()[::-1][:top_n]

            # Store top terms for this cluster
            topics[cluster_id] = terms[top_indices].tolist()

        return topics

    @staticmethod
    def extract_top_words(
        tfidf_matrix: Any, vectorizer: TfidfVectorizer, top_n: int = 10
    ) -> List[str]:
        """
        Extracts top TF-IDF words across all documents.

        Args:
            tfidf_matrix: TF-IDF matrix
            vectorizer: Fitted TfidfVectorizer instance
            top_n: Number of top terms to return

        Returns:
            List of top words with the highest average TF-IDF scores
        """
        # Calculate a mean TF-IDF score for each term across all documents
        mean_scores = tfidf_matrix.mean(axis=0).A1

        # Get indices of top terms
        top_indices = mean_scores.argsort()[::-1][:top_n]

        # Return top terms
        return vectorizer.get_feature_names_out()[top_indices].tolist()

    @staticmethod
    def cluster_documents(
        tfidf_matrix: Any, method: str = "kmeans", **kwargs
    ) -> Tuple[Union[KMeans, DBSCAN], np.ndarray]:
        """
        Clusters documents using the specified method (KMeans or DBSCAN).

        Args:
            tfidf_matrix: TF-IDF matrix
            method: Clustering method ('kmeans' or 'dbscan')
            kwargs: Additional arguments for the clustering model

        Returns:
            Tuple containing:
                - Fitted clustering model
                - Cluster labels for each document
        """
        if method.lower() == "kmeans":
            model = KMeans(
                n_clusters=kwargs.get("n_clusters", 5),
                random_state=42,
                n_init=10,
                max_iter=300,
            )
        elif method.lower() == "dbscan":
            model = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5),
                metric=kwargs.get("metric", "cosine"),
            )
        else:
            raise ValueError("Unsupported clustering method. Choose 'kmeans' or 'dbscan'.")

        # Fit the model and get cluster labels
        labels = model.fit_predict(tfidf_matrix)

        return model, labels

    def extract_entities(self, df: pd.DataFrame, text_column: str = "body") -> pd.DataFrame:
        """
        Extract named entities (PERSON, ORG, GPE) from a text column using spaCy.

        Args:
            df: DataFrame containing text data
            text_column: Column with the text to process

        Returns:
            DataFrame with additional columns for persons, organizations, and locations
        """
        if self.ner_model is None:
            logger.error("NER model not initialized")
            return df.copy()

        df = df.copy()

        def ner_extract(text: str) -> pd.Series:
            """Extract named entities from text using spaCy"""
            try:
                # Process text with spaCy
                doc = self.ner_model(text[:100000])  # Limit text length to avoid memory issues

                # Extract entities by type
                persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

                # Remove duplicates
                persons = list(dict.fromkeys(persons))
                orgs = list(dict.fromkeys(orgs))
                locations = list(dict.fromkeys(locations))

                return pd.Series([persons, orgs, locations])
            except Exception as e:
                logger.error(f"Error in entity extraction: {e}")
                return pd.Series([[], [], []])

        # Apply NER extraction to each row
        df[["persons", "organizations", "locations"]] = (
            df[text_column].fillna("").apply(ner_extract)
        )

        return df

    def analyze_sentiment(self, df: pd.DataFrame, text_column: str = "body") -> pd.DataFrame:
        """
        Performs sentiment analysis using HuggingFace Transformers pipeline.

        Args:
            df: Input DataFrame with text
            text_column: Column to analyze

        Returns:
            DataFrame with a new 'sentiment' column
        """
        df = df.copy()

        # Check if the sentiment pipeline is available
        if self.sentiment_pipeline is None:
            logger.warning("Sentiment pipeline not initialized. Skipping sentiment analysis.")
            df["sentiment"] = "NEUTRAL"
            return df

        def get_sentiment(text: str) -> str:
            """Extract sentiment from text using transformers pipeline"""
            if not text or len(text.strip()) == 0:
                return "NEUTRAL"

            try:
                # Use the sentiment pipeline to analyze the text
                if self.sentiment_pipeline is None:
                    return "NEUTRAL"
                # Limit text length to avoid memory issues and comply with model limits
                result = self.sentiment_pipeline(text[:512])
                if not result:
                    return "NEUTRAL"
                # result can be of type list[Unknown] | list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Unknown | Any
                if isinstance(result, list) and len(result) > 0:
                    # Extract the label from the first result
                    result = result[0]
                elif isinstance(result, dict):
                    # If result is a dictionary, extract the label directly
                    result = result
                else:
                    # If result is not a list or dict, return NEUTRAL
                    return "NEUTRAL"
                return result["label"]
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                return "NEUTRAL"

        # Apply sentiment analysis to each row
        df["sentiment"] = df[text_column].fillna("").apply(get_sentiment)

        return df

    @staticmethod
    def summarize_corpus(
        df: pd.DataFrame,
        text_column: str = "body",
        max_sentences: int = 5,
        max_input_sentences: int = 1000,
    ) -> str:
        """
        Generates an extractive summary from the full corpus using TF-IDF and cosine similarity.
        Limits the number of input sentences to avoid memory issues.

        Args:
            df: Input DataFrame
            text_column: Column containing text
            max_sentences: Max number of summary sentences to return
            max_input_sentences: Max number of input sentences to consider

        Returns:
            A summary string
        """
        # Check if DataFrame is empty
        if df.empty or text_column not in df.columns:
            return ""

        # Join all text into one string
        full_text = " ".join(df[text_column].dropna())
        if not full_text.strip():
            return ""

        # Split into sentences
        sentences = sent_tokenize(full_text)

        # Limit the number of sentences to avoid memory issues
        if len(sentences) > max_input_sentences:
            sentences = sentences[:max_input_sentences]

        # If we have fewer sentences than the requested summary length, return all sentences
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        try:
            # Vectorize sentences
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(sentences)

            # Calculate similarity matrix
            sim_matrix = cosine_similarity(tfidf_matrix)

            # Score sentences by their centrality (sum of similarities)
            scores = sim_matrix.sum(axis=1)

            # Get indices of top sentences
            top_indices = np.argsort(scores)[-max_sentences:]

            # Sort indices to maintain original order
            top_indices.sort()

            # Join top sentences into summary
            summary = " ".join([sentences[i] for i in top_indices])
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fall back to the first few sentences if summarization fails
            return " ".join(sentences[:max_sentences])

    def save_to_csv(self, df: pd.DataFrame):
        """
        Save the DataFrame to a CSV file in the output directory.

        Args:
            df: DataFrame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            output_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved data to CSV: {output_path}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

    def save_to_json(self, df: pd.DataFrame) -> None:
        """
        Save the DataFrame to a JSON file in the output directory.

        Args:
            df: DataFrame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            output_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}.json")
            df.to_json(output_path, orient="records", indent=4)
            logger.info(f"Saved data to JSON: {output_path}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")

    def save_to_sqlite(
        self,
        df: pd.DataFrame,
        db_filename: str = "analysis_results.db",
        table_name: str = "emails",
    ) -> None:
        """
        Save the DataFrame to an SQLite database in the output directory.

        Args:
            df: DataFrame to save
            db_filename: Name of the database file
            table_name: Name of the table to create/replace
        """
        try:
            # Create a full path to a database file
            db_path = os.path.join(self.output_dir, db_filename)

            # Make a copy to avoid modifying the original DataFrame
            df_copy = df.copy()

            # Convert list columns to comma-separated strings for SQLite compatibility
            for col in df_copy.columns:
                if df_copy[col].apply(lambda x: isinstance(x, list)).any():
                    df_copy[col] = df_copy[col].apply(
                        lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
                    )

            # Connect to the database and save data
            conn = sqlite3.connect(db_path)
            df_copy.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()

            logger.info(f"Saved data to SQLite database: {db_path}, table: {table_name}")
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")

    def save_to_pickle(self, df: pd.DataFrame):
        """
        Save the DataFrame to a pickle file in the output directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}.pkl")
        df.to_pickle(output_path)
        logger.success(f"\n✅ Saved {len(df)} cleaned emails to {output_path}")

    @staticmethod
    def save_to_postgresql(
        df: pd.DataFrame,
        db_url: Optional[str] = None,
        table_name: str = "summarized_emails",
        if_exists: str = "replace",
    ) -> None:
        """
        Save the DataFrame to a PostgreSQL database in Neon.

        Args:
            df: DataFrame to save
            db_url: Database connection URL (if None, will use environment variable DATABASE_URL)
            table_name: Name of the table to create/replace
            if_exists: How to behave if the table already exists ('fail', 'replace', or 'append')
        """
        success_message = f"Saved data to PostgreSQL table: {table_name}"
        save_to_postgresql(df, db_url, table_name, if_exists, success_message)

    def process_data(
        self,
        df: Optional[pd.DataFrame] = None,
        save_to_db: bool = True,
        db_url: Optional[str] = None,
    ):
        """
        Process the email data through the full pipeline:
        1. Clean text
        2. Tokenize text
        3. Vectorize documents
        4. Cluster documents
        5. Extract entities
        6. Analyze sentiment
        7. Generate summary
        8. Save to the database (if save_to_db is True)

        Args:
            df: Input DataFrame (if None, will load from the file)
            save_to_db: Whether to save the processed data to PostgreSQL database
            db_url: Database connection URL (if None, will use environment variable DATABASE_URL)

        Returns:
            Processed DataFrame with analysis results or None if processing failed
        """
        # Load data if not provided
        if df is None:
            logger.info("Loading data from files...")
            df = self.load_data()
            if df.empty:
                logger.error("No data to process.")
                return pd.DataFrame()

        logger.info(f"Starting data processing for {len(df)} documents...")

        try:
            # Step 1: Clean text
            logger.info("Cleaning text...")
            df = self.clean_text_column(df, column="body", new_column="clean_body")

            # Step 2: Tokenize text
            logger.info("Tokenizing text...")
            df = self.tokenize_column(df, text_column="clean_body", new_column="tokens")

            # Step 3: Vectorize documents
            logger.info("Vectorizing documents...")
            tfidf_matrix, vectorizer = self.vectorize_document(df["clean_body"])

            # Step 4: Cluster documents
            logger.info("Clustering documents...")
            model, labels = self.cluster_documents(tfidf_matrix, method="kmeans", n_clusters=5)
            df["cluster"] = labels

            # Generate cluster topics
            topics = self.generate_cluster_topics(labels, tfidf_matrix, vectorizer)
            logger.info(f"Generated {len(topics)} cluster topics")

            # Extract top words
            top_words = self.extract_top_words(tfidf_matrix, vectorizer)
            logger.info(f"Top words: {', '.join(top_words[:5])}...")

            # Step 5: Extract entities
            logger.info("Extracting named entities...")
            df = self.extract_entities(df, text_column="clean_body")

            # Step 6: Analyze sentiment
            logger.info("Analyzing sentiment...")
            df = self.analyze_sentiment(df, text_column="clean_body")

            # Step 7: Generate corpus summary
            logger.info("Generating corpus summary...")
            summary = self.summarize_corpus(df, text_column="clean_body")
            logger.info(f"Generated summary of {len(summary)} characters")

            # Add metadata
            df["processed_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            df["corpus_summary"] = summary

            logger.info("Data processing completed successfully")

            # Save to PostgreSQL if requested
            if save_to_db:
                try:
                    logger.info("Saving processed and summarized data to PostgreSQL database...")
                    self.save_to_postgresql(df, db_url=db_url, table_name="summarized_emails")
                except Exception as e:
                    logger.error(f"Failed to save to PostgreSQL: {e}")
                    logger.info("Continuing without saving to database.")

            return df

        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            return pd.DataFrame()


# Main execution block
if __name__ == "__main__":
    # Create SummarizationClassification instance
    logger.info("Initializing SummarizationClassification...")
    analyzer = SummarizationClassification(
        input_dir="./output/processed_data/", output_dir="./output/analysis_results/"
    )

    # Process data
    logger.info("Processing email data...")
    processed_df = analyzer.process_data()

    if processed_df is not None:
        # Save to different formats
        analyzer.save_to_csv(processed_df)
        analyzer.save_to_json(processed_df)
        analyzer.save_to_sqlite(processed_df)
        analyzer.save_to_pickle(processed_df)

        # Print summary statistics
        print("\n=== Analysis Results ===")
        print(f"Processed {len(processed_df)} emails")
        print(f"Found {len(processed_df['cluster'].unique())} clusters")
        print(f"Most common sentiment: {processed_df['sentiment'].value_counts().index[0]}")
        print(f"Number of unique persons mentioned: {sum(processed_df['persons'].apply(len))}")
        print(
            f"Number of unique organizations mentioned: {sum(processed_df['organizations'].apply(len))}"
        )
        print("\nDone!")
