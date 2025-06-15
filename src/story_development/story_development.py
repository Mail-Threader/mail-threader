import json
import os
import pickle
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import initialize_nltk, save_to_postgresql


# Custom exceptions
class StoryDevelopmentError(Exception):
    """Base exception for StoryDevelopment class."""

    pass


class DataLoadingError(StoryDevelopmentError):
    """Exception raised for errors in loading data."""

    pass


class AnalysisError(StoryDevelopmentError):
    """Exception raised for errors in analysis."""

    pass


class StoryGenerationError(StoryDevelopmentError):
    """Exception raised for errors in story generation."""

    pass


class FileOperationError(StoryDevelopmentError):
    """Exception raised for errors in file operations."""

    pass


class DatabaseError(StoryDevelopmentError):
    """Exception raised for errors in database operations."""

    pass


# Initialize NLTK resources
try:
    initialize_nltk()
except Exception as e:
    logger.error(f"Error initializing NLTK resources: {e}")
    raise DataLoadingError(f"Failed to initialize NLTK resources: {e}")


class StoryDevelopment:
    """
    Enhanced class responsible for story development tasks:
    - Identifying key actors and their relationships
    - Tracking topics over time
    - Detecting significant events
    - Constructing narratives from email threads
    - Generating story summaries
    - Analyzing email connections and patterns
    - Generating comprehensive story reports
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

        Raises:
            FileOperationError: If directories cannot be created
        """
        try:
            self.input_dir = input_dir
            self.analysis_dir = analysis_dir
            self.output_dir = output_dir

            # Create directories if they don't exist
            for directory in [input_dir, analysis_dir, output_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)

            # Initialize stopwords and other NLP components
            try:
                self.stop_words = set(stopwords.words("english"))
                self.lemmatizer = nltk.WordNetLemmatizer()
            except Exception as e:
                raise DataLoadingError(f"Failed to initialize NLP components: {e}")

            # Initialize TF-IDF vectorizer for text similarity
            self.vectorizer = TfidfVectorizer(
                max_features=5000, min_df=5, max_df=0.8, stop_words="english"
            )
        except Exception as e:
            raise FileOperationError(f"Failed to initialize StoryDevelopment: {e}")

    def load_data(self, file_path=None):
        """
        Load processed email data from a file.

        Args:
            file_path (str, optional): Path to the processed data file.
                If not provided, the most recent file in input_dir will be used.

        Returns:
            pandas.DataFrame: DataFrame containing the processed email data

        Raises:
            DataLoadingError: If data cannot be loaded
            FileOperationError: If file operations fail
        """
        try:
            if file_path is None:
                # Find the most recent processed data file
                pkl_files = [
                    f
                    for f in os.listdir(self.input_dir)
                    if f.startswith("processed_data_") and f.endswith(".pkl")
                ]
                if not pkl_files:
                    raise DataLoadingError(f"No processed data files found in {self.input_dir}")

                # Sort by timestamp in files
                pkl_files.sort(reverse=True)
                file_path = os.path.join(self.input_dir, pkl_files[0])

            if not os.path.exists(file_path):
                raise FileOperationError(f"File not found: {file_path}")

            df = pd.read_pickle(file_path)
            if df.empty:
                raise DataLoadingError(f"Loaded data from {file_path} is empty")

            logger.info(f"Loaded data from {file_path}: {len(df)} emails")
            return df
        except Exception as e:
            if isinstance(e, (DataLoadingError, FileOperationError)):
                raise
            raise DataLoadingError(f"Error loading data: {e}")

    def load_analysis_results(self, file_path=None):
        """
        Load analysis results from a file.

        Args:
            file_path (str, optional): Path to the analysis results file.
                If not provided, the most recent file in analysis_dir will be used.

        Returns:
            dict: Dictionary containing analysis results
        """
        if file_path is None:
            # Find the most recent analysis results file
            pkl_files = [
                f
                for f in os.listdir(self.analysis_dir)
                if f.startswith("analysis_results_") and f.endswith(".pkl")
            ]
            if not pkl_files:
                logger.error(f"No analysis results files found in {self.analysis_dir}")
                return None

            # Sort by timestamp in files
            pkl_files.sort(reverse=True)
            file_path = os.path.join(self.analysis_dir, pkl_files[0])

        try:
            with open(file_path, "rb") as f:
                results = pickle.load(f)
            logger.info(f"Loaded analysis results from {file_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading analysis results from {file_path}: {e}")
            return None

    @staticmethod
    def identify_key_actors(df, top_n=20):
        """
        Identify key actors in the email dataset based on email frequency and centrality.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            top_n (int): Number of top actors to identify

        Returns:
            dict: Dictionary containing key actors and their metrics
        """
        logger.info("Identifying key actors")

        # Extract email addresses
        email_pattern = r"[\w\.-]+@[\w\.-]+"

        # Create a directed graph
        G = nx.DiGraph()

        # Track email frequencies
        sender_counts = Counter()
        recipient_counts = Counter()
        edge_weights = Counter()

        # Process each email with progress bar
        logger.info("Processing emails to build actor network...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building actor network"):
            sender_emails = (
                re.findall(email_pattern, row["from"]) if row["from"] is not None else []
            )
            recipient_emails = re.findall(email_pattern, row["to"]) if row["to"] is not None else []

            # Add edges from sender to recipients
            for sender in sender_emails:
                sender_counts[sender] += 1
                for recipient in recipient_emails:
                    recipient_counts[recipient] += 1
                    edge_weights[(sender, recipient)] += 1

        # Add nodes and edges to the graph
        logger.info("Adding nodes and edges to graph...")
        for (sender, recipient), weight in tqdm(edge_weights.items(), desc="Building graph"):
            if not G.has_node(sender):
                G.add_node(sender, type="sender", sent_count=sender_counts[sender])
            if not G.has_node(recipient):
                G.add_node(
                    recipient,
                    type="recipient",
                    received_count=recipient_counts[recipient],
                )
            G.add_edge(sender, recipient, weight=weight)

        # Calculate network centrality metrics
        try:
            logger.info("Calculating network metrics...")
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)

            # Betweenness centrality (who connects different groups)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G)))

            # PageRank (importance based on connections)
            pagerank = nx.pagerank(G)

            # Combine metrics
            actor_metrics = {}
            for actor in tqdm(G.nodes(), desc="Calculating actor metrics"):
                actor_metrics[actor] = {
                    "sent": sender_counts.get(actor, 0),
                    "received": recipient_counts.get(actor, 0),
                    "total": sender_counts.get(actor, 0) + recipient_counts.get(actor, 0),
                    "degree_centrality": degree_centrality.get(actor, 0),
                    "betweenness_centrality": betweenness_centrality.get(actor, 0),
                    "pagerank": pagerank.get(actor, 0),
                }

            # Sort actors by total email count
            sorted_actors = sorted(actor_metrics.items(), key=lambda x: x[1]["total"], reverse=True)

            # Get top actors
            top_actors = {actor: metrics for actor, metrics in sorted_actors[:top_n]}

            return {"top_actors": top_actors, "graph": G}

        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            return {
                "top_actors": {
                    actor: {"sent": count, "received": recipient_counts.get(actor, 0)}
                    for actor, count in sender_counts.most_common(top_n)
                },
                "graph": G,
            }

    @staticmethod
    def track_topics_over_time(df, analysis_results):
        """
        Track how topics evolve over time.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            dict: Dictionary containing topic evolution data
        """
        logger.info("Tracking topics over time")

        if (
            analysis_results is None
            or "topics" not in analysis_results
            or analysis_results["topics"] is None
        ):
            logger.warning("No topic modeling results available for tracking")
            return {"time_periods": [], "topic_counts": {}, "topic_keywords": {}}

        # Check if the date column exists and has valid data
        if "date" not in df.columns or df["date"].isna().all():
            logger.warning("No date information available for topic tracking")
            return {"time_periods": [], "topic_counts": {}, "topic_keywords": {}}

        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                # Try different date formats
                date_formats = [
                    "%d/%m/%Y %H:%M:%S",  # DD/MM/YYYY HH:MM:SS
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY HH:MM:SS
                    "%Y/%m/%d %H:%M:%S",  # YYYY/MM/DD HH:MM:SS
                ]

                for date_format in date_formats:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
                        # If we successfully parsed any dates, break the loop
                        if not df["date"].isna().all():
                            break
                    except Exception:
                        continue

                # If all formats failed, try without specifying format
                if df["date"].isna().all():
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                return {"time_periods": [], "topic_counts": {}, "topic_keywords": {}}

        # Drop rows with missing dates
        df_with_date = df.dropna(subset=["date"]).copy()
        if len(df_with_date) == 0:
            logger.warning("No valid dates available for topic tracking")
            return {"time_periods": [], "topic_counts": {}, "topic_keywords": {}}

        # Get topic assignments and document-topic matrix
        if "dominant_topic" not in df.columns and "doc_topic_matrix" in analysis_results["topics"]:
            # Get indices of rows with valid dates
            valid_indices = df.dropna(subset=["date"]).index

            # Assign the dominant topic to each document
            doc_topic_matrix = analysis_results["topics"]["doc_topic_matrix"]

            # Filter doc_topic_matrix to match the rows in df_with_date
            if len(doc_topic_matrix) != len(df_with_date):
                logger.info(
                    f"Filtering doc_topic_matrix from {len(doc_topic_matrix)} to {len(df_with_date)} rows"
                )
                # Get the indices in the original dataframe
                valid_indices_list = valid_indices.tolist()
                # Filter doc_topic_matrix to only include rows with valid dates
                if len(valid_indices_list) <= len(doc_topic_matrix):
                    doc_topic_matrix = doc_topic_matrix[valid_indices_list]
                else:
                    logger.warning(
                        f"Cannot filter doc_topic_matrix: valid indices ({len(valid_indices_list)}) > matrix rows ({len(doc_topic_matrix)})"
                    )
                    return {
                        "time_periods": [],
                        "topic_counts": {},
                        "topic_keywords": {},
                    }

            df_with_date["dominant_topic"] = np.argmax(doc_topic_matrix, axis=1) + 1

        # Create time-based features
        df_with_date["year"] = df_with_date["date"].dt.year
        df_with_date["month"] = df_with_date["date"].dt.month
        df_with_date["week"] = df_with_date["date"].dt.isocalendar().week

        # Create a date string for grouping by month
        df_with_date["month_year"] = df_with_date["date"].dt.strftime("%Y-%m")

        # Group by month and topic
        topic_time_counts = (
            df_with_date.groupby(["month_year", "dominant_topic"]).size().unstack(fill_value=0)
        )

        # Get topic keywords
        topics = analysis_results["topics"]["topics"]

        # Create a dictionary with topic evolution data
        topic_evolution = {
            "time_periods": list(topic_time_counts.index),
            "topic_counts": topic_time_counts.to_dict(),
            "topic_keywords": topics,
        }

        return topic_evolution

    def detect_significant_events(self, df, window_size=7, threshold=2):
        """
        Detect significant events based on email volume spikes and content.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            window_size (int): Size of the rolling window in days
            threshold (float): Number of standard deviations above mean to consider as a spike

        Returns:
            list: List of detected events with dates and related emails
        """
        logger.info("Detecting significant events")

        # Check if the date column exists and has valid data
        if "date" not in df.columns or df["date"].isna().all():
            logger.warning("No date information available for event detection")
            return []

        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                # Try different date formats
                date_formats = [
                    "%d/%m/%Y %H:%M:%S",  # DD/MM/YYYY HH:MM:SS
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY HH:MM:SS
                    "%Y/%m/%d %H:%M:%S",  # YYYY/MM/DD HH:MM:SS
                ]

                for date_format in date_formats:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
                        # If we successfully parsed any dates, break the loop
                        if not df["date"].isna().all():
                            break
                    except Exception:
                        continue

                # If all formats failed, try without specifying format
                if df["date"].isna().all():
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                return []

        # Drop rows with missing dates
        df_with_date = df.dropna(subset=["date"]).copy()
        if len(df_with_date) == 0:
            logger.warning("No valid dates available for event detection")
            return []

        # Sort by date
        df_with_date = df_with_date.sort_values("date")

        # Count emails per day
        logger.info("Counting emails per day...")
        daily_counts = df_with_date.groupby(df_with_date["date"].dt.date).size()

        # Calculate rolling statistics
        logger.info("Calculating rolling statistics...")
        rolling_mean = daily_counts.rolling(window=window_size, min_periods=1).mean()
        rolling_std = daily_counts.rolling(window=window_size, min_periods=1).std()

        # Identify spikes
        logger.info("Identifying email volume spikes...")
        spikes = daily_counts[daily_counts > (rolling_mean + threshold * rolling_std)]

        # Extract events
        events = []
        logger.info("Analyzing detected events...")
        for date, count in tqdm(spikes.items(), desc="Processing events"):
            # Convert date to datetime for filtering
            event_date = pd.to_datetime(date)

            # Get emails from the spike day
            event_emails = df_with_date[
                (pd.to_datetime(df_with_date["date"].dt.date) >= event_date)
                & (pd.to_datetime(df_with_date["date"].dt.date) < event_date + timedelta(days=1))
            ]

            # Extract common words from these emails
            if len(event_emails) > 0:
                # Use clean_body column if available, otherwise use body
                text_column = "clean_body" if "clean_body" in event_emails.columns else "body"

                # Combine all text
                all_text = " ".join(event_emails[text_column].fillna(""))

                # Tokenize and count words
                words = re.findall(r"\b\w+\b", all_text.lower())
                words = [word for word in words if word not in self.stop_words and len(word) > 2]
                common_words = Counter(words).most_common(10)

                # Get sample subjects
                sample_subjects = event_emails["subject"].head(5).tolist()

                events.append(
                    {
                        "date": date,
                        "email_count": count,
                        "normal_level": rolling_mean[date],
                        "std_dev": rolling_std[date],
                        "deviation": (
                            (count - rolling_mean[date]) / rolling_std[date]
                            if rolling_std[date] > 0
                            else 0
                        ),
                        "common_words": common_words,
                        "sample_subjects": sample_subjects,
                        "email_ids": event_emails.index.tolist(),
                    }
                )

        # Sort events by deviation
        events.sort(key=lambda x: x["deviation"], reverse=True)

        return events

    @staticmethod
    def construct_email_threads(df):
        """
        Construct email threads based on subject lines and timestamps.

        Args:
            df (pandas.DataFrame): DataFrame containing email data

        Returns:
            dict: Dictionary of email threads
        """
        logger.info("Constructing email threads")

        # Check if required columns exist
        if "subject" not in df.columns or "date" not in df.columns:
            logger.warning("Required columns missing for thread construction")
            return {}

        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                # Try different date formats
                date_formats = [
                    "%d/%m/%Y %H:%M:%S",  # DD/MM/YYYY HH:MM:SS
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY HH:MM:SS
                    "%Y/%m/%d %H:%M:%S",  # YYYY/MM/DD HH:MM:SS
                ]

                for date_format in date_formats:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
                        # If we successfully parsed any dates, break the loop
                        if not df["date"].isna().all():
                            break
                    except Exception:
                        continue

                # If all formats failed, try without specifying format
                if df["date"].isna().all():
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                return {}

        # Create a copy of the dataframe with only the necessary columns
        thread_df = df[["subject", "date", "from", "to", "body"]].copy()

        # Clean subject lines for better matching
        def clean_subject(subject):
            # Remove 'Re:', 'Fwd:', etc.
            cleaned = re.sub(r"^(re|fwd|fw):\s*", "", subject.lower(), flags=re.IGNORECASE)
            # Remove other common prefixes/suffixes
            cleaned = re.sub(r"\[.*?\]", "", cleaned)
            # Remove extra whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        thread_df["clean_subject"] = thread_df["subject"].fillna("").apply(clean_subject)

        # Group by cleaned subject
        threads = {}
        for subject, group in thread_df.groupby("clean_subject"):
            if not subject:  # Skip empty subjects
                continue

            # Sort by date
            sorted_group = group.sort_values("date")

            # Only consider as thread if there are multiple emails
            if len(sorted_group) > 1:
                threads[subject] = sorted_group.to_dict("records")

        # Sort threads by size (number of emails)
        sorted_threads = {
            k: v for k, v in sorted(threads.items(), key=lambda item: len(item[1]), reverse=True)
        }

        return sorted_threads

    def analyze_email_connections(self, df: pd.DataFrame, analysis_results: dict) -> dict:
        """
        Analyze how emails are connected through events, topics, and content similarity.
        Optimized with parallel processing.

        Args:
            df (pd.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            dict: Dictionary containing email connection analysis
        """
        logger.info("Analyzing email connections")

        # Create a graph for email connections
        G = nx.Graph()

        # Add nodes (emails)
        logger.info("Adding email nodes to graph...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
            G.add_node(
                idx,
                date=row.get("date"),
                subject=row.get("subject"),
                from_email=row.get("from"),
                to_email=row.get("to"),
            )

        # Process different types of connections in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # 1. Connect emails by topic
            if "topics" in analysis_results and "doc_topic_matrix" in analysis_results["topics"]:
                futures.append(
                    executor.submit(
                        self._process_topic_connections,
                        analysis_results["topics"]["doc_topic_matrix"],
                        G,
                    )
                )

            # 2. Connect emails by temporal proximity
            futures.append(executor.submit(self._process_temporal_connections, df, G))

            # 3. Connect emails by content similarity
            text_column = "clean_body" if "clean_body" in df.columns else "body"
            tfidf_matrix = self.vectorizer.fit_transform(df[text_column].fillna(""))
            futures.append(executor.submit(self._process_content_connections, tfidf_matrix, G))

            # 4. Connect emails by participants
            futures.append(executor.submit(self._process_participant_connections, df, G))

            # Wait for all connection processing to complete
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing connections"
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in connection processing: {e}")

        # Analyze the graph
        logger.info("Analyzing graph properties...")
        connection_analysis = {
            "graph": G,
            "communities": list(nx.community.greedy_modularity_communities(G)),
            "central_emails": self._find_central_emails(G),
            "connection_types": self._analyze_connection_types(G),
            "temporal_patterns": self._analyze_temporal_patterns(G, df),
        }

        return connection_analysis

    @staticmethod
    def _process_topic_connections(doc_topic_matrix: np.ndarray, G: nx.Graph) -> None:
        """Process topic-based connections in parallel."""
        logger.info("Connecting emails by topic similarity...")
        for i in tqdm(range(len(doc_topic_matrix)), desc="Processing topic connections"):
            for j in range(i + 1, len(doc_topic_matrix)):
                similarity = cosine_similarity(
                    doc_topic_matrix[i].reshape(1, -1),
                    doc_topic_matrix[j].reshape(1, -1),
                )[0][0]
                if similarity > 0.7:  # Threshold for topic similarity
                    G.add_edge(i, j, type="topic", similarity=float(similarity))

    def _process_temporal_connections(self, df: pd.DataFrame, G: nx.Graph) -> None:
        """Process temporal-based connections in parallel."""
        logger.info("Connecting emails by temporal proximity...")
        df["date"] = pd.to_datetime(df["date"])
        for i in tqdm(range(len(df)), desc="Processing temporal connections"):
            for j in range(i + 1, len(df)):
                time_diff = abs((df.iloc[i]["date"] - df.iloc[j]["date"]).total_seconds())
                if time_diff < 3600:  # Within 1 hour
                    G.add_edge(i, j, type="temporal", time_diff=float(time_diff))

    def _process_content_connections(self, tfidf_matrix: Any, G: nx.Graph) -> None:
        """Process content-based connections in parallel."""
        logger.info("Connecting emails by content similarity...")
        # Convert sparse matrix to dense for easier processing
        dense_matrix = tfidf_matrix.toarray()
        for i in tqdm(range(len(dense_matrix)), desc="Processing content connections"):
            for j in range(i + 1, len(dense_matrix)):
                similarity = cosine_similarity(
                    dense_matrix[i].reshape(1, -1), dense_matrix[j].reshape(1, -1)
                )[0][0]
                if similarity > 0.5:  # Threshold for content similarity
                    G.add_edge(i, j, type="content", similarity=float(similarity))

    def _process_participant_connections(self, df: pd.DataFrame, G: nx.Graph) -> None:
        """Process participant-based connections in parallel."""
        logger.info("Connecting emails by participants...")
        for i in tqdm(range(len(df)), desc="Processing participant connections"):
            for j in range(i + 1, len(df)):
                if self._share_participants(df.iloc[i], df.iloc[j]):
                    G.add_edge(i, j, type="participant")

    @staticmethod
    def _share_participants(email1: pd.Series, email2: pd.Series) -> bool:
        """Check if two emails share participants."""
        email_pattern = r"[\w\.-]+@[\w\.-]+"

        participants1 = set()
        if email1["from"]:
            participants1.update(re.findall(email_pattern, email1["from"]))
        if email1["to"]:
            participants1.update(re.findall(email_pattern, email1["to"]))

        participants2 = set()
        if email2["from"]:
            participants2.update(re.findall(email_pattern, email2["from"]))
        if email2["to"]:
            participants2.update(re.findall(email_pattern, email2["to"]))

        return bool(participants1.intersection(participants2))

    @staticmethod
    def _find_central_emails(G: nx.Graph) -> List[Tuple[int, float]]:
        """Find central emails in the graph using various centrality measures."""
        central_emails = []

        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

        # Combine centrality measures
        for node in G.nodes():
            combined_score = (
                degree_centrality[node]
                + betweenness_centrality[node]
                + eigenvector_centrality[node]
            ) / 3
            central_emails.append((node, combined_score))

        return sorted(central_emails, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _analyze_connection_types(G: nx.Graph) -> Dict[str, int]:
        """Analyze the distribution of connection types in the graph."""
        connection_types = defaultdict(int)
        for _, _, data in G.edges(data=True):
            connection_types[data["type"]] += 1
        return dict(connection_types)

    @staticmethod
    def _analyze_temporal_patterns(G: nx.Graph, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in email connections."""
        temporal_patterns = {
            "hourly_distribution": defaultdict(int),
            "daily_distribution": defaultdict(int),
            "weekly_distribution": defaultdict(int),
        }

        for node in G.nodes():
            date = df.iloc[node]["date"]
            # Skip NaT values
            if pd.isna(date):
                continue
            temporal_patterns["hourly_distribution"][date.hour] += 1
            temporal_patterns["daily_distribution"][date.day_name()] += 1
            # Use isocalendar() only if date is valid
            if not pd.isna(date):
                temporal_patterns["weekly_distribution"][date.isocalendar()[1]] += 1

        return temporal_patterns

    def generate_comprehensive_story(self, df: pd.DataFrame, analysis_results: dict) -> dict:
        """
        Generate a comprehensive story combining all analysis aspects.

        Args:
            df (pd.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            dict: Comprehensive story combining all aspects
        """
        logger.info("Generating comprehensive story")

        # Get all analysis components
        key_actors = self.identify_key_actors(df)
        topic_evolution = self.track_topics_over_time(df, analysis_results)
        significant_events = self.detect_significant_events(df)
        email_threads = self.construct_email_threads(df)
        email_connections = self.analyze_email_connections(df, analysis_results)

        # Generate a comprehensive story
        story = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "total_emails": len(df),
                "time_span": {
                    "start": df["date"].min().isoformat(),
                    "end": df["date"].max().isoformat(),
                },
            },
            "key_actors": self._format_key_actors_story(key_actors),
            "topic_evolution": self._format_topic_evolution_story(topic_evolution),
            "significant_events": self._format_events_story(significant_events),
            "email_threads": self._format_threads_story(email_threads),
            "email_connections": self._format_connections_story(email_connections),
            "narrative_summary": self._generate_narrative_summary(
                key_actors, topic_evolution, significant_events, email_threads
            ),
        }

        return story

    @staticmethod
    def _format_key_actors_story(key_actors: dict) -> dict:
        """Format key actors analysis into a story format."""
        if not key_actors or "top_actors" not in key_actors:
            return {}

        story = {
            "title": "Key Actors in the Email Network",
            "main_characters": [],
            "relationships": [],
        }

        # Format main characters
        for actor, metrics in list(key_actors["top_actors"].items())[:5]:
            character = {
                "name": actor,
                "role": "Key Participant",
                "metrics": metrics,
                "influence_score": (
                    metrics.get("degree_centrality", 0)
                    + metrics.get("betweenness_centrality", 0)
                    + metrics.get("pagerank", 0)
                )
                / 3,
            }
            story["main_characters"].append(character)

        # Format relationships
        if "graph" in key_actors:
            G = key_actors["graph"]
            for actor1, actor2 in G.edges():
                if actor1 in key_actors["top_actors"] and actor2 in key_actors["top_actors"]:
                    relationship = {
                        "participants": [actor1, actor2],
                        "strength": G[actor1][actor2].get("weight", 1),
                        "type": "Email Communication",
                    }
                    story["relationships"].append(relationship)

        return story

    @staticmethod
    def _format_topic_evolution_story(topic_evolution: dict) -> dict:
        """Format topic evolution analysis into a story format."""
        if not topic_evolution:
            return {}

        story = {
            "title": "Evolution of Topics Over Time",
            "timeline": [],
            "key_topics": [],
        }

        # Format timeline
        if "time_periods" in topic_evolution and "topic_counts" in topic_evolution:
            for period in topic_evolution["time_periods"]:
                period_data = {"period": period, "topics": []}
                for topic_id, count in topic_evolution["topic_counts"].items():
                    if period in count:
                        period_data["topics"].append(
                            {
                                "id": topic_id,
                                "count": count[period],
                                "keywords": topic_evolution["topic_keywords"].get(topic_id, []),
                            }
                        )
                story["timeline"].append(period_data)

        # Format key topics
        if "topic_keywords" in topic_evolution:
            for topic_id, keywords in topic_evolution["topic_keywords"].items():
                story["key_topics"].append(
                    {
                        "id": topic_id,
                        "keywords": keywords,
                        "significance": len(keywords),  # Simple significance metric
                    }
                )

        return story

    @staticmethod
    def _format_events_story(events: List[dict]) -> dict:
        """Format significant events analysis into a story format."""
        if not events:
            return {}

        story = {"title": "Significant Events Timeline", "events": []}

        for event in events[:10]:  # Top 10 events
            event_story = {
                "date": (
                    event["date"].isoformat()
                    if isinstance(event["date"], datetime)
                    else str(event["date"])
                ),
                "significance": event["deviation"],
                "description": f"Email volume spike: {event['email_count']} emails",
                "key_topics": [word for word, _ in event["common_words"][:5]],
                "sample_subjects": event["sample_subjects"],
            }
            story["events"].append(event_story)

        return story

    @staticmethod
    def _format_threads_story(threads: dict) -> dict:
        """Format email threads analysis into a story format."""
        if not threads:
            return {}

        story = {"title": "Major Email Threads", "threads": []}

        for subject, thread in list(threads.items())[:10]:  # Top 10 threads
            thread_story = {
                "subject": subject,
                "size": len(thread),
                "participants": list(set(email["from"] for email in thread if "from" in email)),
                "timeline": {
                    "start": (thread[0]["date"].isoformat() if "date" in thread[0] else None),
                    "end": (thread[-1]["date"].isoformat() if "date" in thread[-1] else None),
                },
            }
            story["threads"].append(thread_story)

        return story

    @staticmethod
    def _format_connections_story(connections: dict) -> dict:
        """Format email connections analysis into a story format."""
        if not connections:
            return {}

        story = {
            "title": "Email Network Analysis",
            "network_properties": {
                "total_connections": connections["graph"].number_of_edges(),
                "connection_types": connections["connection_types"],
                "communities": len(connections["communities"]),
            },
            "central_emails": [
                {"email_id": email_id, "centrality_score": score}
                for email_id, score in connections["central_emails"][:5]
            ],
            "temporal_patterns": connections["temporal_patterns"],
        }

        return story

    @staticmethod
    def _generate_narrative_summary(
        key_actors: dict,
        topic_evolution: dict,
        significant_events: List[dict],
        email_threads: dict,
    ) -> str:
        """Generate a narrative summary combining all story elements."""
        summary_parts = []

        # Add key actors summary
        if key_actors and "top_actors" in key_actors:
            top_actors = list(key_actors["top_actors"].items())[:3]
            actors_summary = "The main participants in this email network are "
            actors_summary += ", ".join(
                f"{actor} ({metrics['sent']} emails sent)" for actor, metrics in top_actors
            )
            summary_parts.append(actors_summary)

        # Add topic evolution summary
        if topic_evolution and "topic_keywords" in topic_evolution:
            topics_summary = "The main topics discussed include "
            topics_summary += ", ".join(
                f"{topic_id} ({', '.join(keywords[:3])})"
                for topic_id, keywords in list(topic_evolution["topic_keywords"].items())[:3]
            )
            summary_parts.append(topics_summary)

        # Add significant events summary
        if significant_events:
            events_summary = "Key events occurred on "
            events_summary += ", ".join(
                f"{event['date']} ({event['email_count']} emails)"
                for event in significant_events[:3]
            )
            summary_parts.append(events_summary)

        # Add email threads summary
        if email_threads:
            threads_summary = "Major email threads include "
            threads_summary += ", ".join(
                f"'{subject}' ({len(thread)} emails)"
                for subject, thread in list(email_threads.items())[:3]
            )
            summary_parts.append(threads_summary)

        return " ".join(summary_parts)

    @staticmethod
    def _generate_enhanced_summaries(df: pd.DataFrame, analysis_results: dict) -> dict:
        """
        Generate enhanced summaries using multiple styles and metrics.

        Args:
            df (pd.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            dict: Enhanced summaries with multiple styles and metrics
        """
        if not analysis_results or "summaries" not in analysis_results:
            return {}

        summaries = analysis_results["summaries"]
        enhanced_summaries = {"styles": {}, "metrics": {}, "key_information": {}}

        # Process each summary style
        for subject, styles in summaries.items():
            for style, summary in styles.items():
                if style not in enhanced_summaries["styles"]:
                    enhanced_summaries["styles"][style] = []
                    enhanced_summaries["metrics"][style] = []
                    enhanced_summaries["key_information"][style] = []

                # Add summary content
                enhanced_summaries["styles"][style].append(
                    {
                        "subject": subject,
                        "summary": summary["summary"],
                        "original_length": summary["original_length"],
                    }
                )

                # Add metrics
                enhanced_summaries["metrics"][style].append(summary["metrics"])

                # Add key information
                enhanced_summaries["key_information"][style].append(summary["key_information"])

        return enhanced_summaries

    @staticmethod
    def _generate_visualization_insights(analysis_results: dict) -> dict:
        """
        Generate insights from visualization data.

        Args:
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            dict: Visualization insights
        """
        insights = {
            "email_volume": {},
            "network_analysis": {},
            "topic_distribution": {},
            "sentiment_analysis": {},
            "entity_relationships": {},
        }

        # Process email volume insights
        if "email_volume" in analysis_results:
            volume_data = analysis_results["email_volume"]
            insights["email_volume"] = {
                "peak_periods": volume_data.get("peak_periods", []),
                "temporal_patterns": volume_data.get("temporal_patterns", {}),
                "volume_trends": volume_data.get("volume_trends", {}),
            }

        # Process network analysis insights
        if "network" in analysis_results:
            network_data = analysis_results["network"]
            insights["network_analysis"] = {
                "central_nodes": network_data.get("central_nodes", []),
                "community_structure": network_data.get("communities", {}),
                "connection_patterns": network_data.get("connection_patterns", {}),
            }

        # Process topic distribution insights
        if "topics" in analysis_results:
            topic_data = analysis_results["topics"]
            insights["topic_distribution"] = {
                "dominant_topics": topic_data.get("dominant_topics", []),
                "topic_evolution": topic_data.get("topic_evolution", {}),
                "topic_relationships": topic_data.get("topic_relationships", {}),
            }

        # Process sentiment analysis insights
        if "sentiment" in analysis_results:
            sentiment_data = analysis_results["sentiment"]
            insights["sentiment_analysis"] = {
                "overall_sentiment": sentiment_data.get("overall_sentiment", {}),
                "sentiment_trends": sentiment_data.get("sentiment_trends", {}),
                "key_sentiment_events": sentiment_data.get("key_events", []),
            }

        # Process entity relationship insights
        if "entities" in analysis_results:
            entity_data = analysis_results["entities"]
            insights["entity_relationships"] = {
                "key_entities": entity_data.get("key_entities", []),
                "entity_networks": entity_data.get("entity_networks", {}),
                "entity_evolution": entity_data.get("entity_evolution", {}),
            }

        return insights

    def develop_stories(self, df=None, analysis_results=None, max_workers=None):
        """
        Enhanced story development from email data with parallel processing.

        Args:
            df (pandas.DataFrame, optional): DataFrame containing email data.
                If not provided, data will be loaded from the most recent file.
            analysis_results (dict, optional): Dictionary containing analysis results.
                If not provided, results will be loaded from the most recent file.
            max_workers (int, optional): Maximum number of worker threads.
                If None, it will use min(32, os.cpu_count() + 4)

        Returns:
            list: List of story summaries for HTML report
        """
        try:
            logger.info("Starting story development process...")

            # Load data if not provided
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logger.info("Loading email data...")
                df = self.load_data()
                if isinstance(df, pd.DataFrame) and df.empty:
                    raise DataLoadingError("No data available for story development")

            # Load analysis results if not provided
            if analysis_results is None:
                logger.info("Loading analysis results...")
                analysis_results = self.load_analysis_results()
                if not analysis_results:
                    raise AnalysisError("No analysis results available")

            # Validate required columns
            required_columns = ["date", "subject", "from", "to", "body"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataLoadingError(f"Missing required columns: {missing_columns}")

            logger.info(f"Developing comprehensive stories from {len(df)} emails")

            # Generate story components
            key_actors = self.identify_key_actors(df)
            topic_evolution = self.track_topics_over_time(df, analysis_results)
            significant_events = self.detect_significant_events(df)
            email_threads = self.construct_email_threads(df)

            # Generate story summaries for HTML report
            stories = self.generate_story_summaries(
                df,
                analysis_results,
                key_actors,
                topic_evolution,
                significant_events,
                email_threads,
            )

            # Save stories
            logger.info("Saving stories...")
            stories_path = self.save_stories(stories)

            logger.info("Story development completed successfully!")
            return stories

        except Exception as e:
            if isinstance(e, (DataLoadingError, AnalysisError, StoryGenerationError)):
                raise
            raise StoryGenerationError(f"Unexpected error in story development: {e}")

    def save_stories(self, stories, save_to_db=False):
        """
        Save generated stories to files and database.

        Args:
            stories (list): List of story summaries
            save_to_db (bool): Whether to save to database

        Returns:
            str: Path to the saved stories

        Raises:
            FileOperationError: If file operations fail
            DatabaseError: If database operations fail
        """
        if not stories:
            logger.warning("No stories to save")
            return None

        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as JSON
            json_path = os.path.join(self.output_dir, f"stories_{timestamp}_v1.json")
            try:
                logger.info("Saving stories to JSON...")
                with open(json_path, "w") as f:
                    json.dump(stories, f, indent=2, default=str)
                logger.info(f"Successfully saved stories to {json_path}")
            except Exception as e:
                raise FileOperationError(f"Failed to save stories to JSON: {e}")

            # Save as HTML
            html_path = os.path.join(self.output_dir, f"stories_{timestamp}_v1.html")
            try:
                logger.info("Generating and saving HTML report...")
                html_content = self._generate_html_report(stories)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"Successfully saved HTML report to {html_path}")
            except Exception as e:
                raise FileOperationError(f"Failed to save stories to HTML: {e}")

            if save_to_db:
                try:
                    logger.info("Saving stories to database...")
                    # Convert stories list to DataFrame
                    stories_df = pd.DataFrame(stories)

                    # Add timestamp column
                    stories_df["timestamp"] = timestamp

                    # Save to PostgreSQL database
                    save_to_postgresql(
                        stories_df,
                        table_name="stories",
                        if_exists="append",
                        success_message=f"Saved {len(stories)} stories to database",
                    )
                    logger.info("Successfully saved stories to database")
                except Exception as e:
                    raise DatabaseError(f"Failed to save stories to database: {e}")

            return html_path  # Return the HTML path instead of JSON path
        except Exception as e:
            if isinstance(e, (FileOperationError, DatabaseError)):
                raise
            raise FileOperationError(f"Unexpected error saving stories: {e}")

    def _generate_html_report(self, stories):
        """Generate a simple HTML report from the stories."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Analysis Stories</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .story {
                    margin-bottom: 30px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .story-title {
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }
                .story-content {
                    margin-bottom: 10px;
                }
                .story-metrics {
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 3px;
                    margin-top: 10px;
                }
                .metric {
                    margin: 5px 0;
                }
                .search-box {
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 20px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .filter-buttons {
                    margin-bottom: 20px;
                }
                .filter-button {
                    padding: 5px 10px;
                    margin-right: 5px;
                    border: 1px solid #ddd;
                    background: #f5f5f5;
                    cursor: pointer;
                }
                .filter-button.active {
                    background: #007bff;
                    color: white;
                }
                .related-emails {
                    margin-top: 15px;
                    border-top: 1px solid #ddd;
                    padding-top: 15px;
                }
                .email-item {
                    margin-bottom: 15px;
                    padding: 10px;
                    background: #f9f9f9;
                    border-radius: 4px;
                }
                .email-header {
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .email-preview {
                    font-size: 0.9em;
                    color: #666;
                }
                .email-meta {
                    font-size: 0.8em;
                    color: #888;
                    margin-top: 5px;
                }
                .email-keyword {
                    display: inline-block;
                    background: #e9ecef;
                    padding: 2px 6px;
                    border-radius: 3px;
                    margin-right: 5px;
                    font-size: 0.8em;
                }
            </style>
        </head>
        <body>
            <h1>Email Analysis Stories</h1>
            <input type="text" class="search-box" id="searchBox" placeholder="Search stories...">
            <div class="filter-buttons">
                <button class="filter-button active" data-filter="all">All</button>
                <button class="filter-button" data-filter="key_actor">Key Actors</button>
                <button class="filter-button" data-filter="significant_event">Events</button>
                <button class="filter-button" data-filter="email_thread">Threads</button>
                <button class="filter-button" data-filter="topic_evolution">Topics</button>
            </div>
            <div id="stories">
        """

        for story in stories:
            story_type = story.get("type", "")
            story_html = f"""
            <div class="story" data-type="{story_type}">
                <div class="story-title">{story.get('title', 'Untitled Story')}</div>
                <div class="story-content">{story.get('summary', '')}</div>
            """

            # Add metrics based on story type
            if story_type == "key_actor":
                metrics = story.get("metrics", {})
                story_html += f"""
                <div class="story-metrics">
                    <div class="metric">Emails Sent: {metrics.get('sent', 0)}</div>
                    <div class="metric">Emails Received: {metrics.get('received', 0)}</div>
                    <div class="metric">Busiest Day: {story.get('communication_patterns', {}).get('busiest_day', 'N/A')}</div>
                    <div class="metric">Average Response Time: {story.get('communication_patterns', {}).get('avg_response_time', 'N/A')}</div>
                </div>
                """
            elif story_type == "significant_event":
                metrics = story.get("event_metrics", {})
                story_html += f"""
                <div class="story-metrics">
                    <div class="metric">Email Count: {story.get('email_count', 0)}</div>
                    <div class="metric">Participants: {metrics.get('participant_count', 0)}</div>
                    <div class="metric">Average Email Length: {metrics.get('avg_email_length', 0):.0f} characters</div>
                    <div class="metric">Reply Rate: {metrics.get('reply_rate', 0):.1%}</div>
                </div>
                """
            elif story_type == "email_thread":
                metrics = story.get("thread_metrics", {})
                story_html += f"""
                <div class="story-metrics">
                    <div class="metric">Number of Emails: {story.get('num_emails', 0)}</div>
                    <div class="metric">Participants: {metrics.get('participant_count', 0)}</div>
                    <div class="metric">Duration: {metrics.get('duration_hours', 'N/A')} hours</div>
                    <div class="metric">Average Response Time: {metrics.get('avg_response_time', 'N/A')}</div>
                </div>
                """
            elif story_type == "topic_evolution":
                metrics = story.get("topic_metrics", {})
                story_html += f"""
                <div class="story-metrics">
                    <div class="metric">Trend: {metrics.get('trend', 'Unknown')}</div>
                    <div class="metric">Peak Period: {metrics.get('peak_period', 'N/A')}</div>
                    <div class="metric">Peak Count: {metrics.get('peak_count', 0)}</div>
                </div>
                """

            # Add related emails section
            if "related_emails" in story and story["related_emails"]:
                story_html += """
                <div class="related-emails">
                    <h3>Related Emails</h3>
                """
                for email in story["related_emails"]:
                    story_html += f"""
                    <div class="email-item">
                        <div class="email-header">{email.get('subject', 'No Subject')}</div>
                        <div class="email-preview">{email.get('body_preview', '')}</div>
                        <div class="email-meta">
                            From: {email.get('from', 'Unknown')}<br>
                            To: {email.get('to', 'Unknown')}<br>
                            Date: {email.get('date', 'Unknown')}
                    """
                    if "matching_keyword" in email:
                        story_html += f"""
                            <br>Matching Keyword: <span class="email-keyword">{email['matching_keyword']}</span>
                        """
                    story_html += """
                        </div>
                    </div>
                    """
                story_html += "</div>"

            story_html += "</div>"
            html += story_html

        html += """
            </div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const searchBox = document.getElementById('searchBox');
                    const filterButtons = document.querySelectorAll('.filter-button');
                    const stories = document.querySelectorAll('.story');

                    // Search functionality
                    searchBox.addEventListener('input', function(e) {
                        const searchTerm = e.target.value.toLowerCase();
                        stories.forEach(story => {
                            const text = story.textContent.toLowerCase();
                            story.style.display = text.includes(searchTerm) ? 'block' : 'none';
                        });
                    });

                    // Filter functionality
                    filterButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            const filter = this.dataset.filter;

                            // Update active button
                            filterButtons.forEach(btn => btn.classList.remove('active'));
                            this.classList.add('active');

                            // Show/hide stories based on filter
                            stories.forEach(story => {
                                if (filter === 'all' || story.dataset.type === filter) {
                                    story.style.display = 'block';
                                } else {
                                    story.style.display = 'none';
                                }
                            });
                        });
                    });
                });
            </script>
        </body>
        </html>
        """

        return html

    def generate_story_summaries(
        self,
        df,
        analysis_results,
        key_actors,
        topic_evolution,
        significant_events,
        email_threads,
    ):
        """
        Generate comprehensive story summaries based on all available data.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results
            key_actors (dict): Dictionary containing key actors data
            topic_evolution (dict): Dictionary containing topic evolution data
            significant_events (list): List of significant events
            email_threads (dict): Dictionary of email threads

        Returns:
            list: List of story summaries
        """
        logger.info("Generating story summaries")
        stories = []

        # 1. Stories based on key actors
        if key_actors and "top_actors" in key_actors:
            for actor, metrics in list(key_actors["top_actors"].items())[:5]:
                # Get emails sent by this actor
                actor_emails = df[df["from"].str.contains(actor, na=False)]

                if len(actor_emails) > 0:
                    # Get related emails (emails in threads where actor participated)
                    related_emails = []
                    for _, email in actor_emails.iterrows():
                        thread_emails = df[df["subject"].str.contains(email["subject"], na=False)]
                        for _, thread_email in thread_emails.iterrows():
                            related_emails.append(
                                {
                                    "subject": thread_email["subject"],
                                    "date": thread_email["date"],
                                    "from": thread_email["from"],
                                    "to": thread_email["to"],
                                    "body_preview": thread_email["body"][:200] + "..."
                                    if len(thread_email["body"]) > 200
                                    else thread_email["body"],
                                }
                            )

                    # Use clean_body column if available, otherwise use body
                    text_column = "clean_body" if "clean_body" in actor_emails.columns else "body"
                    all_text = " ".join(actor_emails[text_column].fillna(""))
                    words = re.findall(r"\b\w+\b", all_text.lower())
                    words = [
                        word for word in words if word not in self.stop_words and len(word) > 2
                    ]
                    common_words = Counter(words).most_common(20)
                    sample_subjects = actor_emails["subject"].head(5).tolist()

                    # Calculate communication patterns
                    daily_patterns = actor_emails.groupby(actor_emails["date"].dt.day_name()).size()
                    busiest_day = daily_patterns.idxmax()
                    busiest_day_count = daily_patterns.max()

                    # Calculate average response time
                    response_times = []
                    for _, email in actor_emails.iterrows():
                        if pd.notna(email["date"]):
                            replies = df[
                                (df["subject"].str.contains(email["subject"], na=False))
                                & (df["date"] > email["date"])
                            ]
                            if not replies.empty:
                                response_time = (
                                    replies["date"].min() - email["date"]
                                ).total_seconds() / 3600
                                response_times.append(response_time)

                    avg_response_time = np.mean(response_times) if response_times else None

                    story = {
                        "title": f"The Story of {actor}",
                        "type": "key_actor",
                        "actor": actor,
                        "metrics": metrics,
                        "common_topics": common_words,
                        "sample_subjects": sample_subjects,
                        "communication_patterns": {
                            "busiest_day": busiest_day,
                            "busiest_day_count": int(busiest_day_count),
                            "avg_response_time": f"{avg_response_time:.1f} hours"
                            if avg_response_time
                            else "N/A",
                        },
                        "related_emails": related_emails,
                        "summary": self._generate_actor_summary(
                            actor,
                            metrics,
                            common_words,
                            busiest_day,
                            busiest_day_count,
                            avg_response_time,
                        ),
                    }
                    stories.append(story)

        # 2. Stories based on significant events
        for event in significant_events[:5]:
            event_date = event["date"]
            event_emails = df[
                (pd.to_datetime(df["date"].dt.date) >= pd.to_datetime(event_date))
                & (
                    pd.to_datetime(df["date"].dt.date)
                    < pd.to_datetime(event_date) + timedelta(days=1)
                )
            ]

            # Get all related emails (including replies and forwards)
            related_emails = []
            for _, email in event_emails.iterrows():
                # Get emails in the same thread
                thread_emails = df[df["subject"].str.contains(email["subject"], na=False)]
                for _, thread_email in thread_emails.iterrows():
                    related_emails.append(
                        {
                            "subject": thread_email["subject"],
                            "date": thread_email["date"],
                            "from": thread_email["from"],
                            "to": thread_email["to"],
                            "body_preview": thread_email["body"][:200] + "..."
                            if len(thread_email["body"]) > 200
                            else thread_email["body"],
                        }
                    )

            # Analyze event participants
            participants = set()
            for _, email in event_emails.iterrows():
                if pd.notna(email["from"]):
                    participants.update(re.findall(r"[\w\.-]+@[\w\.-]+", email["from"]))
                if pd.notna(email["to"]):
                    participants.update(re.findall(r"[\w\.-]+@[\w\.-]+", email["to"]))

            # Calculate event metrics
            event_metrics = {
                "participant_count": len(participants),
                "avg_email_length": event_emails["body"].str.len().mean(),
                "reply_rate": len(
                    event_emails[event_emails["subject"].str.contains("Re:", na=False)]
                )
                / len(event_emails),
            }

            story = {
                "title": f"Significant Event on {event_date}",
                "type": "significant_event",
                "date": event_date,
                "email_count": event["email_count"],
                "common_words": event["common_words"],
                "sample_subjects": event["sample_subjects"],
                "event_metrics": event_metrics,
                "participants": list(participants),
                "related_emails": related_emails,
                "summary": self._generate_event_summary(
                    event_date,
                    event["email_count"],
                    event["deviation"],
                    event["common_words"],
                    event_metrics,
                ),
            }
            stories.append(story)

        # 3. Stories based on email threads
        for subject, thread in list(email_threads.items())[:5]:
            if len(thread) >= 3:
                # Extract participants
                participants = set()
                for email in thread:
                    if "from" in email and email["from"]:
                        sender_emails = re.findall(r"[\w\.-]+@[\w\.-]+", email["from"])
                        participants.update(sender_emails)

                # Get all related emails
                related_emails = []
                for email in thread:
                    related_emails.append(
                        {
                            "subject": email.get("subject", subject),
                            "date": email.get("date"),
                            "from": email.get("from"),
                            "to": email.get("to"),
                            "body_preview": email.get("body", "")[:200] + "..."
                            if len(email.get("body", "")) > 200
                            else email.get("body", ""),
                        }
                    )

                # Calculate thread metrics
                thread_duration = None
                if thread[0]["date"] and thread[-1]["date"]:
                    thread_duration = (
                        thread[-1]["date"] - thread[0]["date"]
                    ).total_seconds() / 3600

                avg_response_time = None
                response_times = []
                for i in range(len(thread) - 1):
                    if thread[i]["date"] and thread[i + 1]["date"]:
                        response_time = (
                            thread[i + 1]["date"] - thread[i]["date"]
                        ).total_seconds() / 3600
                        response_times.append(response_time)
                if response_times:
                    avg_response_time = np.mean(response_times)

                story = {
                    "title": f"Email Thread: {subject}",
                    "type": "email_thread",
                    "subject": subject,
                    "num_emails": len(thread),
                    "participants": list(participants),
                    "start_date": thread[0]["date"] if "date" in thread[0] else None,
                    "end_date": thread[-1]["date"] if "date" in thread[-1] else None,
                    "thread_metrics": {
                        "duration_hours": f"{thread_duration:.1f}" if thread_duration else "N/A",
                        "avg_response_time": f"{avg_response_time:.1f} hours"
                        if avg_response_time
                        else "N/A",
                        "participant_count": len(participants),
                    },
                    "related_emails": related_emails,
                    "summary": self._generate_thread_summary(
                        subject,
                        len(thread),
                        len(participants),
                        thread[0]["date"] if "date" in thread[0] else None,
                        thread[-1]["date"] if "date" in thread[-1] else None,
                        thread_duration,
                        avg_response_time,
                    ),
                }
                stories.append(story)

        # 4. Stories based on topic evolution
        if topic_evolution and "topic_keywords" in topic_evolution:
            for topic_id, keywords in list(topic_evolution["topic_keywords"].items())[:5]:
                # Get emails related to this topic
                topic_emails = []
                for keyword in keywords[:5]:  # Use top 5 keywords
                    keyword_emails = df[df["body"].str.contains(keyword, case=False, na=False)]
                    for _, email in keyword_emails.iterrows():
                        topic_emails.append(
                            {
                                "subject": email["subject"],
                                "date": email["date"],
                                "from": email["from"],
                                "to": email["to"],
                                "body_preview": email["body"][:200] + "..."
                                if len(email["body"]) > 200
                                else email["body"],
                                "matching_keyword": keyword,
                            }
                        )

                # Extract topic number
                topic_num = int(topic_id.split()[-1])

                # Get topic evolution data
                topic_counts = topic_evolution.get("topic_counts", {})
                topic_trend = self._analyze_topic_trend(topic_counts, topic_num)

                story = {
                    "title": f"The Evolution of {topic_id}",
                    "type": "topic_evolution",
                    "topic_id": topic_id,
                    "keywords": keywords,
                    "topic_metrics": {
                        "trend": topic_trend["trend"],
                        "peak_period": topic_trend["peak_period"],
                        "peak_count": topic_trend["peak_count"],
                    },
                    "related_emails": topic_emails,
                    "summary": self._generate_topic_summary(
                        topic_id,
                        keywords,
                        topic_trend,
                    ),
                }
                stories.append(story)

        return stories

    def _generate_actor_summary(
        self,
        actor,
        metrics,
        common_words,
        busiest_day,
        busiest_day_count,
        avg_response_time,
    ):
        """Generate a summary for a key actor story."""
        # Create an engaging hook
        name = actor.split("@")[0].replace(".", " ").title()
        summary = f"Unveiling the Digital Footprint: The Enron Emails of {name}\n\n"

        # Add role and influence context
        influence_score = (
            metrics.get("degree_centrality", 0)
            + metrics.get("betweenness_centrality", 0)
            + metrics.get("pagerank", 0)
        ) / 3
        influence_level = (
            "highly influential"
            if influence_score > 0.1
            else "moderately influential"
            if influence_score > 0.05
            else "influential"
        )
        summary += f"In the intricate web of Enron's corporate communications, {name} emerges as a {influence_level} figure, "
        summary += (
            f"having sent {metrics['sent']} emails and received {metrics['received']} emails. "
        )

        # Add communication patterns with context
        summary += f"\n\nCommunication Patterns:\n"
        summary += f"• Peak Activity: {name} is most active on {busiest_day}s, sending {busiest_day_count} emails. "
        if avg_response_time:
            response_context = (
                "remarkably quick"
                if avg_response_time < 4
                else "moderate"
                if avg_response_time < 8
                else "deliberate"
            )
            summary += f"This suggests a {response_context} response pattern, with an average response time of {avg_response_time:.1f} hours. "

        # Add topic analysis with deeper context
        summary += f"\n\nKey Focus Areas:\n"
        top_topics = [word for word, _ in common_words[:5]]
        topic_counts = [count for _, count in common_words[:5]]
        topic_analysis = []
        for topic, count in zip(top_topics, topic_counts):
            if topic.lower() in ["energy", "power", "gas"]:
                topic_analysis.append(f"energy sector operations ({count} mentions)")
            elif topic.lower() in ["state", "regulatory", "policy"]:
                topic_analysis.append(f"regulatory affairs ({count} mentions)")
            elif topic.lower() in ["market", "trading", "price"]:
                topic_analysis.append(f"market activities ({count} mentions)")
            else:
                topic_analysis.append(f"{topic} ({count} mentions)")

        summary += f"• Primary Focus: {', '.join(topic_analysis[:-1])}, and {topic_analysis[-1]}. "

        # Add network analysis
        summary += f"\n\nNetwork Impact:\n"
        summary += f"• Influence Score: {influence_score:.3f} (combining centrality measures)\n"
        summary += f"• Communication Reach: {metrics.get('degree_centrality', 0):.3f} (direct connections)\n"
        summary += (
            f"• Information Flow: {metrics.get('betweenness_centrality', 0):.3f} (brokerage role)\n"
        )
        summary += (
            f"• Overall Importance: {metrics.get('pagerank', 0):.3f} (network-wide significance)"
        )

        # Add temporal analysis
        summary += f"\n\nTemporal Patterns:\n"
        if busiest_day_count > 10:
            summary += f"• High Activity: {name} shows intense engagement on {busiest_day}s, suggesting this day may be crucial for weekly operations or reporting.\n"
        if avg_response_time and avg_response_time < 4:
            summary += f"• Quick Response: The rapid response time indicates a key operational role or high-priority communications.\n"

        # Add concluding insights
        summary += f"\n\nKey Insights:\n"
        if influence_score > 0.1:
            summary += f"• {name} plays a central role in Enron's communication network, acting as a key information hub.\n"
        if any("energy" in topic.lower() for topic in top_topics):
            summary += f"• Strong focus on energy sector operations suggests involvement in core business activities.\n"
        if avg_response_time and avg_response_time < 4:
            summary += (
                f"• Quick response times indicate a hands-on role in critical communications.\n"
            )

        return summary

    def _generate_event_summary(
        self,
        date,
        email_count,
        deviation,
        common_words,
        event_metrics,
    ):
        """Generate a summary for a significant event story."""
        # Create an engaging hook
        summary = f"Unusual Activity Detected: Email Surge on {date}\n\n"

        # Add event significance
        significance_level = (
            "extremely significant"
            if deviation > 3
            else "highly significant"
            if deviation > 2
            else "significant"
        )
        summary += f"A {significance_level} spike in email activity was detected, with {email_count} emails exchanged "
        summary += f"({deviation:.1f} standard deviations above normal). "

        # Add participant analysis
        summary += f"\n\nParticipant Analysis:\n"
        summary += f"• Scale: {event_metrics['participant_count']} individuals were involved in this communication surge\n"
        summary += f"• Engagement: Average email length of {event_metrics['avg_email_length']:.0f} characters suggests "
        summary += f"{'detailed discussions' if event_metrics['avg_email_length'] > 500 else 'brief exchanges'}\n"
        summary += f"• Interaction: Reply rate of {event_metrics['reply_rate']:.1%} indicates "
        summary += f"{'highly interactive' if event_metrics['reply_rate'] > 0.5 else 'moderate'} communication patterns"

        # Add topic analysis
        summary += f"\n\nKey Topics:\n"
        top_words = [word for word, _ in common_words[:5]]
        word_counts = [count for _, count in common_words[:5]]
        topic_analysis = []
        for word, count in zip(top_words, word_counts):
            if word.lower() in ["urgent", "emergency", "critical"]:
                topic_analysis.append(f"urgent matters ({count} mentions)")
            elif word.lower() in ["meeting", "conference", "call"]:
                topic_analysis.append(f"coordination activities ({count} mentions)")
            elif word.lower() in ["report", "update", "status"]:
                topic_analysis.append(f"status updates ({count} mentions)")
            else:
                topic_analysis.append(f"{word} ({count} mentions)")

        summary += f"• Primary Focus: {', '.join(topic_analysis[:-1])}, and {topic_analysis[-1]}\n"

        # Add temporal context
        summary += f"\n\nTemporal Context:\n"
        if deviation > 3:
            summary += f"• Exceptional Activity: This spike represents one of the most significant communication events in the dataset\n"
        if event_metrics["reply_rate"] > 0.7:
            summary += f"• Rapid Response: The high reply rate suggests urgent or time-sensitive matters were being discussed\n"
        if event_metrics["avg_email_length"] > 1000:
            summary += f"• Detailed Communication: The lengthy emails indicate complex or important discussions\n"

        # Add potential implications
        summary += f"\n\nPotential Implications:\n"
        if any(word.lower() in ["urgent", "emergency", "critical"] for word in top_words):
            summary += f"• This event may represent a critical business situation requiring immediate attention\n"
        if event_metrics["participant_count"] > 10:
            summary += f"• The large number of participants suggests a company-wide or department-wide communication event\n"
        if event_metrics["reply_rate"] > 0.5:
            summary += f"• The high level of interaction indicates active problem-solving or decision-making\n"

        return summary

    def _generate_thread_summary(
        self,
        subject,
        num_emails,
        num_participants,
        start_date,
        end_date,
        duration,
        avg_response_time,
    ):
        """Generate a summary for an email thread story."""
        # Create an engaging hook
        summary = f"Thread Analysis: '{subject}'\n\n"

        # Add thread overview
        thread_size = (
            "extensive" if num_emails > 10 else "substantial" if num_emails > 5 else "moderate"
        )
        summary += f"An {thread_size} email thread involving {num_participants} participants "
        summary += f"unfolded over {num_emails} messages. "

        # Add temporal analysis
        if start_date and end_date:
            summary += f"\n\nTemporal Analysis:\n"
            summary += f"• Duration: {duration:.1f} hours from {start_date} to {end_date}\n"
            if duration:
                pace = "rapid" if duration < 24 else "moderate" if duration < 72 else "extended"
                summary += f"• Pace: {pace} discussion pace\n"
            if avg_response_time:
                response_character = (
                    "immediate"
                    if avg_response_time < 2
                    else "prompt"
                    if avg_response_time < 6
                    else "deliberate"
                )
                summary += f"• Response Pattern: {response_character} responses (avg. {avg_response_time:.1f} hours)\n"

        # Add participant analysis
        summary += f"\n\nParticipant Dynamics:\n"
        if num_participants > 5:
            summary += (
                f"• Scale: Large group discussion involving {num_participants} participants\n"
            )
        elif num_participants > 2:
            summary += f"• Scale: Multi-party conversation with {num_participants} participants\n"
        else:
            summary += f"• Scale: Direct exchange between {num_participants} participants\n"

        # Add thread characteristics
        summary += f"\n\nThread Characteristics:\n"
        if num_emails > 10:
            summary += f"• Depth: Extensive discussion with multiple sub-threads\n"
        if avg_response_time and avg_response_time < 4:
            summary += f"• Urgency: Quick response times suggest time-sensitive matters\n"
        if duration and duration > 72:
            summary += f"• Persistence: Extended duration indicates ongoing or complex discussion\n"

        # Add potential implications
        summary += f"\n\nPotential Significance:\n"
        if num_participants > 5 and avg_response_time and avg_response_time < 4:
            summary += f"• This thread may represent a critical decision-making process or urgent coordination effort\n"
        if duration and duration > 72:
            summary += f"• The extended duration suggests complex or ongoing business matters\n"
        if num_emails > 10:
            summary += f"• The high message count indicates detailed discussion or multiple decision points\n"

        return summary

    def _generate_topic_summary(
        self,
        topic_id,
        keywords,
        topic_trend,
    ):
        """Generate a summary for a topic evolution story."""
        # Create an engaging hook
        summary = f"Topic Evolution: {topic_id}\n\n"

        # Add topic overview
        summary += f"Analysis of communication patterns reveals the evolution of {topic_id}, "
        summary += f"characterized by keywords such as {', '.join(keywords[:3])}. "

        # Add trend analysis
        summary += f"\n\nTrend Analysis:\n"
        trend_character = {
            "increasing": "growing",
            "decreasing": "declining",
            "stable": "consistent",
        }.get(topic_trend["trend"], "variable")

        summary += f"• Overall Trend: {trend_character} interest in this topic\n"
        if topic_trend["peak_period"]:
            summary += f"• Peak Activity: {topic_trend['peak_count']} mentions during {topic_trend['peak_period']}\n"

        # Add keyword analysis
        summary += f"\n\nKeyword Analysis:\n"
        keyword_categories = {
            "energy": "energy sector",
            "power": "power operations",
            "gas": "gas operations",
            "market": "market activities",
            "trading": "trading operations",
            "price": "pricing",
            "state": "regulatory affairs",
            "regulatory": "regulatory matters",
            "policy": "policy issues",
        }

        categorized_keywords = []
        for keyword in keywords[:5]:
            category = next(
                (v for k, v in keyword_categories.items() if k in keyword.lower()), None
            )
            if category:
                categorized_keywords.append(f"{category} ({keyword})")
            else:
                categorized_keywords.append(keyword)

        summary += f"• Primary Focus: {', '.join(categorized_keywords[:-1])}, and {categorized_keywords[-1]}\n"

        # Add temporal patterns
        summary += f"\n\nTemporal Patterns:\n"
        if topic_trend["trend"] == "increasing":
            summary += f"• Growing Interest: The topic shows increasing relevance over time\n"
        elif topic_trend["trend"] == "decreasing":
            summary += f"• Declining Focus: The topic shows decreasing prominence\n"
        else:
            summary += f"• Stable Presence: The topic maintains consistent attention\n"

        # Add potential implications
        summary += f"\n\nPotential Implications:\n"
        if topic_trend["trend"] == "increasing":
            summary += f"• This topic may represent an emerging area of focus or concern\n"
        if topic_trend["peak_count"] > 50:
            summary += f"• The high peak activity suggests significant business impact\n"
        if any(kw.lower() in ["urgent", "critical", "emergency"] for kw in keywords):
            summary += f"• The presence of urgent keywords indicates time-sensitive matters\n"

        return summary

    def _analyze_topic_trend(self, topic_counts, topic_num):
        """Analyze the trend of a topic over time."""
        if not topic_counts:
            return {"trend": "unknown", "peak_period": None, "peak_count": 0}

        # Extract counts for this topic
        topic_data = {period: counts.get(topic_num, 0) for period, counts in topic_counts.items()}

        if not topic_data:
            return {"trend": "unknown", "peak_period": None, "peak_count": 0}

        # Find peak period
        peak_period = max(topic_data.items(), key=lambda x: x[1])

        # Determine trend
        values = list(topic_data.values())
        if len(values) < 2:
            trend = "stable"
        else:
            slope = np.polyfit(range(len(values)), values, 1)[0]
            if slope > 0.1:
                trend = "increasing"
            elif slope < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"

        return {
            "trend": trend,
            "peak_period": peak_period[0],
            "peak_count": peak_period[1],
        }


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
