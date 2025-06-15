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
            dict: Dictionary containing comprehensive story development results

        Raises:
            StoryGenerationError: If story generation fails
            DataLoadingError: If data cannot be loaded
            AnalysisError: If analysis fails
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

            # Set up parallel processing
            if max_workers is None:
                max_workers = min(32, os.cpu_count() + 4)
            logger.info(f"Using {max_workers} worker threads for parallel processing")

            # Generate comprehensive story with parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks for parallel execution
                future_to_task = {
                    executor.submit(self.identify_key_actors, df): "key_actors",
                    executor.submit(
                        self.track_topics_over_time, df, analysis_results
                    ): "topic_evolution",
                    executor.submit(self.detect_significant_events, df): "significant_events",
                    executor.submit(self.construct_email_threads, df): "email_threads",
                    executor.submit(
                        self.analyze_email_connections, df, analysis_results
                    ): "email_connections",
                }

                # Initialize results dictionary
                results = {}

                # Process completed tasks with progress bar
                with tqdm(total=len(future_to_task), desc="Generating story components") as pbar:
                    for future in as_completed(future_to_task):
                        task_name = future_to_task[future]
                        try:
                            results[task_name] = future.result()
                            if results[task_name] is None:
                                raise AnalysisError(f"Task {task_name} returned None")
                            logger.info(f"Completed {task_name} analysis")
                        except Exception as e:
                            logger.error(f"Error in {task_name} analysis: {e}")
                            raise AnalysisError(f"Failed to complete {task_name} analysis: {e}")
                        pbar.update(1)

            # Generate a comprehensive story
            try:
                logger.info("Formatting story components...")
                story = {
                    "metadata": {
                        "generation_date": datetime.now().isoformat(),
                        "total_emails": len(df),
                        "time_span": {
                            "start": df["date"].min().isoformat(),
                            "end": df["date"].max().isoformat(),
                        },
                    },
                    "key_actors": self._format_key_actors_story(results["key_actors"]),
                    "topic_evolution": self._format_topic_evolution_story(
                        results["topic_evolution"]
                    ),
                    "significant_events": self._format_events_story(results["significant_events"]),
                    "email_threads": self._format_threads_story(results["email_threads"]),
                    "email_connections": self._format_connections_story(
                        results["email_connections"]
                    ),
                    "narrative_summary": self._generate_narrative_summary(
                        results["key_actors"],
                        results["topic_evolution"],
                        results["significant_events"],
                        results["email_threads"],
                    ),
                }

                # Add enhanced summaries and visualization insights in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_summaries = executor.submit(
                        self._generate_enhanced_summaries, df, analysis_results
                    )
                    future_insights = executor.submit(
                        self._generate_visualization_insights, analysis_results
                    )

                    story["enhanced_summaries"] = future_summaries.result()
                    story["visualization_insights"] = future_insights.result()

                # Save story
                logger.info("Saving stories...")
                stories_path = self.save_stories([story])

                logger.info("Story development completed successfully!")

                return {"story": story, "stories_path": stories_path}
            except Exception as e:
                raise StoryGenerationError(f"Failed to generate story: {e}")

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

            return json_path
        except Exception as e:
            if isinstance(e, (FileOperationError, DatabaseError)):
                raise
            raise FileOperationError(f"Unexpected error saving stories: {e}")

    @staticmethod
    def _generate_html_report(stories):
        """
        Generate an HTML report from the stories.

        Args:
            stories (list): List of story summaries

        Returns:
            str: HTML content

        Raises:
            FileOperationError: If HTML generation fails
        """
        try:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enron Email Stories</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    .story {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                    .story-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
                    .story-summary {{ margin-bottom: 15px; }}
                    .story-details {{ font-size: 0.9em; color: #555; }}
                    .key-actor {{ background-color: #e8f4f8; }}
                    .significant-event {{ background-color: #f8f4e8; }}
                    .email-thread {{ background-color: #f4f8e8; }}
                    .topic-evolution {{ background-color: #f8e8f4; }}
                    .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .list-item {{ margin: 5px 0; }}
                    .highlight {{ background-color: #fff3cd; padding: 2px 5px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <h1>Enron Email Stories</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            """

            if not stories:
                logger.warning("No stories provided for HTML report generation")
                return html + "</body></html>"

            for story in stories:
                try:
                    # Add metadata section
                    if "metadata" in story:
                        html += """
                        <div class="metadata">
                            <h2>Metadata</h2>
                            <p>Total Emails: {}</p>
                            <p>Time Span: {} to {}</p>
                        </div>
                        """.format(
                            story["metadata"].get("total_emails", "N/A"),
                            story["metadata"].get("time_span", {}).get("start", "N/A"),
                            story["metadata"].get("time_span", {}).get("end", "N/A"),
                        )

                    # Add key actors section
                    if "key_actors" in story and story["key_actors"]:
                        html += """
                        <div class="section">
                            <h2>Key Actors</h2>
                        """
                        for actor in story["key_actors"].get("main_characters", []):
                            try:
                                html += """
                                <div class="story key-actor">
                                    <div class="story-title">{}</div>
                                    <div class="story-details">
                                        <p>Role: {}</p>
                                        <p>Influence Score: {:.2f}</p>
                                        <p>Metrics:</p>
                                        <ul>
                                            <li>Emails Sent: {}</li>
                                            <li>Emails Received: {}</li>
                                            <li>Degree Centrality: {:.2f}</li>
                                            <li>Betweenness Centrality: {:.2f}</li>
                                            <li>PageRank: {:.2f}</li>
                                        </ul>
                                    </div>
                                </div>
                                """.format(
                                    actor.get("name", "Unknown"),
                                    actor.get("role", "Unknown"),
                                    actor.get("influence_score", 0),
                                    actor.get("metrics", {}).get("sent", 0),
                                    actor.get("metrics", {}).get("received", 0),
                                    actor.get("metrics", {}).get("degree_centrality", 0),
                                    actor.get("metrics", {}).get("betweenness_centrality", 0),
                                    actor.get("metrics", {}).get("pagerank", 0),
                                )
                            except Exception as e:
                                logger.error(f"Error formatting actor data: {e}")
                                continue
                        html += "</div>"

                    # Add significant events section
                    if "significant_events" in story and story["significant_events"]:
                        html += """
                        <div class="section">
                            <h2>Significant Events</h2>
                        """
                        for event in story["significant_events"].get("events", []):
                            try:
                                html += """
                                <div class="story significant-event">
                                    <div class="story-title">Event on {}</div>
                                    <div class="story-details">
                                        <p>Significance: {:.2f}</p>
                                        <p>{}</p>
                                        <p>Key Topics: {}</p>
                                        <p>Sample Subjects:</p>
                                        <ul>
                                """.format(
                                    event.get("date", "Unknown"),
                                    event.get("significance", 0),
                                    event.get("description", "No description available"),
                                    ", ".join(event.get("key_topics", [])),
                                )
                                for subject in event.get("sample_subjects", []):
                                    html += f"<li>{subject}</li>"
                                html += """
                                        </ul>
                                    </div>
                                </div>
                                """
                            except Exception as e:
                                logger.error(f"Error formatting event data: {e}")
                                continue
                        html += "</div>"

                    # Add email threads section
                    if "email_threads" in story and story["email_threads"]:
                        html += """
                        <div class="section">
                            <h2>Email Threads</h2>
                        """
                        for thread in story["email_threads"].get("threads", []):
                            try:
                                html += """
                                <div class="story email-thread">
                                    <div class="story-title">{}</div>
                                    <div class="story-details">
                                        <p>Size: {} emails</p>
                                        <p>Participants: {}</p>
                                        <p>Time Span: {} to {}</p>
                                    </div>
                                </div>
                                """.format(
                                    thread.get("subject", "Unknown"),
                                    thread.get("size", 0),
                                    ", ".join(thread.get("participants", [])[:5])
                                    + ("..." if len(thread.get("participants", [])) > 5 else ""),
                                    thread.get("timeline", {}).get("start", "Unknown"),
                                    thread.get("timeline", {}).get("end", "Unknown"),
                                )
                            except Exception as e:
                                logger.error(f"Error formatting thread data: {e}")
                                continue
                        html += "</div>"

                    # Add topic evolution section
                    if "topic_evolution" in story and story["topic_evolution"]:
                        html += """
                        <div class="section">
                            <h2>Topic Evolution</h2>
                        """
                        for topic in story["topic_evolution"].get("key_topics", []):
                            try:
                                html += """
                                <div class="story topic-evolution">
                                    <div class="story-title">Topic {}</div>
                                    <div class="story-details">
                                        <p>Keywords: {}</p>
                                        <p>Significance: {}</p>
                                    </div>
                                </div>
                                """.format(
                                    topic.get("id", "Unknown"),
                                    ", ".join(topic.get("keywords", [])),
                                    topic.get("significance", 0),
                                )
                            except Exception as e:
                                logger.error(f"Error formatting topic data: {e}")
                                continue
                        html += "</div>"

                    # Add narrative summary
                    if "narrative_summary" in story:
                        try:
                            html += """
                            <div class="section">
                                <h2>Narrative Summary</h2>
                                <div class="story">
                                    <div class="story-summary">{}</div>
                                </div>
                            </div>
                            """.format(
                                story["narrative_summary"]
                            )
                        except Exception as e:
                            logger.error(f"Error formatting narrative summary: {e}")

                except Exception as e:
                    logger.error(f"Error processing story section: {e}")
                    continue

            html += """
            </body>
            </html>
            """

            return html

        except Exception as e:
            logger.error(e)
            logger.error(f"Error generating HTML report: {e}")


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
