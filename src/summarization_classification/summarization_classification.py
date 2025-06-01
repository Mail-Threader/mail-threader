import concurrent.futures  # Import for threading
import os
import pickle
import shutil  # For deleting temporary directories
import tempfile  # For temporary files
from collections import Counter
from datetime import datetime
from typing import Optional

import nltk
import numpy as np
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Import tqdm
from transformers import pipeline


class SummarizationClassification:
    """
    Class responsible for summarization, classification, and categorization tasks:
    - Text preprocessing (tokenization, lemmatization)
    - Topic modeling
    - Email clustering
    - Entity recognitionE
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

        # Create an output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize NLTK components
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Initialize transformers pipelines
        try:
            logger.info("Loading summarization pipeline...")
            # Use device=-1 for CPU, or 0 for GPU if available and desired
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
            logger.info("Loading sentiment analysis pipeline...")
            self.sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
            logger.info("Loading named entity recognition pipeline...")
            self.ner = pipeline(
                "ner", aggregation_strategy="simple", device=-1
            )  # Use aggregation_strategy for NER
            logger.info("Transformers pipelines loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing transformers pipelines: {e}")
            self.summarizer = None
            self.sentiment_analyzer = None
            self.ner = None

    def load_data(self, file_path: Optional[str] = None, skip=False):
        """
        Load processed email data from a file.

        Args:
            file_path (str, optional): Path to the processed data file.
                If not provided, the most recent file in input_dir will be used.
            skip (bool): Whether to skip loading data if a file is found in output_dir.

        Returns:
            pandas.DataFrame: DataFrame containing the processed email data
        """

        if file_path is None:
            # Find the most recent processed data file
            search_dir = self.output_dir if skip else self.input_dir
            search_file_pattern = "analysis_results" if skip else "processed_data_"
            pkl_files = [
                f
                for f in os.listdir(search_dir)
                if f.startswith(search_file_pattern) and f.endswith(".pkl")
            ]
            json_files = [
                f
                for f in os.listdir(search_dir)
                if f.startswith(search_file_pattern) and f.endswith(".json")
            ]
            logger.info(f"Found {len(pkl_files)} processed data files in {search_dir}")

        try:
            if pkl_files:
                # Sort by timestamp in the files
                pkl_files.sort(reverse=True)
                file_path = os.path.join(search_dir, pkl_files[0])
                df = pd.DataFrame(pd.read_pickle(file_path))
                logger.info(f"Loaded data from {file_path}: {len(df)} emails")
                return df

            if json_files:
                # Sort by timestamp in the files
                json_files.sort(reverse=True)
                file_path = os.path.join(search_dir, json_files[0])
                df = pd.DataFrame(pd.read_json(file_path))
                logger.info(f"Loaded data from {file_path}: {len(df)} emails")
                return df

            logger.error(f"No processed data files found in {self.input_dir}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_text(self, text):
        """
        Preprocess text for NLP tasks.

        Args:
            text (str): Text to preprocess

        Returns:
            list: List of preprocessed tokens
        """
        if not isinstance(text, str) or not text:
            return []

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    @staticmethod
    def extract_topics(df, n_topics=10, method="lda"):
        """
        Extract topics from email content using topic modeling.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            n_topics (int): Number of topics to extract
            method (str): Topic modeling method ('lda' or 'nmf')

        Returns:
            dict: Dictionary containing topic model and topic keywords
        """
        logger.info(f"Extracting {n_topics} topics using {method}")

        # Use clean_body column if available, otherwise use body
        text_column = "clean_body" if "clean_body" in df.columns else "body"

        # Create document-term matrix
        if method == "nmf":
            # For NMF, use TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=5000, min_df=5, max_df=0.8, stop_words="english"
            )
        else:
            # For LDA, use CountVectorizer
            vectorizer = CountVectorizer(
                max_features=5000, min_df=5, max_df=0.8, stop_words="english"
            )

        try:
            # Fit vectorizer and transform documents
            logger.info("Vectorizing text for topic modeling...")
            dtm = vectorizer.fit_transform(df[text_column].fillna(""))
            feature_names = vectorizer.get_feature_names_out()
            logger.info("Text vectorization complete.")

            # Create and fit a topic model
            if method == "nmf":
                model = NMF(n_components=n_topics, random_state=42)
            else:
                model = LatentDirichletAllocation(
                    n_components=n_topics, random_state=42, max_iter=10
                )

            # Fit the model - this is often the most time-consuming part
            logger.info(f"Fitting {method.upper()} model...")
            model.fit(dtm)
            logger.info(f"{method.upper()} model fitting complete.")

            # Extract the top keywords for each topic
            topics = {}
            for topic_idx, topic in tqdm(
                enumerate(model.components_), total=n_topics, desc="Extracting topic keywords"
            ):
                top_keywords_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 keywords
                top_keywords = [feature_names[i] for i in top_keywords_idx]
                topics[f"Topic {topic_idx + 1}"] = top_keywords

            # Assign topics to documents
            doc_topic_matrix = model.transform(dtm)
            df["dominant_topic"] = np.argmax(doc_topic_matrix, axis=1) + 1

            # Save topic model results
            result = {
                "model": model,
                "vectorizer": vectorizer,
                "topics": topics,
                "doc_topic_matrix": doc_topic_matrix,
            }

            return result

        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return None

    def cluster_emails(self, df, n_clusters=None, method="kmeans"):
        """
        Cluster emails based on their content.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            n_clusters (int, optional): Number of clusters. If None, will be determined automatically.
            method (str): Clustering method ('kmeans' or 'dbscan')

        Returns:
            dict: Dictionary containing clustering results
        """
        logger.info(f"Clustering emails using {method}")

        # Use clean_body column if available, otherwise use body
        text_column = "clean_body" if "clean_body" in df.columns else "body"

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.8, stop_words="english")

        try:
            # Fit vectorizer and transform documents
            logger.info("Vectorizing text for clustering...")
            tfidf_matrix = vectorizer.fit_transform(df[text_column].fillna(""))
            logger.info("Text vectorization complete.")

            # Determine the optimal number of clusters if not provided
            if method == "kmeans" and n_clusters is None:
                # Try different numbers of clusters and evaluate using silhouette score
                silhouette_scores = []
                range_n_clusters = range(2, min(20, len(df) // 10))
                for n in tqdm(range_n_clusters, desc="Finding optimal KMeans clusters"):
                    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(tfidf_matrix)
                    if len(np.unique(cluster_labels)) > 1 and tfidf_matrix.shape[0] > 1:
                        silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
                        silhouette_scores.append(silhouette_avg)
                    else:
                        silhouette_scores.append(-1)

                if silhouette_scores:
                    n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
                    logger.info(f"Optimal number of clusters determined: {n_clusters}")
                else:
                    n_clusters = 5
                    logger.warning("Could not determine optimal clusters, defaulting to 5.")

            # Perform clustering
            if method == "kmeans":
                n_clusters = n_clusters or 5
                logger.info(f"Performing KMeans clustering with {n_clusters} clusters...")
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df["cluster"] = model.fit_predict(tfidf_matrix)
                logger.info("KMeans clustering complete.")
            else:  # DBSCAN
                logger.info("Performing DBSCAN clustering...")
                scaler = StandardScaler(with_mean=False)
                scaled_tfidf = scaler.fit_transform(tfidf_matrix)

                model = DBSCAN(eps=0.5, min_samples=5)
                df["cluster"] = model.fit_predict(scaled_tfidf)
                logger.info("DBSCAN clustering complete.")

            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in tqdm(sorted(df["cluster"].unique()), desc="Analyzing clusters"):
                cluster_emails = df[df["cluster"] == cluster_id]

                if len(cluster_emails) > 0:
                    all_text = " ".join(cluster_emails[text_column].fillna(""))
                    tokens = self.preprocess_text(all_text)
                    most_common = Counter(tokens).most_common(10)

                    cluster_analysis[f"Cluster {cluster_id}"] = {
                        "size": len(cluster_emails),
                        "common_words": [word for word, count in most_common],
                        "sample_subjects": cluster_emails["subject"].head(5).tolist(),
                    }

            # Save clustering results
            result = {
                "model": model,
                "vectorizer": vectorizer,
                "cluster_analysis": cluster_analysis,
            }

            return result

        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return None

    def _process_entity_batch(self, batch_texts):
        """
        Processes a batch of texts for entity extraction using the NER pipeline.
        The transformers pipeline itself handles internal parallelism for the batch.
        """
        if self.ner is None:
            logger.warning("NER pipeline not initialized.")
            return [[] for _ in batch_texts]  # Return empty lists for each text

        # The pipeline handles batching internally when given a list of texts
        ner_results_batch = self.ner(batch_texts)

        processed_entities = []
        for ner_results_for_text in ner_results_batch:
            row_entities = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}
            for entity in ner_results_for_text:
                entity_type = (
                    entity["entity_group"] if "entity_group" in entity else entity["entity"]
                )
                if entity_type.startswith("B-") or entity_type.startswith("I-"):
                    entity_type = entity_type[2:]

                if entity_type in row_entities:
                    row_entities[entity_type].append(entity["word"])
                else:
                    row_entities["MISC"].append(entity["word"])
            processed_entities.append(row_entities)
        return processed_entities

    def extract_named_entities(self, df, use_sample=True, sample_size=100, batch_size=1000):
        """
        Extract named entities from email content, with batch processing for memory efficiency.
        The transformers pipeline handles internal parallelism within each batch.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            sample_size (int): Number of emails to sample for entity extraction (if use_sample is True)
            batch_size (int): Number of rows to process in each batch.

        Returns:
            dict: Dictionary containing extracted entities
        """
        if self.ner is None:
            logger.warning("Named entity recognition not available")
            return None

        logger.info(f"Extracting named entities (batch processing) with batch size {batch_size}.")

        text_column = "clean_body" if "clean_body" in df.columns else "body"

        if use_sample:
            if len(df) > sample_size:
                df_to_process = df.sample(sample_size, random_state=42)
            else:
                df_to_process = df
        else:
            df_to_process = df

        total_rows = len(df_to_process)
        all_entities_temp_dir = tempfile.mkdtemp(prefix="ner_temp_")
        logger.info(f"Created temporary directory for NER results: {all_entities_temp_dir}")

        try:
            for i in tqdm(range(0, total_rows, batch_size), desc="Processing NER batches"):
                batch_df = df_to_process.iloc[i : i + batch_size]
                # Prepare texts for the pipeline, limiting length
                batch_texts = [str(text)[:1000] for text in batch_df[text_column].fillna("")]
                # The _process_entity_batch function now directly calls the pipeline with the batch_texts
                batch_results = self._process_entity_batch(batch_texts)

                # Save batch result to a temporary file
                temp_file_path = os.path.join(all_entities_temp_dir, f"ner_batch_{i}.pkl")
                with open(temp_file_path, "wb") as f:
                    pickle.dump(batch_results, f)  # Save list of dicts
                del batch_df  # Free up memory
                del batch_texts
                del batch_results  # Free up memory

            # Aggregate results from temporary files
            final_entities = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}
            temp_files = [
                f
                for f in os.listdir(all_entities_temp_dir)
                if f.startswith("ner_batch_") and f.endswith(".pkl")
            ]
            for temp_file in tqdm(temp_files, desc="Aggregating NER results"):
                file_path = os.path.join(all_entities_temp_dir, temp_file)
                with open(file_path, "rb") as f:
                    batch_data = pickle.load(f)  # This is a list of row_entities dicts
                    for row_entities in batch_data:
                        for entity_type, entity_list in row_entities.items():
                            final_entities[entity_type].extend(entity_list)

            # Count entity frequencies
            entity_counts = {}
            for entity_type, entity_list in final_entities.items():
                entity_counts[entity_type] = Counter(entity_list).most_common(20)

            return entity_counts

        except Exception as e:
            logger.error(f"Error in named entity extraction: {e}")
            return None
        finally:
            # Clean up temporary directory
            if os.path.exists(all_entities_temp_dir):
                shutil.rmtree(all_entities_temp_dir)
                logger.info(f"Cleaned up temporary directory: {all_entities_temp_dir}")

    def _process_sentiment_batch(self, batch_texts, batch_subjects):
        """
        Processes a batch of texts for sentiment analysis using the pipeline.
        The transformers pipeline itself handles internal parallelism for the batch.
        """
        batch_sentiments = []
        if self.sentiment_analyzer is None:
            logger.warning("Sentiment analysis pipeline not initialized.")
            return batch_sentiments

        sentiment_results_batch = self.sentiment_analyzer(batch_texts)

        for idx, sentiment_result in enumerate(sentiment_results_batch):
            batch_sentiments.append(
                {
                    "subject": batch_subjects[idx],
                    "label": sentiment_result["label"],
                    "score": sentiment_result["score"],
                }
            )
        return batch_sentiments

    def analyze_sentiment(self, df, use_sample=True, sample_size=100, batch_size=1000):
        """
        Analyze sentiment of email content, with batch processing for memory efficiency.
        The transformers pipeline handles internal parallelism within each batch.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            sample_size (int): Number of emails to sample for sentiment analysis (if use_sample is True)
            batch_size (int): Number of rows to process in each batch.

        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        if self.sentiment_analyzer is None:
            logger.warning("Sentiment analysis not available")
            return None

        logger.info(f"Analyzing sentiment (batch processing) with batch size {batch_size}.")

        text_column = "clean_body" if "clean_body" in df.columns else "body"
        subject_column = "subject"

        if use_sample:
            if len(df) > sample_size:
                df_to_process = df.sample(sample_size, random_state=42)
            else:
                df_to_process = df
        else:
            df_to_process = df

        total_rows = len(df_to_process)
        all_sentiments_temp_dir = tempfile.mkdtemp(prefix="sentiment_temp_")
        logger.info(f"Created temporary directory for sentiment results: {all_sentiments_temp_dir}")

        try:
            for i in tqdm(range(0, total_rows, batch_size), desc="Processing sentiment batches"):
                batch_df = df_to_process.iloc[i : i + batch_size]
                batch_texts = [str(text)[:1000] for text in batch_df[text_column].fillna("")]
                batch_subjects = batch_df[subject_column].tolist()
                # The _process_sentiment_batch function now directly calls the pipeline with the batch_texts
                batch_results = self._process_sentiment_batch(batch_texts, batch_subjects)

                temp_file_path = os.path.join(all_sentiments_temp_dir, f"sentiment_batch_{i}.pkl")
                with open(temp_file_path, "wb") as f:
                    pickle.dump(batch_results, f)
                del batch_df
                del batch_texts
                del batch_subjects
                del batch_results

            final_sentiments = []
            temp_files = [
                f
                for f in os.listdir(all_sentiments_temp_dir)
                if f.startswith("sentiment_batch_") and f.endswith(".pkl")
            ]
            for temp_file in tqdm(temp_files, desc="Aggregating sentiment results"):
                file_path = os.path.join(all_sentiments_temp_dir, temp_file)
                with open(file_path, "rb") as f:
                    batch_data = pickle.load(f)
                    final_sentiments.extend(batch_data)

            sentiment_distribution = Counter([s["label"] for s in final_sentiments])

            return {"sentiments": final_sentiments, "distribution": sentiment_distribution}

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
        finally:
            if os.path.exists(all_sentiments_temp_dir):
                shutil.rmtree(all_sentiments_temp_dir)
                logger.info(f"Cleaned up temporary directory: {all_sentiments_temp_dir}")

    def _process_summary_batch(self, batch_texts, batch_subjects, max_length):
        """
        Processes a batch of texts for summarization using the pipeline.
        The transformers pipeline itself handles internal parallelism for the batch.
        """
        batch_summaries = []
        if self.summarizer is None:
            logger.warning("Summarization pipeline not initialized.")
            return batch_summaries

        summary_results_batch = self.summarizer(
            batch_texts, max_length=max_length, min_length=30, do_sample=False
        )

        for idx, summary_result in enumerate(summary_results_batch):
            batch_summaries.append(
                {
                    "subject": batch_subjects[idx],
                    "original_length": len(batch_texts[idx]),
                    "summary": summary_result["summary_text"],
                }
            )
        return batch_summaries

    def generate_summaries(
        self, df, use_sample=True, sample_size=20, max_length=50, batch_size=1000
    ):
        """
        Generate summaries of email content, with batch processing for memory efficiency.
        The transformers pipeline handles internal parallelism within each batch.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            sample_size (int): Number of emails to sample for summarization (if use_sample is True)
            max_length (int): Maximum length of generated summaries
            batch_size (int): Number of rows to process in each batch.

        Returns:
            list: List of dictionaries containing email subjects and summaries
        """
        if self.summarizer is None:
            logger.warning("Text summarization not available")
            return None

        logger.info(f"Generating summaries (batch processing) with batch size {batch_size}.")

        text_column = "body"
        subject_column = "subject"

        df_for_summary = df[df[text_column].str.len() > 200].copy()

        if use_sample:
            if len(df_for_summary) > sample_size:
                df_to_process = df_for_summary.sample(sample_size, random_state=42)
            else:
                df_to_process = df_for_summary
        else:
            df_to_process = df_for_summary

        total_rows = len(df_to_process)
        all_summaries_temp_dir = tempfile.mkdtemp(prefix="summary_temp_")
        logger.info(
            f"Created temporary directory for summarization results: {all_summaries_temp_dir}"
        )

        try:
            for i in tqdm(range(0, total_rows, batch_size), desc="Processing summary batches"):
                batch_df = df_to_process.iloc[i : i + batch_size]
                batch_texts = [str(text)[:1000] for text in batch_df[text_column].fillna("")]
                batch_subjects = batch_df[subject_column].tolist()
                # The _process_summary_batch function now directly calls the pipeline with the batch_texts
                batch_results = self._process_summary_batch(batch_texts, batch_subjects, max_length)

                temp_file_path = os.path.join(all_summaries_temp_dir, f"summary_batch_{i}.pkl")
                with open(temp_file_path, "wb") as f:
                    pickle.dump(batch_results, f)
                del batch_df
                del batch_texts
                del batch_subjects
                del batch_results

            final_summaries = []
            temp_files = [
                f
                for f in os.listdir(all_summaries_temp_dir)
                if f.startswith("summary_batch_") and f.endswith(".pkl")
            ]
            for temp_file in tqdm(temp_files, desc="Aggregating summary results"):
                file_path = os.path.join(all_summaries_temp_dir, temp_file)
                with open(file_path, "rb") as f:
                    batch_data = pickle.load(f)
                    final_summaries.extend(batch_data)

            return final_summaries

        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return None
        finally:
            if os.path.exists(all_summaries_temp_dir):
                shutil.rmtree(all_summaries_temp_dir)
                logger.info(f"Cleaned up temporary directory: {all_summaries_temp_dir}")

    def analyze_emails(self, df=None, num_threads=None, batch_size=1000):
        """
        Perform comprehensive analysis of email data.

        Args:
            df (pandas.DataFrame, optional): DataFrame containing email data.
                If not provided, data will be loaded from the most recent file.
            num_threads (int, optional): Number of threads to use for parallel execution
                                         of independent analysis tasks (e.g., NER and Summarization).
                                         Defaults to os.cpu_count().
            batch_size (int): Number of rows to process in each batch for memory-intensive operations.

        Returns:
            dict: Dictionary containing all analysis results
        """
        # Load data if not provided
        if df is None or df.empty:
            df = self.load_data()
            if df.empty:
                logger.error("No data available for analysis")
                return None

        # Determine the number of threads to use for parallel tasks
        if num_threads is None:
            num_threads = os.cpu_count()
            if num_threads is None:
                num_threads = 1
            logger.info(
                f"Automatically determined {num_threads} CPU cores available for parallel tasks. Transformers pipelines manage their own internal parallelism."
            )
        else:
            logger.info(
                f"Using {num_threads} (user-specified) threads for parallel tasks. Transformers pipelines manage their own internal parallelism."
            )

        logger.info(
            f"Starting comprehensive analysis of {len(df)} emails with batch size {batch_size}"
        )

        # Create timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Perform various analyses
        results = {}

        # 1. Topic modeling (not parallelized at this level)
        results["topics"] = self.extract_topics(df, n_topics=100, method="nmf")

        # 2. Clustering (not parallelized at this level)
        results["clusters"] = self.cluster_emails(df)

        # 3. Parallel execution of Named Entity Extraction and Summarization
        # Using ThreadPoolExecutor to run these two methods concurrently
        # max_workers is set to min(num_threads, 2) because we are running 2 specific tasks in parallel.
        # The internal parallelism within each task (e.g., within transformers pipelines) is handled by the pipelines themselves.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_threads, 2)) as executor:
            logger.info(
                "Submitting Named Entity Extraction and Summarization tasks for parallel execution."
            )
            ner_future = executor.submit(
                self.extract_named_entities, df, use_sample=False, batch_size=batch_size
            )
            summaries_future = executor.submit(
                self.generate_summaries, df, use_sample=False, batch_size=batch_size, max_length=35
            )

            # Wait for results and store them
            results["entities"] = ner_future.result()
            results["summaries"] = summaries_future.result()
            logger.info("Named Entity Extraction and Summarization tasks completed.")

        # 4. Sentiment analysis (can run separately or also be parallelized if needed)
        # For now, keeping it sequential after the above parallel block.
        results["sentiment"] = self.analyze_sentiment(df, use_sample=False, batch_size=batch_size)

        # Save results
        results_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}.pkl")
        try:
            with open(results_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved analysis results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")

        return results

    def save_to_json(self, df: Optional[pd.DataFrame], file_path=None):
        """
        Save analysis results to JSON format.

        Args:
            df (pandas.DataFrame, optional): DataFrame containing email data.
            file_path (str, optional): Path to save the JSON file. If not provided, the file will be saved to the output directory.
        """

        if df is None:
            df = self.load_data(file_path)
            if df.empty:
                logger.error("No data available for analysis")
                return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}.json")
        df.to_json(results_path)
        logger.info(f"Saved analysis results to {results_path}")


if __name__ == "__main__":
    try:
        logger.info("Downloading NLTK resources")
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")

    analyzer = SummarizationClassification()

    # Analyze emails with batching and parallel execution of NER and Summarization
    # num_threads for the outer ThreadPoolExecutor will be min(os.cpu_count(), 2)
    results = analyzer.analyze_emails(num_threads=os.cpu_count(), batch_size=1000)

    print("Done!")
