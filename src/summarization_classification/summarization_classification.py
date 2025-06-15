import concurrent.futures  # Import for threading
import json
import os
import pickle
import re
import shutil  # For deleting temporary directories
import tempfile  # For temporary files
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from transformers import logging as transformers_logging
from transformers import pipeline

from utils import load_processed_df

transformers_logging.set_verbosity_error()


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

    def __init__(
        self,
        input_dir="./processed_data/",
        output_dir="./analysis_results/",
        use_checkpoint=False,
        skip=False,
    ):
        """
        Initialize the SummarizationClassification class.

        Args:
            input_dir (str): Directory containing processed email data
            output_dir (str): Directory to store analysis results
            use_checkpoint (bool): Whether to use checkpoint functionality for resuming interrupted analysis
            skip (bool): Whether to skip initialization and skip all other functions
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_checkpoint = use_checkpoint
        self.skip = skip

        if skip:
            return

        # Create an output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize NLTK components
        self._initialize_nltk()

        # Initialize transformer pipelines with optimized settings
        self.summarizer_factual = None
        self.summarizer_creative = None
        self.summarizer_narrative = None
        self.summarizer_key_points = None
        self.sentiment_analyzer = None
        self.ner = None

        try:
            logger.info("Loading summarization pipelines...")
            # Use smaller, more focused models with optimized settings
            try:
                self.summarizer_factual = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device="mps",
                    max_length=130,
                    min_length=30,
                    truncation=True,
                    num_beams=2,  # Reduced from 4
                    do_sample=False,  # Disable sampling for faster generation
                )
                logger.info("Loaded factual summarization model (BART-CNN)")
            except Exception as e:
                logger.error(f"Error loading factual summarization model: {e}")

            try:
                self.summarizer_creative = pipeline(
                    "summarization",
                    model="google/pegasus-xsum",
                    device="mps",
                    max_length=130,
                    min_length=30,
                    truncation=True,
                    num_beams=2,
                    do_sample=False,
                )
                logger.info("Loaded creative summarization model (PEGASUS-XSUM)")
            except Exception as e:
                logger.error(f"Error loading creative summarization model: {e}")

            try:
                self.summarizer_narrative = pipeline(
                    "summarization",
                    model="facebook/bart-base",
                    device="mps",
                    max_length=130,
                    min_length=30,
                    truncation=True,
                    num_beams=2,
                    do_sample=False,
                )
                logger.info("Loaded narrative summarization model (BART-base)")
            except Exception as e:
                logger.error(f"Error loading narrative summarization model: {e}")

            # Add key_points summarizer
            try:
                self.summarizer_key_points = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",  # Using BART-CNN for key points as well
                    device="mps",
                    max_length=130,
                    min_length=30,
                    truncation=True,
                    num_beams=2,
                    do_sample=False,
                )
                logger.info("Loaded key points summarization model (BART-CNN)")
            except Exception as e:
                logger.error(f"Error loading key points summarization model: {e}")

            logger.info("Loading sentiment analysis pipeline...")
            try:
                # Use a smaller, more focused sentiment model with optimized settings
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device="mps",
                    truncation=True,
                    max_length=512,  # Limit sequence length
                    batch_size=32,  # Optimize batch size
                )
                logger.info("Loaded sentiment analysis model")
            except Exception as e:
                logger.error(f"Error loading sentiment analysis model: {e}")

            logger.info("Loading named entity recognition pipeline...")
            try:
                # Remove truncation parameter for NER
                self.ner = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device="mps",
                )
                logger.info("Loaded NER model")
            except Exception as e:
                logger.error(f"Error loading NER model: {e}")

        except Exception as e:
            logger.error(f"Error initializing transformers pipelines: {e}")

        # Initialize checkpoint file path
        self.checkpoint_file = os.path.join(self.output_dir, "processing_checkpoint.json")

    def _initialize_nltk(self):
        """
        Initialize NLTK components with proper error handling and verification.
        """
        try:
            # First, ensure NLTK data directory exists
            nltk_data_dir = os.path.expanduser("~/nltk_data")

            logger.info(f"NLTK data directory: {nltk_data_dir}")

            if not os.path.exists(nltk_data_dir):
                os.makedirs(nltk_data_dir)

            # Download required NLTK resources
            required_packages = [
                "punkt",
                "stopwords",
                "wordnet",
                "omw-1.4",
                "punkt_tab",
                "averaged_perceptron_tagger_eng",
            ]

            for package in required_packages:
                try:
                    nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                except Exception as e:
                    logger.error(f"Error downloading NLTK package {package}: {e}")

            # Verify downloads and initialize components
            try:
                # Initialize components first
                self.stop_words = set(stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()

                # Simple verification of components
                if not self.stop_words:
                    raise Exception("Stopwords not loaded properly")

                # Test lemmatizer with a simple word
                test_word = self.lemmatizer.lemmatize("running")
                if not test_word:
                    raise Exception("WordNet lemmatizer not working correctly")

                logger.info("NLTK components initialized successfully")

            except Exception as e:
                logger.error(f"Error verifying NLTK components: {e}")
                raise

        except Exception as e:
            logger.error(f"Failed to initialize NLTK: {e}")
            raise

    def load_data(self):
        """
        Load processed email data from a file.

        Args:
            df (pd.DataFrame): Dataframe containing processed email data.

        Returns:
            pandas.DataFrame: DataFrame containing the processed email data
        """

        search_dir = self.output_dir if self.skip else self.input_dir
        search_file_pattern = "analysis_results" if self.skip else "processed_data_"

        df = load_processed_df(search_dir, search_file_pattern)
        return df

    def preprocess_text(self, text):
        """
        Preprocess text for NLP tasks with improved handling of empty content.

        Args:
            text (str): Text to preprocess

        Returns:
            list: List of preprocessed tokens
        """
        if not isinstance(text, str) or not text:
            return []

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Keep more meaningful terms and reduce stopword filtering
        tokens = [
            token
            for token in tokens
            if (token.isalpha() or token.isdigit())  # Keep both words and numbers
            and len(token) > 1  # Keep words longer than 1 character
            and token not in self.stop_words  # Remove stopwords
            # and
            # not token.startswith('http') and  # Remove URLs
            # not token.startswith('www') and
            # not token.startswith('@')  # Remove mentions
        ]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def extract_topics(self, df: pd.DataFrame, n_topics=10, method="lda"):
        """
        Extract topics from email content using topic modeling with improved error handling.
        """
        logger.info(f"Extracting {n_topics} topics using {method}")

        # Use clean_body column if available, otherwise use body
        text_column = "clean_body" if "clean_body" in df.columns else "body"

        # Preprocess and combine all texts
        all_texts = []
        valid_indices = []
        for idx, text in enumerate(df[text_column].fillna("")):
            tokens = self.preprocess_text(text)
            if tokens:  # Only add non-empty token lists
                all_texts.append(" ".join(tokens))
                valid_indices.append(idx)

        if not all_texts:
            logger.error("No valid texts found after preprocessing")
            return None

        # Create document-term matrix with more lenient parameters
        if method == "nmf":
            vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=1,  # Allow single occurrences
                max_df=1.0,  # Allow all terms
                stop_words=None,  # Don't remove stopwords again
                ngram_range=(1, 2),  # Include bigrams
            )
        else:
            vectorizer = CountVectorizer(
                max_features=5000,
                min_df=1,  # Allow single occurrences
                max_df=1.0,  # Allow all terms
                stop_words=None,  # Don't remove stopwords again
                ngram_range=(1, 2),  # Include bigrams
            )

        try:
            # Fit vectorizer and transform documents
            logger.info("Vectorizing text for topic modeling...")
            dtm = vectorizer.fit_transform(all_texts)

            if dtm.shape[1] == 0:
                logger.error("No features found after vectorization")
                return None

            feature_names = vectorizer.get_feature_names_out()
            logger.info(f"Text vectorization complete. Found {len(feature_names)} features.")

            # Create and fit a topic model
            if method == "nmf":
                model = NMF(n_components=n_topics, random_state=42)
            else:
                model = LatentDirichletAllocation(
                    n_components=n_topics, random_state=42, max_iter=10
                )

            # Fit the model
            logger.info(f"Fitting {method.upper()} model...")
            model.fit(dtm)
            logger.info(f"{method.upper()} model fitting complete.")

            # Extract the top keywords for each topic
            topics = {}
            for topic_idx, topic in enumerate(model.components_):
                top_keywords_idx = topic.argsort()[:-11:-1]
                top_keywords = [feature_names[i] for i in top_keywords_idx]
                topics[f"Topic {topic_idx + 1}"] = top_keywords

            # Assign topics to documents
            doc_topic_matrix = model.transform(dtm)
            # Reset index to ensure we have a clean numeric index
            df = df.reset_index(drop=True)
            # Create a new column for topics, initialized with -1 (unassigned)
            df["dominant_topic"] = -1
            # Assign topics only to valid indices
            df.loc[valid_indices, "dominant_topic"] = np.argmax(doc_topic_matrix, axis=1) + 1

            return {
                "vectorizer": vectorizer,
                "topics": topics,
            }

        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return None

    def cluster_emails(self, df: pd.DataFrame, n_clusters=None, method="kmeans"):
        """
        Cluster emails based on their content with improved error handling.
        """
        logger.info(f"Clustering emails using {method}")

        # Use clean_body column if available, otherwise use body
        text_column = "clean_body" if "clean_body" in df.columns else "body"

        # Preprocess and combine all texts
        all_texts = []
        valid_indices = []
        for idx, text in enumerate(df[text_column].fillna("")):
            tokens = self.preprocess_text(text)
            if tokens:  # Only add non-empty token lists
                all_texts.append(" ".join(tokens))
                valid_indices.append(idx)

        if not all_texts:
            logger.error("No valid texts found after preprocessing")
            return None

        # Create a TF-IDF matrix with more lenient parameters
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,  # Allow single occurrences
            max_df=1.0,  # Allow all terms
            stop_words=None,  # Don't remove stopwords again
            ngram_range=(1, 2),  # Include bigrams
        )

        try:
            # Fit vectorizer and transform documents
            logger.info("Vectorizing text for clustering...")
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            if tfidf_matrix.shape[1] == 0:
                logger.error("No features found after vectorization")
                return None

            logger.info(f"Text vectorization complete. Found {tfidf_matrix.shape[1]} features.")
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            logger.info(f"Number of valid texts: {len(all_texts)}")
            logger.info(f"Number of valid indices: {len(valid_indices)}")
            logger.info(f"Original DataFrame shape: {df.shape}")

            # Determine the optimal number of clusters if not provided
            if method == "kmeans" and n_clusters is None:
                # Try different numbers of clusters and evaluate using silhouette score
                silhouette_scores = []
                # Adjust range calculation to ensure we have at least 2 clusters
                max_clusters = min(
                    10, max(2, len(df) // 5)
                )  # Changed from 20 to 5 to get more clusters
                range_n_clusters = range(2, max_clusters + 1)
                logger.info(
                    f"Attempting to find optimal clusters in range: {list(range_n_clusters)}"
                )

                for n in tqdm(range_n_clusters, desc="Finding optimal KMeans clusters"):
                    kmeans = KMeans(n_clusters=n, random_state=42, n_init=5)  # Reduced n_init
                    cluster_labels = kmeans.fit_predict(tfidf_matrix)

                    # Log cluster distribution
                    unique_labels = np.unique(cluster_labels)
                    # label_counts = np.bincount(cluster_labels)
                    # logger.info(f"For n_clusters={n}:")
                    # logger.info(f"  - Found {len(unique_labels)} unique clusters")
                    # logger.info(f"  - Cluster sizes: {label_counts}")

                    if len(unique_labels) > 1 and tfidf_matrix.shape[0] > 1:
                        try:
                            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
                            silhouette_scores.append(silhouette_avg)
                            # logger.info(f"  - Silhouette score: {silhouette_avg:.3f}")
                        except Exception as e:
                            logger.warning(f"  - Could not calculate silhouette score: {str(e)}")
                            # silhouette_scores.append(-1) # DONT USE THIS
                    else:
                        # logger.warning(f"  - Skipping silhouette score: insufficient unique clusters or data points") # DONT USE THIS
                        silhouette_scores.append(-1)

                if silhouette_scores and max(silhouette_scores) > 0:
                    n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
                    logger.info(
                        f"Optimal number of clusters determined: {n_clusters} (score: {max(silhouette_scores):.3f})"
                    )
                else:
                    n_clusters = 5
                    logger.warning(
                        "Could not determine optimal clusters (all silhouette scores were invalid), defaulting to 5."
                    )

            # Perform clustering
            if method == "kmeans":
                n_clusters = n_clusters or 5
                logger.info(f"Performing KMeans clustering with {n_clusters} clusters...")
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)  # Reduced n_init
                cluster_labels = model.fit_predict(tfidf_matrix)

                # Create a new column for clusters, initialized with -1 (unclustered)
                df["cluster"] = -1
                # Reset index to ensure we have a clean numeric index
                df = df.reset_index(drop=True)
                # Assign cluster labels only to valid indices
                df.loc[valid_indices, "cluster"] = cluster_labels

                logger.info("KMeans clustering complete.")
            else:  # DBSCAN
                logger.info("Performing DBSCAN clustering...")
                scaler = StandardScaler(with_mean=False)
                scaled_tfidf = scaler.fit_transform(tfidf_matrix)

                model = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = model.fit_predict(scaled_tfidf)

                # Create a new column for clusters, initialized with -1 (unclustered)
                df["cluster"] = -1
                # Assign cluster labels only to valid indices
                df.loc[valid_indices, "cluster"] = cluster_labels

                logger.info("DBSCAN clustering complete.")

            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in tqdm(sorted(df["cluster"].unique()), desc="Analyzing clusters"):
                cluster_emails = df[df["cluster"] == cluster_id]

                if len(cluster_emails) > 0:
                    all_text = " ".join(cluster_emails[text_column].fillna(""))
                    tokens = self.preprocess_text(all_text)
                    most_common = Counter(tokens).most_common(10)

                    # Filter out empty subjects and ensure we have valid subjects
                    sample_subjects = cluster_emails["subject"].fillna("").astype(str)
                    sample_subjects = (
                        sample_subjects[sample_subjects.str.strip() != ""].head(5).tolist()
                    )

                    # If we still don't have enough subjects, try to get more
                    if len(sample_subjects) < 5:
                        additional_subjects = cluster_emails["subject"].fillna("").astype(str)
                        additional_subjects = (
                            additional_subjects[additional_subjects.str.strip() != ""]
                            .head(10)
                            .tolist()
                        )
                        sample_subjects.extend(additional_subjects)
                        sample_subjects = list(dict.fromkeys(sample_subjects))[
                            :5
                        ]  # Remove duplicates and limit to 5

                    cluster_analysis[f"Cluster {cluster_id}"] = {
                        "size": len(cluster_emails),
                        "common_words": [word for word, count in most_common],
                        "sample_subjects": sample_subjects,
                    }

            # Save clustering results
            result = {
                "vectorizer": vectorizer,
                "cluster_analysis": cluster_analysis,
            }

            return result

        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return None

    def _process_entity_batch(self, batch_texts: list[str]):
        """
        Processes a batch of texts for entity extraction using the NER pipeline.
        The transformer pipeline itself handles internal parallelism for the batch.
        """
        if self.ner is None:
            logger.warning("NER pipeline not initialized.")
            return [[] for _ in batch_texts]  # Return empty lists for each text

        # The pipeline handles batching internally when given a list of texts
        ner_results_batch = self.ner(batch_texts)

        processed_entities = []
        for ner_results_for_text in ner_results_batch:
            row_entities = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}

            # Track the last entity type and word for proper merging
            last_entity_type = None
            last_word = None

            for entity in ner_results_for_text:
                # Get the entity type, handling both formats (with/without B-/I- prefix)
                entity_type = (
                    entity["entity_group"] if "entity_group" in entity else entity["entity"]
                )

                # Remove B-/I- prefix if present
                if entity_type.startswith("B-") or entity_type.startswith("I-"):
                    entity_type = entity_type[2:]

                # Special handling for person names
                if entity_type in ["PER", "PERSON"]:
                    entity_type = "PERSON"
                elif entity_type in ["ORG", "ORGANIZATION"]:
                    entity_type = "ORG"
                elif entity_type in ["LOC", "GPE", "LOCATION"]:
                    entity_type = "LOC"

                # Handle multi-word entities
                if last_entity_type == entity_type and last_word:
                    # Merge with previous word if it's the same entity type
                    last_word = f"{last_word} {entity['word']}"
                else:
                    # If we had a previous word, add it to the appropriate category
                    if last_word:
                        if last_entity_type in row_entities:
                            row_entities[last_entity_type].append(last_word)
                        else:
                            row_entities["MISC"].append(last_word)
                    # Start new word
                    last_word = entity["word"]
                    last_entity_type = entity_type

            # Don't forget to add the last entity
            if last_word:
                if last_entity_type in row_entities:
                    row_entities[last_entity_type].append(last_word)
                else:
                    row_entities["MISC"].append(last_word)

            # Remove duplicates while preserving order
            for entity_type in row_entities:
                row_entities[entity_type] = list(dict.fromkeys(row_entities[entity_type]))

            processed_entities.append(row_entities)

        return processed_entities

    def extract_named_entities(self, df, use_sample=True, sample_size=100, batch_size=500):
        """
        Extract named entities from email content, with batch processing for memory efficiency.
        The transformer pipeline handles internal parallelism within each batch.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            use_sample (bool): Whether to use sample for batch processing
            sample_size (int): Number of emails to sample for entity extraction (if use_sample is True)
            batch_size (int): Number of rows to process in each batch.
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
                batch_texts = [str(text) for text in batch_df[text_column].fillna("")]
                # The _process_entity_batch function now directly calls the pipeline with the batch_texts
                batch_results = self._process_entity_batch(batch_texts)

                # Save batch result to a temporary file
                temp_file_path = os.path.join(all_entities_temp_dir, f"ner_batch_{i}.pkl")
                with open(temp_file_path, "wb") as f:
                    pickle.dump(batch_results, f)  # Save the list of dicts
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

    def _process_sentiment_batch(self, batch_texts: list[str], batch_subjects: list[str]):
        """
        Optimized batch processing for sentiment analysis.
        """
        if self.sentiment_analyzer is None:
            logger.warning("Sentiment analysis pipeline not initialized.")
            return []

        try:
            # Process texts in smaller chunks to avoid memory issues
            chunk_size = 32
            batch_sentiments = []

            for i in range(0, len(batch_texts), chunk_size):
                chunk_texts = batch_texts[i : i + chunk_size]
                chunk_subjects = batch_subjects[i : i + chunk_size]

                # Process chunk
                sentiment_results = self.sentiment_analyzer(chunk_texts)

                # Combine results
                for idx, sentiment_result in enumerate(sentiment_results):
                    batch_sentiments.append(
                        {
                            "subject": chunk_subjects[idx],
                            "label": sentiment_result["label"],
                            "score": sentiment_result["score"],
                        }
                    )

                # Free up memory
                del sentiment_results

            return batch_sentiments

        except Exception as e:
            logger.error(f"Error in sentiment batch processing: {e}")
            return []

    def analyze_sentiment(
        self, df: pd.DataFrame, use_sample=True, sample_size=100, batch_size=1000
    ):
        """
        Optimized sentiment analysis with improved batch processing.
        """
        if self.sentiment_analyzer is None:
            logger.warning("Sentiment analysis not available")
            return None

        logger.info(
            f"Analyzing sentiment with optimized batch processing (batch size: {batch_size})"
        )

        text_column = "clean_body" if "clean_body" in df.columns else "body"
        subject_column = "subject"

        # Filter out empty texts and limit text length
        df_for_sentiment = df[df[text_column].str.len() > 0].copy()
        df_for_sentiment[text_column] = df_for_sentiment[text_column].apply(
            lambda x: str(x)[:512] if pd.notnull(x) else ""  # Limit text length
        )

        if use_sample:
            if len(df_for_sentiment) > sample_size:
                df_to_process = df_for_sentiment.sample(sample_size, random_state=42)
            else:
                df_to_process = df_for_sentiment
        else:
            df_to_process = df_for_sentiment

        total_rows = len(df_to_process)
        all_sentiments = []

        try:
            # Process in optimized batches
            for i in tqdm(range(0, total_rows, batch_size), desc="Processing sentiment batches"):
                batch_df = df_to_process.iloc[i : i + batch_size]
                batch_texts = batch_df[text_column].tolist()
                batch_subjects = batch_df[subject_column].tolist()

                # Process batch with optimized settings
                batch_results = self._process_sentiment_batch(batch_texts, batch_subjects)
                all_sentiments.extend(batch_results)

                # Free up memory
                del batch_df
                del batch_texts
                del batch_subjects
                del batch_results

            # Calculate sentiment distribution
            sentiment_distribution = Counter([s["label"] for s in all_sentiments])

            return {
                "sentiments": all_sentiments,
                "distribution": sentiment_distribution,
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None

    def _process_summary_batch(
        self,
        batch_texts: list[str],
        batch_subjects,
        max_length,
        min_length=30,
        style="factual",
        creativity_level=0.0,
        narrative_focus="general",
    ):
        """
        Enhanced batch processing for summarization with multiple styles and options.
        """
        batch_summaries = []

        # Select the appropriate summarizer based on style
        summarizer = {
            "factual": self.summarizer_factual,
            "creative": self.summarizer_creative,
            "narrative": self.summarizer_narrative,
            "key_points": self.summarizer_key_points,
        }.get(style)

        if summarizer is None:
            logger.warning(f"Summarization pipeline for style {style} not available.")
            return batch_summaries

        # Enhanced generation parameters based on style
        generation_params = {
            "min_length": min_length,
            "do_sample": style not in ["factual", "key_points", "chronological"],
            "num_beams": 2,  # Reduced from 4
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "early_stopping": False,  # True if explain early_stopping ->
        }

        try:
            # Process each text individually to handle dynamic max_length
            for idx, text in enumerate(batch_texts):
                # # Limit text length to improve performance
                # text = text[:500]  # Reduced from 1000 to 500

                # Calculate appropriate max_length based on input length
                input_length = len(text.split())
                # Ensure max_length is significantly shorter than input_length for summarization
                current_max_length = min(max_length, max(min_length, int(input_length * 0.3)))
                if (
                    current_max_length >= input_length * 0.5
                ):  # If summary would be more than 50% of input
                    current_max_length = max(
                        min_length, int(input_length * 0.4)
                    )  # Force it to be at most 40% of input

                # Update generation parameters with the current max_length
                current_params = generation_params.copy()
                current_params["max_length"] = current_max_length

                # Generate summary for this text
                summary_result = summarizer(text, **current_params)[0]
                summary_text = summary_result["summary_text"]

                # Extract key information using NER if available
                key_info = {}
                if self.ner:
                    ner_results = self.ner(summary_text)
                    for entity in ner_results:
                        entity_type = entity["entity_group"]
                        if entity_type not in key_info:
                            key_info[entity_type] = []
                        key_info[entity_type].append(entity["word"])

                summary_info = {
                    "subject": batch_subjects[idx],
                    "original_length": len(text),
                    "summary": summary_text,
                    "style": style,
                    "creativity_level": creativity_level,
                    "narrative_focus": (narrative_focus if style == "narrative" else None),
                    "generation_params": current_params,
                    "key_information": {"entities": key_info},
                }

                batch_summaries.append(summary_info)

        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")

        return batch_summaries

    def _process_batch(
        self,
        batch_df,
        text_column: str,
        subject_column: str,
        max_length: int,
        min_length: int,
        style_name: str,
        creativity_level: float,
        narrative_focus: str,
    ) -> List[Dict[str, Any]]:
        """Helper function to process a single batch of data."""
        try:
            batch_texts = [str(text)[:500] for text in batch_df[text_column].fillna("")]
            batch_subjects = batch_df[subject_column].tolist()

            batch_results = self._process_summary_batch(
                batch_texts,
                batch_subjects,
                max_length,
                min_length,
                style_name,
                creativity_level,
                narrative_focus,
            )

            return batch_results if batch_results else []
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []

    def generate_summaries(
        self,
        df,
        use_sample=True,
        sample_size=20,
        batch_size=250,
        style_config=None,
        max_length=150,
        min_length=50,
    ):
        """
        Enhanced summary generation with multiple styles and comprehensive output.
        Now includes caching, performance optimizations, and parallel batch processing.
        """
        try:
            if not any(
                [
                    self.summarizer_factual,
                    self.summarizer_creative,
                    self.summarizer_narrative,
                    self.summarizer_key_points,
                ]
            ):
                logger.warning("No summarization pipelines available")
                return None

            # Default styles if none provided - reduced to the most important ones
            if style_config is None:
                style_config = {
                    "style": "key_points",
                }

            logger.info(f"Generating summaries with {style_config.get('style')} styles")

            text_column = "body"
            subject_column = "subject"
            # Filter for substantial content and clean text

            df_for_summary = df[df[text_column].str.len() > 200].copy()
            df_for_summary[text_column] = df_for_summary[text_column].apply(
                lambda x: self.clean_body(str(x)) if pd.notnull(x) else ""
            )

            if use_sample:
                if len(df_for_summary) > sample_size:
                    df_to_process = df_for_summary.sample(sample_size, random_state=42)
                else:
                    df_to_process = df_for_summary
            else:
                df_to_process = df_for_summary

            all_summaries = []

            creativity_level = style_config.get("creativity_level", 0.0)
            narrative_focus = style_config.get("narrative_focus", "general")

            # Create batches
            batches = [
                df_to_process.iloc[i : i + batch_size]
                for i in range(0, len(df_to_process), batch_size)
            ]

            # Process batches in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(batches), 4)
            ) as executor:
                # Submit all batch processing tasks
                future_to_batch = {
                    executor.submit(
                        self._process_batch,
                        batch,
                        text_column,
                        subject_column,
                        max_length,
                        min_length,
                        style_config.get("style"),
                        creativity_level,
                        narrative_focus,
                    ): i
                    for i, batch in enumerate(batches)
                }

                # Process results as they complete
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_batch),
                    total=len(batches),
                    desc=f"Processing {style_config.get('style')} batches",
                ):
                    try:
                        batch_results = future.result()
                        if batch_results:
                            all_summaries.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")

            # Organize results by document and style
            organized_results = {}
            for summary in all_summaries:
                subject = summary["subject"]
                style = summary["style"]

                if subject not in organized_results:
                    organized_results[subject] = {}

                organized_results[subject][style] = summary

            return {
                "summaries": organized_results,
                "metadata": {
                    "total_documents": len(df_to_process),
                    "styles_generated": style_config.get("style"),
                    "generation_timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Error in summarization generation: {e}")
            return None

    def analyze_narrative_structure(self, text):
        """
        Analyze the narrative structure of a text.

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Narrative structure analysis
        """
        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(text)

            # Basic story element detection
            structure = {
                "beginning": sentences[: len(sentences) // 4],
                "middle": sentences[len(sentences) // 4 : 3 * len(sentences) // 4],
                "end": sentences[3 * len(sentences) // 4 :],
                "length": len(sentences),
            }

            # Detect potential story elements
            story_elements = {
                "settings": [],
                "characters": [],
                "actions": [],
                "dialogue": [],
            }

            for sentence in sentences:
                # Detect dialogue (sentences with quotation marks)
                if '"' in sentence or "'" in sentence:
                    story_elements["dialogue"].append(sentence)

                # Use NER to detect characters and locations
                if self.ner:
                    ner_results = self.ner(sentence)
                    for entity in ner_results:
                        if entity["entity_group"] == "PERSON":
                            story_elements["characters"].append(entity["word"])
                        elif entity["entity_group"] in ["LOC", "GPE"]:
                            story_elements["settings"].append(entity["word"])

                # Detect action sentences (those with verbs)
                tokens = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(tokens)
                if any(tag.startswith("VB") for _, tag in pos_tags):
                    story_elements["actions"].append(sentence)

            # Remove duplicates while preserving order
            for key in story_elements:
                story_elements[key] = list(dict.fromkeys(story_elements[key]))

            # Add emotional analysis if available
            if self.sentiment_analyzer:
                emotions = self.sentiment_analyzer(text)
                structure["emotional_arc"] = emotions

            return {"structure": structure, "elements": story_elements}
        except Exception as e:
            logger.error(f"Error in narrative structure analysis: {e}")
            return None

    def generate_story_elements(self, summary_results):
        """
        Generate additional story elements from summarization results.

        Args:
            summary_results (dict): Results from generate_summaries()

        Returns:
            dict: Enhanced story elements
        """
        if summary_results is None:
            logger.warning("No summary results available to generate story elements")
            return {}

        if "summaries" not in summary_results:
            logger.warning("No 'summaries' key found in summary results")
            return {}

        enhanced_results = {}

        for subject, style_data in summary_results["summaries"].items():
            enhanced_results[subject] = {
                "summary": style_data,
                "narrative_analysis": {},
            }

            # Get the summary text
            summary_text = style_data["summary"]

            # Analyze narrative structure
            narrative_analysis = self.analyze_narrative_structure(summary_text)
            if narrative_analysis:
                enhanced_results[subject]["narrative_analysis"] = narrative_analysis

                # Generate potential story suggestions
                suggestions = {
                    "plot_points": [],
                    "character_arcs": [],
                    "setting_details": [],
                    "themes": [],
                }

                # Extract potential plot points from actions
                if narrative_analysis["elements"]["actions"]:
                    suggestions["plot_points"] = narrative_analysis["elements"]["actions"][:5]

                # Generate character arcs for main characters
                for character in narrative_analysis["elements"]["characters"][:3]:
                    if self.sentiment_analyzer:
                        character_mentions = [
                            s for s in nltk.sent_tokenize(summary_text) if character in s
                        ]
                        if character_mentions:
                            arc = {
                                "character": character,
                                "arc_points": character_mentions[:3],
                                "emotional_journey": self.sentiment_analyzer(
                                    " ".join(character_mentions)
                                ),
                            }
                            suggestions["character_arcs"].append(arc)

                # Extract setting details
                if narrative_analysis["elements"]["settings"]:
                    suggestions["setting_details"] = narrative_analysis["elements"]["settings"]

                # Add suggestions to results
                enhanced_results[subject]["story_suggestions"] = suggestions

        return enhanced_results

    @staticmethod
    def clean_body(text):
        """
        Clean and normalize text content.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""

        # Remove common email markers and quoted text
        text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
        text = re.sub(
            r"^(to:|cc:|subject:|from:|copy:|re:).*\n?",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Remove separator lines
        text = re.sub(r"={3,}|-{3,}|\*{3,}", "", text)

        # Remove escape characters and their sequences
        text = re.sub(r"\\[a-zA-Z]", "", text)
        text = re.sub(r"\\/", "", text)

        # Remove repeating symbols
        text = re.sub(r"(\s*[?\.!;:-])\1+", " ", text)

        # Remove multiple newlines and normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+", " ", text)

        # Remove any remaining HTML-like tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove any remaining special characters
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        return text.strip()

    def save_checkpoint(self, current_index, total_items, processed_items):
        """
        Save the current processing state to a checkpoint file.

        Args:
            current_index (int): Current index in the processing
            total_items (int): Total number of items to process
            processed_items (list): List of processed items
        """
        if not self.use_checkpoint:
            return

        checkpoint_data = {
            "current_index": current_index,
            "total_items": total_items,
            "processed_items": processed_items,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(self) -> tuple[int, int, list[str]]:
        """
        Load the processing state from the checkpoint file.

        Returns:
            tuple: (current_index, total_items, processed_items) or (0, 0, []) if no checkpoint exists
        """
        res = (0, 0, [])
        if not self.use_checkpoint:
            return res

        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Loaded checkpoint from {checkpoint_data['timestamp']}")
                return (
                    checkpoint_data["current_index"],
                    checkpoint_data["total_items"],
                    checkpoint_data["processed_items"],
                )
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        return res

    def analyze_emails(self, df: Optional[pd.DataFrame], num_threads=None, batch_size=1000):
        """
        Optimized comprehensive analysis with improved parallel processing.
        Returns both a DataFrame with results and the full results dictionary.

        Returns:
            tuple: (DataFrame with results, dict with full analysis results)
        """
        current_index, total_items, processed_items = self.load_checkpoint()

        try:
            # Load data if not provided
            if df is None or df.empty:
                df = self.load_data()
                if df.empty:
                    logger.error("No data available for analysis")
                    return None, None

            logger.info(
                f"Analyzing emails with optimized batch processing (batch size: {batch_size}) on {df.shape} items"
            )

            # Load checkpoint if exists and checkpoint is enabled
            if self.use_checkpoint and current_index > 0:
                logger.info(f"Resuming processing from index {current_index}")
                df = df.iloc[current_index:]

            # Determine an optimal number of threads
            if num_threads is None:
                num_threads = max(os.cpu_count() or 1, 4)  # use max for performance
                logger.info(f"Using {num_threads} threads for analysis")

            logger.info(f"Starting optimized analysis with {num_threads} threads")

            # Create timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {}

            # Process in parallel with optimized batch sizes
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit tasks with optimized batch sizes
                futures = {
                    "topics": executor.submit(self.extract_topics, df, n_topics=50, method="nmf"),
                    "clusters": executor.submit(self.cluster_emails, df),
                    "entities": executor.submit(
                        self.extract_named_entities,
                        df,
                        use_sample=False,
                        batch_size=500,
                    ),
                    "key_point_summaries": executor.submit(
                        self.generate_summaries,
                        df,
                        use_sample=False,
                        batch_size=500,
                        max_length=75,
                        style_config={"style": "key_points"},
                    ),
                    "factual_summaries": executor.submit(
                        self.generate_summaries,
                        df,
                        use_sample=False,
                        batch_size=500,
                        max_length=75,
                        style_config={"style": "factual"},
                    ),
                    "creative_summaries": executor.submit(
                        self.generate_summaries,
                        df,
                        use_sample=False,
                        batch_size=500,
                        max_length=75,
                        style_config={"style": "creative"},
                    ),
                    "narrative_summaries": executor.submit(
                        self.generate_summaries,
                        df,
                        use_sample=False,
                        batch_size=500,
                        max_length=75,
                        style_config={"style": "narrative"},
                    ),
                    "sentiment": executor.submit(
                        self.analyze_sentiment, df, use_sample=False, batch_size=500
                    ),
                }

            # Collect results as they complete
            for task_name, future in futures.items():
                try:
                    results[task_name] = future.result(timeout=1800)  # 30-minute timeout per task
                    self.save_checkpoint(
                        current_index + len(df),
                        total_items,
                        processed_items + [task_name],
                    )
                except Exception as e:
                    logger.error(e)
                    logger.error(f"Error in {task_name} analysis: {e}")
                    results[task_name] = None

            # Enhance summaries if available
            if results.get("summaries"):
                results["enhanced_summaries"] = self.generate_story_elements(results["summaries"])
                self.save_checkpoint(
                    current_index + len(df),
                    total_items,
                    processed_items + ["enhanced_summaries"],
                )

            # Create a results DataFrame
            results_df = df.copy()

            # Add topic and cluster information if available
            if (
                results.get("topics")
                and "topics" in results["topics"]
                and "dominant_topic" in df.columns
            ):
                try:
                    results_df["dominant_topic"] = df["dominant_topic"]
                    results_df["topic_keywords"] = results_df["dominant_topic"].apply(
                        lambda x: (
                            results["topics"]["topics"].get(f"Topic {x}", [])
                            if pd.notnull(x)
                            else None
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error adding topic information: {e}")

            if (
                results.get("clusters")
                and "cluster_analysis" in results["clusters"]
                and "cluster" in df.columns
            ):
                try:
                    results_df["cluster"] = df["cluster"]
                    results_df["cluster_size"] = results_df["cluster"].apply(
                        lambda x: (
                            results["clusters"]["cluster_analysis"]
                            .get(f"Cluster {x}", {})
                            .get("size", 0)
                            if pd.notnull(x)
                            else None
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error adding cluster information: {e}")

            # Add sentiment information if available
            if results.get("sentiment"):
                sentiment_dict = {
                    s["subject"]: s["label"] for s in results["sentiment"]["sentiments"]
                }
                results_df["sentiment"] = results_df["subject"].map(sentiment_dict)

            # Add summary information if available
            if results.get("summaries"):
                summary_dict = {
                    subject: {style: info["summary"] for style, info in styles.items()}
                    for subject, styles in results["summaries"]["summaries"].items()
                }
                results_df["summaries"] = results_df["subject"].map(summary_dict)

            # Clear checkpoint after successful completion
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)

            return results_df, results

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            self.save_checkpoint(current_index, total_items, processed_items)
            raise

    def save_to_json(self, data: pd.DataFrame | dict, is_dataframe=True):
        """
        Save analysis results to JSON format with proper structuring.

        Args:
            data: Either a DataFrame or dictionary containing analysis results
            is_dataframe (bool): Whether the input is a DataFrame (True) or dictionary (False)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}_v1.json")

            def convert_to_serializable(obj):
                """Helper function to convert non-serializable objects to serializable format."""
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                elif hasattr(obj, "__dict__"):
                    # Handle objects with __dict__ attribute
                    return {
                        key: convert_to_serializable(value)
                        for key, value in obj.__dict__.items()
                        if not key.startswith("_")
                    }
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)  # Convert other objects to string representation

            if is_dataframe:
                # Structure the results in a more readable format
                structured_results = {
                    "metadata": {
                        "timestamp": timestamp,
                        "num_documents": len(data),
                        "columns": data.columns.tolist(),
                    },
                    "data": {
                        "emails": data.to_dict(orient="records"),
                        "topics": (
                            data["dominant_topic"].value_counts().to_dict()
                            if "dominant_topic" in data.columns
                            else None
                        ),
                        "clusters": (
                            data["cluster"].value_counts().to_dict()
                            if "cluster" in data.columns
                            else None
                        ),
                    },
                }
            else:
                # If input is already a dictionary, convert it to serializable format
                structured_results = {
                    "metadata": {
                        "timestamp": timestamp,
                        "num_documents": (
                            len(data.get("summaries", {}).get("summaries", {}))
                            if data.get("summaries")
                            else 0
                        ),
                    },
                    "data": convert_to_serializable(data),
                }

            # Save as JSON with proper formatting
            with open(results_path, "w") as f:
                json.dump(structured_results, f, indent=2)
            logger.info(f"Saved analysis results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")

    def save_to_pickle(self, data: pd.DataFrame | dict, is_dataframe=True):
        """
        Save analysis results to pickle format.

        Args:
            data: Either a DataFrame or dictionary containing analysis results
            is_dataframe (bool): Whether the input is a DataFrame (True) or dictionary (False)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.output_dir, f"analysis_results_{timestamp}_v1.pkl")

            if is_dataframe:
                # Create a dictionary with both DataFrame and metadata
                structured_results = {
                    "metadata": {
                        "timestamp": timestamp,
                        "num_documents": len(data),
                        "columns": data.columns.tolist(),
                    },
                    "data": data,
                }
            else:
                # If input is already a dictionary, just
                #
                #
                # add metadata
                structured_results = {
                    "metadata": {
                        "timestamp": timestamp,
                        "num_documents": (
                            len(data.get("summaries", {}).get("summaries", {}))
                            if data.get("summaries")
                            else 0
                        ),
                    },
                    "data": data,
                }

            # Save as pickle
            with open(results_path, "wb") as f:
                pickle.dump(structured_results, f)
            logger.info(f"Saved analysis results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")


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
