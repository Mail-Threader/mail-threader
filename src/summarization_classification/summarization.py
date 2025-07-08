import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor as FutureExecutor
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from gliner import GLiNER
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import ndarray
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers.pipelines import pipeline
from utils import custom_stop_words, load_processed_df, save_error_log, sort_emails_by_date
from database import DatabaseManager


class Summarization:
	"""
    Class responsible for summarizing and classifying email data.
    - Text preprocessing (tokenization, lemmatization)
    - Topic modeling
    - Email Clustering
    - Entity Recognition
    - Sentiment Analysis
    - Text Summarization
    """

	def __init__(
		self,
		input_dir="./processed_data/",
		output_dir="./analysis_results/",
		error_dir="./error_logs/",
		skip=False,
	):
		"""
        Initialize the Summarization class.

        Args:
            input_dir (str): Directory containing processed email data
            output_dir (str): Directory to store analysis results
            skip (bool): Flag to skip processing if results already exist
        """
		self.input_dir = input_dir
		self.output_dir = output_dir
		self.error_dir = error_dir
		self.skip = skip

		# Ensure directories exist
		for d in [self.output_dir, self.error_dir]:
			if not os.path.exists(d):
				try:
					os.makedirs(d)
					logger.info(f"Created directory: {d}")
				except Exception as e:
					save_error_log(f"Error creating directory {d}: {e}")
					logger.error(f"Error creating directory {d}: {e}")
					raise RuntimeError(f"Error creating directory {d}: {e}")

		if not os.path.exists(self.input_dir):
			save_error_log(f"Input directory {self.input_dir} does not exist.")
			logger.error(f"Input directory {self.input_dir} does not exist.")
			raise FileNotFoundError(f"Input directory {self.input_dir} does not exist.")

		if self.skip:
			return

		self.stop_words = set(stopwords.words("english"))
		self.stop_words = self.stop_words.union(custom_stop_words)

		self.lemmatizer = WordNetLemmatizer()

		try:
			logger.info("Initializing Named Entity Recognition pipeline...")
			self.ner = GLiNER.from_pretrained("E3-JSI/gliner-multi-pii-domains-v1")
			logger.success("Named Entity Recognition pipeline initialized successfully.")
		except Exception as e:
			save_error_log(f"Error initializing Named Entity Recognition pipeline: {e}")
			logger.error(f"Error initializing Named Entity Recognition pipeline: {e}")
			self.ner = None

		try:
			logger.info("Initializing sentiment analysis pipeline...")
			self.sentiment_analyzer = pipeline(
				"sentiment-analysis",
				model="siebert/sentiment-roberta-large-english",
				device="mps",
			)
			logger.success("Sentiment analysis pipeline initialized successfully.")
		except Exception as e:
			save_error_log(f"Error initializing sentiment analysis pipeline: {e}")
			logger.error(f"Error initializing sentiment analysis pipeline: {e}")
			self.sentiment_analyzer = None

	def get_valid_texts_and_indices(self, df: pd.DataFrame, valid_num_tokens=3):
		"""
        Get valid texts and their indices from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing email data.
            valid_num_tokens (int): Minimum number of tokens required for a valid text.

        Returns:
            tuple: (list of valid texts, list of valid indices)
        """
		logger.info("Filtering valid texts...")

		all_texts: list[str] = []
		valid_indices = []

		for idx, row in tqdm(
			df.iterrows(),
			total=len(df),
			desc="Processing email texts for topic extraction",
		):
			try:

				if pd.isna(row.get("body")) and pd.isna(row.get("subject")):
					continue

				body = row.get("body", "")
				subject = row.get("subject", "")

				all_text = f"{subject} {body}"

				tokens = self.tokenize_text(all_text)

				if not tokens or len(tokens) <= valid_num_tokens:
					continue

				valid_indices.append(idx)
				all_texts.append(" ".join(tokens))

			except Exception as e:
				save_error_log(f"Error processing row {idx}: {e}")
				logger.error(f"Error processing row {idx}: {e}")
				continue

		if not all_texts:
			save_error_log("No valid texts found for topic extraction.")
			logger.error("No valid texts found for topic extraction.")
			return [], []

		logger.info(
			f"Found {len(all_texts)} valid texts for topic extraction with minimum {valid_num_tokens} tokens."
		)

		return all_texts, valid_indices

	# Topic Extraction and Analysis Methods
	def tokenize_text(self, text: str):
		"""
        Tokenize the input text into words.

        Args:
            text (str): Input text to tokenize.

        Returns:
            list: List of tokens.
        """
		tokens = word_tokenize(text.lower())

		tokens = [
			token for token in tokens if (
			token.isalpha() or token.isdigit()) and len(token) > 1 and token not in self.stop_words
		]

		tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

		return tokens

	@staticmethod
	def calculate_topic_similarity(topic_matrix: ndarray):
		"""
        Calculate the similarity between topics based on their word distributions.

        Args:
            topic_matrix (list[str]): List of topic vectors (word distributions).

        Returns:
            pd.DataFrame: DataFrame containing pairwise similarity scores.
        """
		logger.info("Calculating topic similarity...")
		return cosine_similarity(topic_matrix)

	@staticmethod
	def analyze_topic_timeline(df: pd.DataFrame):
		"""
        Analyze the timeline of topics in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing email data with topics.

        Returns:
            dict: Dictionary with topic distribution over time.
        """
		df["time_period"] = df["date"].dt.to_period("M")  # Monthly aggregation

		timeline = {}
		for period in df["time_period"].unique():
			period_df = df[df["time_period"] == period]

			topic_counts: Counter[int] = Counter(period_df["dominant_topic"])
			total_emails = len(period_df)

			topic_distribution = {
				topic: count / total_emails
				for topic, count in topic_counts.items()
			}

			avg_topic_strength = (period_df["topic_strength"].mean()
				if not period_df["topic_strength"].isna().all() else 0.0)

			timeline[str(period)] = {
				"topic_distribution":
				topic_distribution,
				"avg_topic_strength":
				avg_topic_strength,
				"top_topic_label": (period_df["topic_label"].mode()[0]
				if not period_df["topic_label"].isna().all() else None),
				"total_emails":
				total_emails,
			}

		return timeline

	@staticmethod
	def extract_topic_labels(
		model: NMF | LatentDirichletAllocation,
		feature_names: list[str],
		n_top_words=10,
	):
		"""
        Extract topic labels from the fitted model.

        Args:
            model (NMF | LatentDirichletAllocation): Fitted topic model.
            feature_names (list[str]): List of feature names (words).
            n_top_words (int): Number of top words to include in each topic label.

        Returns:
            dict: Dictionary mapping topic indices to their labels and keywords.
        """
		topics = {}

		for topic_idx, topic in enumerate(model.components_):
			top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
			top_features = [feature_names[i] for i in top_features_ind]

			topic_label = "_".join(top_features[:3])  # Use first 3 words as label

			topics[topic_idx] = {
				"label": topic_label,
				"keywords": top_features,
			}

		return topics

	def topic_extraction(
		self,
		df: pd.DataFrame,
		valid_texts_indices_result: tuple[list[str], list[int]],
		method: str = "nmf",
		n_topics=50,
		time_window: str = "M",
	):
		"""
        Extract topics from the email DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing email data.
            valid_texts_indices_result (tuple): Tuple containing valid texts and their indices.
            method (str): Method for topic modeling ('nmf' or 'lda').
            n_topics (int): Number of topics to extract.

        Returns:
            pd.DataFrame: DataFrame with extracted topics.
        """

		logger.info("Extracting topics from emails...")

		vectorizer: Optional[TfidfVectorizer | CountVectorizer] = None

		# Create document-term matrix with optimized parameters for better topic coherence
		if method == "nmf":
			vectorizer = TfidfVectorizer(
				max_features=10000,  # Increased from 5000
				min_df=2,  # Require at least 2 occurrences to reduce noise
				max_df=0.95,  # Ignore terms that appear in >95% of documents (too common)
				ngram_range=(1, 3),  # Include up to trigrams for better phrases
				stop_words="english",  # Remove English stopwords
			)
		else:  # LDA
			vectorizer = CountVectorizer(
				max_features=10000,  # Increased from 5000
				min_df=2,  # Require at least 2 occurrences
				max_df=0.95,  # Ignore terms that appear in >95% of documents
				ngram_range=(1, 3),  # Include up to trigrams
				stop_words="english",  # Remove English stopwords
			)

		model: Optional[NMF | LatentDirichletAllocation] = None

		# make n_topics dynamic based on the len of the df
		log_scaled_topics = int(np.log(len(df)) * 10)

		# Heuristic 2: Simple division
		# This provides a more linear initial estimate.
		linear_scaled_topics = len(df) // 50

		# Combine heuristics (e.g., take the average or a weighted sum)
		# Here, we'll take a simple average for a balanced approach.
		suggested = (log_scaled_topics + linear_scaled_topics) // 2

		# Ensure the suggested number is within the defined bounds
		n_topics = max(5, min(150, suggested))

		logger.info(
			f"Suggested number of topics based on heuristics: {suggested}. Using {n_topics} topics."
		)

		# Create and fit a topic model with optimized parameters
		if method == "nmf":
			# NMF with optimized parameters
			model = NMF(
				n_components=n_topics,
				random_state=42,
				# alpha=0.1,  # Regularization parameter
				l1_ratio=0.5,  # Balance between L1 and L2 regularization
				max_iter=500,  # Increased iterations for better convergence
				tol=1e-4,  # Convergence tolerance
			)
		else:
			# LDA with optimized parameters
			model = LatentDirichletAllocation(
				n_components=n_topics,
				random_state=42,
				max_iter=25,  # Increased from 10
				learning_method="online",  # Faster for large datasets
				learning_offset=50.0,
				doc_topic_prior=
				0.1,  # Dirichlet prior on document-topic distribution (more specific topics)
				topic_word_prior=
				0.01,  # Dirichlet prior on topic-word distribution (more specific words)
			)

		# Get valid texts and their indices
		all_texts, valid_indices = valid_texts_indices_result

		if len(all_texts) == 0:
			save_error_log("No valid texts found for topic extraction.")
			logger.error("No valid texts found for topic extraction.")
			return {
				"topics": {},
				"topic_evolution": {},
				"topic_trends": {},
				"df_with_topics": pd.DataFrame(),
				"topic_similarity": pd.DataFrame(),
				"doc_topic_matrix": None,
				"timeline": {},
			}

		try:

			logger.info("Vectorizing texts for topic modeling...")
			dtm = vectorizer.fit_transform(all_texts)

			logger.info(f"Fitting topic model... with method: {method}")
			model.fit(dtm)
			logger.info("Topic model fitted successfully.")

			feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
			topics = self.extract_topic_labels(model, feature_names)

			doc_topic_matrix = model.transform(dtm)

			result_df = df.iloc[valid_indices].copy()
			result_df["dominant_topic"] = doc_topic_matrix.argmax(axis=1)
			result_df["topic_strength"] = doc_topic_matrix.max(axis=1)

			result_df["topic_label"] = result_df["dominant_topic"].map(lambda x: topics[x]["label"])

			# Analyzing topic timeline
			topic_timeline = self.analyze_topic_timeline(result_df)

			topic_similarity = self.calculate_topic_similarity(model.components_)

			result_df["time_period"] = result_df["date"].dt.to_period(time_window)

			period_stats = {}

			for period in result_df["time_period"].unique():
				period_df = result_df[result_df["time_period"] == period]

				if len(period_df) >= 3:
					topic_counts = Counter(period_df["dominant_topic"])
					total_emails = len(period_df)

					topic_distribution = {
						topics[topic]["label"]: count / total_emails
						for topic, count in topic_counts.items()
					}

					topic_probs = list(topic_distribution.values())
					topic_entropy = -sum(p * np.log(p) if p > 0 else 0 for p in topic_probs)

					period_stats[str(period)] = {
						"topic_distribution":
						topic_distribution,
						"topic_diversity":
						topic_entropy,
						"total_emails":
						total_emails,
						"avg_topic_strength":
						period_df["topic_strength"].mean(),
						"dominant_topic": (period_df["dominant_topic"].mode()[0]
						if not period_df["dominant_topic"].isna().all() else None),
						"top_topic_label": (period_df["topic_label"].mode()[0]
						if not period_df["topic_label"].isna().all() else None),
						"date_range": (
						period_df["date"].min(),
						period_df["date"].max(),
						),
					}

			topic_trends = {}
			periods = sorted(period_stats.keys())

			for topic_id in topics.keys():

				topic_prevalence = []

				for period in periods:
					prevalence = period_stats[period]["topic_distribution"].get(
						topics[topic_id]["label"], 0)
					topic_prevalence.append(prevalence)

				if len(topic_prevalence) > 1:
					x = np.arange(len(topic_prevalence))
					trend = np.polyfit(x, topic_prevalence, 1)

					max_prevalence = max(topic_prevalence)
					peak_period = (periods[topic_prevalence.index(max_prevalence)]
						if max_prevalence > 0 else None)

					topic_trends[topic_id] = {
						"label":
						topics[topic_id]["label"],
						"trend":
						trend,
						"max_prevalence":
						max_prevalence,
						"peak_period":
						peak_period,
						"prevalence_timeline":
						dict(zip(periods, topic_prevalence)),
						"trend_direction": ("increasing"
						if trend[0] > 0 else "decreasing" if trend[0] < 0 else "stable"),
					}

			return {
				"topics":
				topics,
				"topic_evolution":
				period_stats,
				"topic_trends":
				topic_trends,
				"df_with_topics":
				result_df,
				"topic_similarity":
				pd.DataFrame(
				topic_similarity,
				index=[topics[i]["label"] for i in range(len(topics))],
				columns=[topics[i]["label"] for i in range(len(topics))],
				),
				"doc_topic_matrix":
				doc_topic_matrix,
				"timeline":
				topic_timeline,
			}

		except Exception as e:
			return {
				"topics": {},
				"topic_evolution": {},
				"topic_trends": {},
				"df_with_topics": pd.DataFrame(),
				"topic_similarity": pd.DataFrame(),
				"doc_topic_matrix": None,
				"timeline": {},
			}

	# Clustering Methods
	def cluster_emails(
		self,
		df: pd.DataFrame,
		valid_texts_indices_result: tuple[list[str], list[int]],
		n_clusters: Optional[int] = None,
		method: str = "kmeans",
	):
		"""
        Cluster emails based on their content.
        Args:
            df (pd.DataFrame): DataFrame containing email data.
            valid_texts_indices_result (tuple): Tuple containing valid texts and their indices.
            n_clusters (int, optional): Number of clusters to form.
            method (str): Clustering method to use ('kmeans', 'agglomerative', etc.).

        Returns:
            pd.DataFrame: DataFrame with cluster labels added.
        """

		logger.info(f"Clustering emails using method: {method}")

		all_texts, valid_indices = valid_texts_indices_result

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
			logger.info(f"Original DataFrame shape: {df.shape}")

			# Determine the optimal number of clusters if not provided
			if method == "kmeans" and n_clusters is None:
				# Try different numbers of clusters and evaluate using silhouette score
				silhouette_scores = []
				# Adjust range calculation to ensure we have at least 2 clusters
				max_clusters = min(10, max(2,
					len(df) // 10))  # Changed from 20 to 10 to get more clusters
				range_n_clusters = range(2, max_clusters + 1)
				logger.info(
					f"Attempting to find optimal clusters in range: {list(range_n_clusters)}")

				for n in tqdm(range_n_clusters, desc="Finding optimal KMeans clusters"):
					kmeans = KMeans(n_clusters=n, random_state=42,
						n_init=10)  # Increased n_init for stability
					cluster_labels = kmeans.fit_predict(tfidf_matrix)

					# Log cluster distribution
					unique_labels = np.unique(cluster_labels)

					if len(unique_labels) > 1 and tfidf_matrix.shape[0] > 1:
						try:
							silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
							silhouette_scores.append(silhouette_avg)
						except Exception as e:
							logger.warning(f"  - Could not calculate silhouette score: {str(e)}")
					else:
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
				model = KMeans(n_clusters=n_clusters, random_state=42,
					n_init=10)  # Increased n_init for stability
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
					all_text = " ".join(cluster_emails["body"].fillna(""))
					tokens = self.tokenize_text(all_text)
					most_common = Counter(tokens).most_common(10)

					# Filter out empty subjects and ensure we have valid subjects
					all_subjects = cluster_emails["subject"].fillna("").astype(str)
					all_subjects = all_subjects[all_subjects.str.strip() != ""].tolist()
					all_subjects = list(dict.fromkeys(all_subjects))

					# If we still don't have enough subjects, try to get more
					# if len(all_subjects) < 5:
					#     additional_subjects = (
					#         cluster_emails["subject"].fillna("").astype(str)
					#     )
					#     additional_subjects = additional_subjects[
					#         additional_subjects.str.strip() != ""
					#     ].tolist()
					#     all_subjects.extend(additional_subjects)
					#     all_subjects = list(dict.fromkeys(sample_subjects))[
					#         :5
					#     ]  # Remove duplicates and limit to 5

					cluster_analysis[f"Cluster {cluster_id}"] = {
						"size": len(cluster_emails),
						"common_words": [word for word, count in most_common],
						"all_subjects": all_subjects,
					}

			# Save clustering results
			result = {
				"vectorizer": vectorizer,
				"cluster_analysis": cluster_analysis,
			}

			return result
		except Exception as e:
			save_error_log(f"Error in cluster_emails: {e}")
			logger.error(f"Error in cluster_emails: {e}")
			return None

	# extract named entities
	def extract_entities(self, df: pd.DataFrame):
		"""
        Extract named entities from the email DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing email data.

        Returns:
            pd.DataFrame: DataFrame with extracted entities.
        """

		if self.ner is None:
			save_error_log("Named Entity Recognition pipeline is not initialized.")
			logger.error("Named Entity Recognition pipeline is not initialized.")
			return pd.DataFrame()

		labels = [
			"person",
			"organization",
			"date",
			"location",
		]

		logger.info("Extracting named entities from emails...")

		if not "entities" in df.columns:
			df["entities"] = pd.Series(dtype=object)

		max_length = 384

		for idx, row in tqdm(
			df.iterrows(),
			total=len(df),
			desc="Extracting named entities",
		):
			try:
				body = row.get("body", "")
				subject = row.get("subject", "")

				if pd.isna(body) and pd.isna(subject):
					continue

				all_text = f"{subject} {body}"

				_entities = {
					"person": set(),
					"organization": set(),
					"date": set(),
					"location": set(),
				}

				for i in range(0, len(all_text), max_length):
					chunk = all_text[i:i + max_length]
					if len(chunk) < 3:
						continue

					entities = self.ner.predict_entities(
						text=chunk,
						labels=labels,
						threshold=0.5,  # Adjust threshold as needed
					)

					if not entities:
						continue

					entities = [
						entity for entity in entities
						if entity["text"] not in self.stop_words and entity["score"] >= 0.5
					]

					if not entities:
						continue

					for entity in entities:
						if entity["label"] in _entities:
							_entities[entity["label"]].add(entity["text"])

				if not any(_entities.values()):
					continue

				list_entities = {}

				for label in _entities:
					list_entities[label] = list(_entities[label])

				df.at[idx, "entities"] = list_entities

			except Exception as e:
				save_error_log(f"Error processing row {idx}: {e}")
				logger.error(f"Error processing row {idx}: {e}")
				continue

		return df

	# extract sentiment
	def analyze_sentiment(self, df: pd.DataFrame):
		"""
        Analyze the sentiment of email bodies.

        Args:
            df (pd.DataFrame): DataFrame containing email data.

        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results.
        """

		if self.sentiment_analyzer is None:
			save_error_log("Sentiment analysis pipeline is not initialized.")
			logger.error("Sentiment analysis pipeline is not initialized.")
			return df

		logger.info("Analyzing sentiment of emails...")

		if not "sentiment" in df.columns:
			df["sentiment"] = pd.Series(dtype=object)

		max_length = 514

		for idx, row in tqdm(
			df.iterrows(),
			total=len(df),
			desc="Analyzing sentiment",
		):
			try:
				body = row.get("body", "")
				subject = row.get("subject", "")

				if pd.isna(body) and pd.isna(subject):
					continue

				all_text = f"{subject} {body}"

				if len(all_text) < 5:
					continue

				sentiments: list[list[dict[str, str | float]]] = []

				for i in range(0, len(all_text), max_length):
					chunk = all_text[i:i + max_length]
					if len(chunk) < 3:
						continue

					result = self.sentiment_analyzer(chunk)

					if not result:
						continue

					if isinstance(result, list) and len(result) > 0:
						sentiments.append(result)

				sentiment = {"label": "UNKNOWN", "score": 0.0}

				for sentiment in sentiments:
					if isinstance(sentiment, list) and len(sentiment) > 0:
						# loop through each sentiment in the list
						max_score = 0.0
						max_label = ""
						for s in sentiment:
							if isinstance(s, dict) and "label" in s and "score" in s:
								label = s["label"]
								score = s["score"]
								if isinstance(score, float) and score > max_score:
									max_score = score
									max_label = label

						if max_label and max_score > 0.0:
							sentiment = {
								"label": max_label,
								"score": max_score,
							}
							break

				df.at[idx, "sentiment"] = sentiment

			except Exception as e:
				save_error_log(f"Error processing row {idx}: {e}")
				logger.error(f"Error processing row {idx}: {e}")
				continue

		return df

	# link relative emails to their threads
	def link_emails_to_threads(self, df: pd.DataFrame):
		"""
        Link emails to their threads based on subject lines.

        Args:
            df (pd.DataFrame): DataFrame containing email data.

        Returns:
            pd.DataFrame: DataFrame with a new 'thread_id' column.
        """
		logger.info("Linking emails to threads...")

		df["thread_id"] = None
		threads = {}
		thread_counter = 0

		for idx, row in tqdm(df.iterrows(), total=len(df), desc="Linking email threads"):
			subject = row.get("subject", "")
			if not subject or pd.isna(subject):
				continue

			# Normalize subject by removing prefixes like "Re:", "Fw:", etc.
			base_subject = re.sub(
				r"^(re|fw|fwd|aw|wg|antwort|sv|vs|odp|betreff|betr|betr\.|re\[\d+\]):\s*",
				"",
				subject.lower().strip(),
			)

			# If a thread with this base subject already exists, add the email to it
			if base_subject in threads:
				threads[base_subject].append(idx)
			else:
				# Start a new thread
				threads[base_subject] = [idx]

		# Assign a unique thread_id to each thread
		for thread_indices in threads.values():
			if len(thread_indices) > 1:  # Only create threads for multiple emails
				df.loc[thread_indices, "thread_id"] = thread_counter
				thread_counter += 1

		logger.info(f"Identified {thread_counter} email threads.")
		return df

	def analyze_emails(
		self,
		processed_emails_df: Optional[pd.DataFrame],
		limit: Optional[int],
	):
		"""
        Analyze emails for summarization and classification.

        Args:
            processed_emails_df (pd.DataFrame, optional): DataFrame containing email data.
                If None, it will load the most recent file from input_dir.
            limit (int, optional): Limit the number of emails to analyze.

        Returns:
            pd.DataFrame: DataFrame containing analysis results.
        """
		try:
			if processed_emails_df is None:
				# Load the dataFrame from the most recent file in input_dir
				processed_emails_df = load_processed_df(self.input_dir,
					"processed_data_",
					limit=limit)

			if processed_emails_df is None or processed_emails_df.empty:
				save_error_log("No data found in the input directory.")
				logger.error("No data found in the input directory.")
				return None

			df = sort_emails_by_date(processed_emails_df)

			logger.info(f"Analyzing {len(df)} emails...")

			num_threads = max(os.cpu_count() or 1, 4)

			logger.info(f"Starting analysis with {num_threads} threads.")

			valid_texts_indices_result = self.get_valid_texts_and_indices(df, valid_num_tokens=3)

			if not valid_texts_indices_result[0]:
				save_error_log("No valid texts found for topic extraction.")
				logger.error("No valid texts found for topic extraction.")
				return pd.DataFrame()

			with FutureExecutor(max_workers=num_threads) as executor:
				futures = {
					"topic_extraction":
					executor.submit(
					self.topic_extraction,
					df,
					valid_texts_indices_result,
					n_topics=50,
					method="nmf",
					),
					"cluster_emails":
					executor.submit(
					self.cluster_emails,
					df,
					valid_texts_indices_result,
					n_clusters=None,
					method="kmeans",
					),
					"extract_entities":
					executor.submit(self.extract_entities, df),
					"analyze_sentiment":
					executor.submit(self.analyze_sentiment, df),
					"link_emails_to_threads":
					executor.submit(self.link_emails_to_threads, df),
				}

			results = {}

			for task_name, future in tqdm(futures.items(), desc="Processing All Tasks"):
				try:
					result = future.result(timeout=600)  # Set a timeout for each task
					results[task_name] = result
				except Exception as e:
					save_error_log(f"Error in task {task_name}: {e}")
					logger.error(f"Error in task {task_name}: {e}")

			# Combine results into the original DataFrame
			if "topic_extraction" in results:
				topic_results = results["topic_extraction"]
				df_with_topics: None | pd.DataFrame = topic_results.get(
					"df_with_topics", pd.DataFrame())
				if df_with_topics is not None and isinstance(df_with_topics,
					pd.DataFrame) and not df_with_topics.empty:
					df = df_with_topics.copy()
					df["dominant_topic"] = df["dominant_topic"].astype(int)
					df["topic_strength"] = df["topic_strength"].astype(float)
					df["topic_label"] = df["topic_label"].astype(str)

			if "cluster_emails" in results:
				cluster_results = results["cluster_emails"]
				if cluster_results is not None:
					df["cluster"] = cluster_results.get("cluster_labels", -1)

			if "extract_entities" in results:
				entity_results = results["extract_entities"]
				if not entity_results.empty:
					df["entities"] = entity_results["entities"]

			if "analyze_sentiment" in results:
				sentiment_results = results["analyze_sentiment"]
				if not sentiment_results.empty:
					df["sentiment"] = sentiment_results["sentiment"]

			if "link_emails_to_threads" in results:
				thread_results = results["link_emails_to_threads"]
				if not thread_results.empty:
					df["thread_id"] = thread_results["thread_id"]

			logger.info("All tasks completed successfully.")
			return df
		except Exception as e:
			save_error_log(f"Error in analyze_emails: {e}")
			logger.error(f"Error in analyze_emails: {e}")
			return None

	def save_to_pkl(
		self,
		df: pd.DataFrame,
	):
		"""
        Save the DataFrame to a pickle file.

        Args:
            df (pd.DataFrame): DataFrame to save.
        """
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

		output_filename = f"analysis_results_{timestamp}.pkl"

		output_path = os.path.join(self.output_dir, output_filename)
		df.to_pickle(output_path)
		logger.info(f"DataFrame saved to {output_path}")

	def save_to_database(self, df: pd.DataFrame, db: DatabaseManager):
		"""Save a Pandas DataFrame to a database table."""
		if df is None or df.empty:
			logger.warning("DataFrame is empty. Skipping database save.")
			return

		if db is None:
			logger.error("Database connection is not provided. Skipping database save.")
			return

		try:
			table_name = "analysis_results"
			columns = [
				"message_id",
				"main_id",
				"filename",
				"type",
				"date",
				"from",
				"to",
				"subject",
				"body",
				"entities",
				"sentiment",
				"dominant_topic",
				"topic_strength",
				"topic_label",
				"cluster",
				"thread_id",
			]
			db.insert_from_dataframe(df, table_name, columns=columns)
			logger.success(f"✅ Successfully saved {len(df)} rows to database table '{table_name}'")
		except Exception as e:
			logger.error(f"Error saving DataFrame to database: {e}")
