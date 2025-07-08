import os
import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from loguru import logger
import spacy
import spacy.cli.download as download
from transformers import pipeline as hf_pipeline
import re
from tqdm import tqdm

nltk.download("punkt")


class SummarizationClassification:
	"""
	Author - Steven Chiddima Adams
	"""

	def __init__(self,
		input_dir="./processed_data/",
		output_dir="./analysis_results/",
		summarization_model="facebook/bart-large-cnn"):
		self.input_dir = input_dir
		self.output_dir = output_dir

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		try:
			self.ner_model = spacy.load("en_core_web_sm")
			self.ner_model.max_length = 2_000_000
		except OSError:
			logger.info("Model not found. Downloading...")
			download("en_core_web_sm")
			self.ner_model = spacy.load("en_core_web_sm")

		self.sentiment_pipeline = hf_pipeline("sentiment-analysis")
		self.summarizer = hf_pipeline("summarization", model=summarization_model)

	CUSTOM_STOPWORDS = {
		'thanks', 'fyi', 'attached', 'mail', 'doc', 'com', 'xls', 'pdf', 'sent', '20', '2001', '11',
		'713', 'http', 'www', "said", "new", "email", "shes", "message", "original", "mailto",
		"subject", "thursday", "state", "november", "heres", "look", "jones", "lay", "enron"
	}

	def load_json_emails(self, filename="clean_emails.json"):
		full_path = os.path.join(self.input_dir, filename)
		with open(full_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		return pd.DataFrame(data)

	def preprocess_text(self, text: str) -> str:
		text = text.lower()
		text = re.sub(r'\s+', ' ', text)
		return text.strip()

	def clean_text_column(self, df, column="body", new_column="clean_body"):
		df = df.copy()
		df[new_column] = df[column].fillna("").str.replace(r'\s+', ' ', regex=True)
		return df

	def tokenize_column(self, df: pd.DataFrame, text_column: str,
		new_column: str = "tokens") -> pd.DataFrame:
		df = df.copy()
		df[new_column] = df[text_column].fillna("").apply(word_tokenize)
		return df

	def vectorize_document(self, documents, max_features=5000):
		cleaned_docs = [self.preprocess_text(doc) for doc in documents]
		vectorizer = TfidfVectorizer(stop_words='english',
			max_features=max_features,
			token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
		tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
		terms = np.array(vectorizer.get_feature_names_out())
		valid_indices = [i for i, term in enumerate(terms) if term not in self.CUSTOM_STOPWORDS]

		if valid_indices:
			tfidf_matrix = tfidf_matrix[:, valid_indices]
			filtered_terms = terms[valid_indices]
			vectorizer.vocabulary_ = {term: i for i, term in enumerate(filtered_terms)}

		return tfidf_matrix, vectorizer

	def create_vectorizer_model_pipeline(self, num_clusters=5):
		return Pipeline([('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
			('kmeans', KMeans(n_clusters=num_clusters, random_state=42))])

	def generate_cluster_topics(self, documents, labels, tfidf_matrix, vectorizer, top_n=6):
		terms = np.array(list(vectorizer.vocabulary_.keys()))
		topics = {}
		for cluster_id in sorted(set(labels)):
			mask = np.array(labels) == cluster_id
			if not np.any(mask):
				topics[cluster_id] = []
				continue
			cluster_matrix = tfidf_matrix[mask].mean(axis=0)
			top_indices = cluster_matrix.A1.argsort()[::-1][:top_n]
			topics[cluster_id] = terms[top_indices].tolist()
		return topics

	def extract_top_words(self, tfidf_matrix, vectorizer, top_n=10):
		mean_scores = tfidf_matrix.mean(axis=0).A1
		top_indices = mean_scores.argsort()[::-1][:top_n]
		return vectorizer.get_feature_names_out()[top_indices].tolist()

	def cluster_documents(self, tfidf_matrix, method="kmeans", **kwargs):
		if method == "kmeans":
			model = KMeans(n_clusters=kwargs.get("n_clusters", 5), random_state=42)
		elif method == "dbscan":
			model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
		else:
			raise ValueError("Unsupported clustering method. Choose 'kmeans' or 'dbscan'.")
		labels = model.fit_predict(tfidf_matrix)
		return model, labels

	def extract_entities(self, df: pd.DataFrame, text_column: str = "body",
		chunk_size=500_000) -> pd.DataFrame:

		def extract_single(text):
			persons, orgs, locations = [], [], []
			chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
			for chunk in chunks:
				try:
					doc = self.ner_model(chunk)
					persons.extend([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
					orgs.extend([ent.text for ent in doc.ents if ent.label_ == "ORG"])
					locations.extend([ent.text for ent in doc.ents if ent.label_ == "GPE"])
				except Exception:
					continue
			return pd.Series([persons, orgs, locations])

		df[['persons', 'organizations',
			'locations']] = df[text_column].fillna("").apply(extract_single)
		return df

	def analyze_sentiment(self,
		df: pd.DataFrame,
		text_column: str = "body",
		batch_size: int = 32,
		max_length: int = 512) -> pd.DataFrame:
		texts = df[text_column].fillna("").apply(lambda x: x[:max_length]).tolist()
		sentiments = []

		for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing Sentiment"):
			batch = texts[i:i + batch_size]
			try:
				results = self.sentiment_pipeline(batch, truncation=True)
				batch_sentiments = [
					res['label'] if isinstance(res, dict) else "ERROR" for res in results
				]
			except Exception:
				batch_sentiments = ["ERROR"] * len(batch)
			sentiments.extend(batch_sentiments)

		df = df.copy()
		df['sentiment'] = sentiments
		return df

	def summarize_abstractive(self,
		df: pd.DataFrame,
		text_column: str = "body",
		max_length: int = 130,
		min_length: int = 30,
		return_dataframe: bool = True,
		json_output_file: str = "summarized_emails.json",
		batch_size: int = 8) -> pd.DataFrame:
		df = df.copy()
		texts = df[text_column].fillna("").apply(lambda x: x[:1024]).tolist()
		summaries = []

		for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing Emails"):
			batch = texts[i:i + batch_size]
			try:
				outputs = self.summarizer(batch,
					max_length=max_length,
					min_length=min_length,
					do_sample=False)
				batch_summaries = [out["summary_text"] for out in outputs]
			except Exception as e:
				batch_summaries = [f"Error: {str(e)}"] * len(batch)
			summaries.extend(batch_summaries)

		df["summary"] = summaries

		records = df.to_dict(orient="records")
		with open(os.path.join(self.output_dir, json_output_file), "w", encoding="utf-8") as f:
			json.dump(records, f, ensure_ascii=False, indent=2)

		return df if return_dataframe else None

	def summarize_corpus(
		self,
		df: pd.DataFrame,
		text_column: str = "clean_body",
		max_length: int = 130,
		min_length: int = 30,
		batch_size: int = 8,
		json_output_file: str = "corpus_summaries.json",
	) -> pd.DataFrame:
		texts = df[text_column].fillna("").apply(lambda x: x[:1024]).tolist()
		summaries = []

		for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing Corpus"):
			batch = texts[i:i + batch_size]
			try:
				outputs = self.summarizer(batch,
					max_length=max_length,
					min_length=min_length,
					do_sample=False)
				batch_summaries = [out["summary_text"] for out in outputs]
			except Exception as e:
				batch_summaries = [f"Error: {str(e)}"] * len(batch)
			summaries.extend(batch_summaries)

		df = df.copy()
		df["summary"] = summaries

		records = df.to_dict(orient="records")
		with open(os.path.join(self.output_dir, json_output_file), "w", encoding="utf-8") as f:
			json.dump(records, f, ensure_ascii=False, indent=2)

		return df

	def save_to_json(self, df: pd.DataFrame, json_output_file: str):
		records = df.to_dict(orient="records")
		with open(json_output_file, "w", encoding="utf-8") as f:
			json.dump(records, f, ensure_ascii=False, indent=2)
