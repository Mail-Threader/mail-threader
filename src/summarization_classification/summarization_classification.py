import os
import json
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import spacy
import spacy.cli.download as download
from transformers import pipeline as hf_pipeline
import sqlite3
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

nltk.download("punkt")
#nltk.download('punkt_tab')


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
        Initialize the SummarizationClassification object with input and output directories.
        Loads the spaCy NER model and Hugging Face sentiment pipeline.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            self.ner_model = spacy.load("en_core_web_sm")
            self.ner_model.max_length = 2_000_000 # increase max length (in characters)
        except OSError as e:
            logger.info("Model not found. Downloading...")
            download("en_core_web_sm")
            self.ner_model = spacy.load("en_core_web_sm")
        self.sentiment_pipeline = hf_pipeline("sentiment-analysis")

    CUSTOM_STOPWORDS = {
        'thanks', 'fyi', 'attached', 'mail', 'doc', 'com', 'xls', 'pdf',
        'sent', '20', '2001', '11', '713', 'http', 'www', "said", "new", "email", "shes",
        "message", "original", "mailto", "subject", "thursday", "state", "november", "heres", "look",
        "jones", "lay", "enron"
    }

    def load_json_emails(self, filename="clean_emails.json"):
        """
        Loads email data from a JSON file.

        Args:
            filename (str): Name of the JSON file to load (default is 'clean_emails.json').

        Returns:
            pd.DataFrame: DataFrame of emails
        """
        full_path = os.path.join(self.input_dir, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def load_pkl_emails(self, filename="clean_emails.json"):
        """
        Loads email data from a JSON file.

        Args:
            filename (str): Name of the JSON file to load (default is 'clean_emails.json').

        Returns:
            pd.DataFrame: DataFrame of emails
        """
        full_path = os.path.join(self.input_dir, filename)
        logger.info(f"path is {full_path}")
        try:
            df = pd.read_pickle(full_path)
            return df
        except:
            logger.error("Error loading pickle file")
            return pd.DataFrame()

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_text_column(self, df, column="body", new_column="clean_body"):
        """
        Clean a text column by removing excessive whitespace.
        """
        df = df.copy()
        df[new_column] = df[column].fillna("").str.replace(r'\s+', ' ', regex=True)
        return df

    def tokenize_column(self, df: pd.DataFrame, text_column: str, new_column: str = "tokens") -> pd.DataFrame:
        """
        Tokenizes the specified text column into a list of word tokens.

        Args:
            df: Input DataFrame.
            text_column: Column in the DataFrame that contains text to tokenize.
            new_column: Name of the column where tokens will be stored.

        Returns:
            DataFrame with an additional column containing tokens.
        """
        df = df.copy()
        df[new_column] = df[text_column].fillna("").apply(word_tokenize)
        return df

    def vectorize_document(self, documents, max_features=5000):
        # Preprocess documents first
        cleaned_docs = [self.preprocess_text(doc) for doc in documents]

        # Vectorizer with custom token pattern and stopwords
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',  # keeps words with 3+ letters only
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned_docs)

        # Manually filter out additional custom stopwords
        terms = np.array(vectorizer.get_feature_names_out())
        valid_indices = [i for i, term in enumerate(terms) if term not in self.CUSTOM_STOPWORDS]

        if valid_indices:
            tfidf_matrix = tfidf_matrix[:, valid_indices]
            filtered_terms = terms[valid_indices]
            vectorizer.vocabulary_ = {term: i for i, term in enumerate(filtered_terms)}

        return tfidf_matrix, vectorizer

    def create_vectorizer_model_pipeline(self, num_clusters=5):
        """
        Creates a pipeline for TF-IDF vectorization followed by KMeans clustering.

        Args:
            num_clusters: Number of clusters for KMeans.

        Returns:
            A scikit-learn Pipeline object.
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('kmeans', KMeans(n_clusters=num_clusters, random_state=42))
        ])
        return pipeline

    def generate_cluster_topics(self, documents, labels, tfidf_matrix, vectorizer, top_n=6):
        """
        Generates top keywords per cluster/topic.
        """
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
        """
        Extracts top TF-IDF words across all documents.

        Args:
            tfidf_matrix: TF-IDF matrix.
            vectorizer: Fitted TfidfVectorizer instance.
            top_n: Number of top terms to return.

        Returns:
            List of top words with highest average TF-IDF scores.
        """
        mean_scores = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:top_n]
        return vectorizer.get_feature_names_out()[top_indices].tolist()

    def cluster_documents(self, tfidf_matrix, method="kmeans", **kwargs):
        """
        Clusters documents using the specified method (KMeans or DBSCAN).
        Args:
            tfidf_matrix: TF-IDF matrix.
            method: 'kmeans' or 'dbscan'
            kwargs: Additional arguments for the clustering model.
        Returns:
            Fitted model and cluster labels.
        """
        if method == "kmeans":
            model = KMeans(n_clusters=kwargs.get("n_clusters", 5), random_state=42)
        elif method == "dbscan":
            model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
        else:
            raise ValueError("Unsupported clustering method. Choose 'kmeans' or 'dbscan'.")
        labels = model.fit_predict(tfidf_matrix)
        return model, labels



    def extract_entities(self, df: pd.DataFrame, text_column: str = "body", chunk_size=500_000) -> pd.DataFrame:
        """
        Extract named entities (PERSON, ORG, GPE) from a text column using threading + chunking.
        """
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
        df[['persons', 'organizations', 'locations']] = df[text_column].fillna("").apply(extract_single)
        return df


    def analyze_sentiment(self, df: pd.DataFrame, text_column: str = "body", batch_size: int = 32, max_length: int = 512) -> pd.DataFrame:
        """

        Performs sentiment analysis on text data using batching and truncation.

        Args:
            df: DataFrame containing the text data.
            text_column: Column in the DataFrame with text.
            batch_size: Number of samples to process at once.
            max_length: Maximum length of input per text item.

        Returns:
            DataFrame with a new 'sentiment' column.
        """
        texts = df[text_column].fillna("").apply(lambda x: x[:max_length]).tolist()
        sentiments = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing Sentiment"):
            batch = texts[i:i + batch_size]
            try:
                results = self.sentiment_pipeline(batch, truncation=True)
                batch_sentiments = [res['label'] if isinstance(res, dict) else "ERROR" for res in results]
            except Exception as e:
                batch_sentiments = ["ERROR"] * len(batch)
            sentiments.extend(batch_sentiments)

        df = df.copy()
        df['sentiment'] = sentiments
        return df

    def summarize_corpus(self, df: pd.DataFrame, text_column: str = "body", max_sentences: int = 5, max_input_sentences: int = 1000) -> str:
        """
        Generates an extractive summary from the full corpus using TF-IDF and cosine similarity.
        Trims text to avoid exceeding spaCy/max input memory limits..
        """

        full_text = " ".join(df[text_column].dropna())
        full_text = full_text[:1_000_000] # salfly limit the length to avoid spaCy issues
        sentences = sent_tokenize(full_text)

        if len(sentences) > max_input_sentences:
            sentences = sentences[:max_input_sentences]

        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix)
        scores = sim_matrix.sum(axis=1)
        top_indices = np.argsort(scores)[-max_sentences:]
        top_indices.sort()
        summary = " ".join([sentences[i] for i in top_indices])
        return summary


    def save_to_json(self, df, filename):
        output_path = os.path.join(self.output_dir, filename)
        df.to_json(output_path, orient="records", indent=4)

    def save_to_sqlite(self, df, db_filename="summarization_results.db", table_name="emails"):
        db_path = os.path.join(self.output_dir, db_filename)


        # Convert list columns (like persons/orgs/locations) to comma-separated strings
        df = df.copy()
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)

        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()

# Main block for independant execution
if __name__ == "__main__":
    # Create SummarizationClassification instance
    analyzer = SummarizationClassification()

    df = analyzer.load_json_emails("data_preparation/clean_emails.json")
    df = analyzer.clean_text_column(df)
    df = analyzer.tokenize_column(df, text_column="clean_body")
    X, vectorizer = analyzer.vectorize_document(df["clean_body"])
    model, labels = analyzer.cluster_documents(X, method="kmeans", n_clusters=5)
    df["cluster"] = labels
    topics = analyzer.generate_cluster_topics(df["clean_body"], labels, X, vectorizer)
    print("Cluster Topics:", topics)
    top_words = analyzer.extract_top_words(X, vectorizer)
    print("Top overall words:", top_words)
    df = analyzer.extract_entities(df)
    df = analyzer.analyze_sentiment(df)
    summary = analyzer.summarize_corpus(df)
    print("Corpus Summary:\n", summary)
    analyzer.save_to_csv(df, "email_analysis_results.csv")
    print("Done!")
