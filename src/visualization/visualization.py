import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from loguru import logger
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from wordcloud import STOPWORDS, WordCloud


class Visualization:
    """
    Class responsible for visualization tasks:
    - Email volume over time
    - Network analysis of email communications
    """

    def __init__(
        self,
        input_dir="./analysis_results/",
        #        analysis_dir="./output/analysis_results/",
        output_dir="./visualizations/",
    ):
        self.input_dir = input_dir
        #        self.analysis_dir = analysis_dir
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, input_dir: Optional[str] = None):
        """
        Load the first JSON file found in the given directory.

        Args:
            input_dir (str): Directory to search for JSON files.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame with parsed dates.
        """

        input_dir = input_dir or self.input_dir or "./analysis_results/"
        # List all JSON files
        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {input_dir}")

        # Pick the first JSON file found
        filename = json_files[0]
        path = os.path.join(input_dir, filename)
        print(f"📂 Loading data from: {path}")

        # Load JSON
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Parse 'date' column if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            print("⚠️ 'date' column not found. Skipping datetime parsing.")

        return df

    # def load_data(self, filename="analysis_results.json"):
    #     """
    #     Load processed email data from JSON file.
    #     Returns a pandas DataFrame.
    #     """
    #     path = os.path.join(self.analysis_dir, filename)
    #     with open(path, "r") as f:
    #         data = json.load(f)
    #     df = pd.DataFrame(data)
    #     df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #     return df

    # def load_data(self, file_path: Optional[str] = None, skip=False):
    #     """
    #     Load processed email data from a file.
    #
    #     Args:
    #         file_path (str, optional): Path to the processed data file.
    #             If not provided, the most recent file in input_dir will be used.
    #         skip (bool): Whether to skip loading data if a file is found in output_dir.
    #
    #     Returns:
    #         pandas.DataFrame: DataFrame containing the processed email data
    #     """
    #
    #     try:
    #         if file_path is None:
    #             # Find the most recent processed data file
    #             search_file_pattern ="analysis_results"
    #             pkl_files = [
    #                 f
    #                 for f in os.listdir(self.input_dir)
    #                 if f.startswith(search_file_pattern) and f.endswith(".pkl")
    #             ]
    #             json_files = [
    #                 f
    #                 for f in os.listdir(self.input_dir)
    #                 if f.startswith(search_file_pattern) and f.endswith(".json")
    #             ]
    #             logger.info(f"Found {len(pkl_files)} pkl or {len(json_files)} json processed data files in {self.input_dir}")
    #             if not pkl_files and not json_files:
    #                 logger.error(f"No processed data files found in {self.input_dir}")
    #                 return pd.DataFrame()
    #
    #             if pkl_files:
    #                 # Sort by timestamp in filename
    #                 pkl_files.sort(reverse=True)
    #                 file_path = os.path.join(self.input_dir, pkl_files[0])
    #                 df = pd.DataFrame(pd.read_pickle(file_path))
    #                 logger.info(f"Loaded data from {file_path}: {len(df)} emails")
    #                 return df
    #
    #             if json_files:
    #                 # Sort by timestamp in filename
    #                 json_files.sort(reverse=True)
    #                 file_path = os.path.join(self.input_dir, json_files[0])
    #                 df = pd.DataFrame(pd.read_json(file_path))
    #                 logger.info(f"Loaded data from {file_path}: {len(df)} emails")
    #                 return df
    #
    #             if file_path is None:
    #                 logger.error("No file path provided for loading data.")
    #                 return pd.DataFrame()
    #
    #     except Exception as e:
    #         logger.error(f"Error loading data from {file_path}: {e}")
    #         return pd.DataFrame()

    def plot_email_volume_over_time(self, df, freq="W"):
        """
        Plot number of emails sent over time.
        """
        time_series = df.set_index("date").resample(freq).size()
        plt.figure(figsize=(10, 5))
        time_series.plot()
        plt.title("Email Volume Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Emails")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "email_volume_over_time.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_sender_network(self, df):
        """
        Generate a cleaned-up email communication network graph.

        Shows the top 20 most active senders and their connections.
        Saves the visualization to output_dir.
        """

        # Count emails sent per sender
        top_senders = df["from"].value_counts().nlargest(50).index

        # Filter emails between top senders only
        filtered_df = df[df["from"].isin(top_senders) & df["to"].isin(top_senders)]

        # Count the number of emails between each pair
        edge_counts = Counter(zip(filtered_df["from"], filtered_df["to"]))

        # Build directed graph
        G = nx.DiGraph()
        for (sender, receiver), count in edge_counts.items():
            G.add_edge(sender, receiver, weight=count)

        # Layout (spring layout for better spacing)
        pos = nx.spring_layout(G, k=0.7, iterations=100)

        # Node sizes based on out-degree (emails sent)
        node_sizes = [float(G.out_degree(n)) * 300.0 for n in G.nodes]

        # Edge widths based on weight (number of emails)
        edge_widths = [float(G[u][v]["weight"]) * 0.1 for u, v in G.edges]

        # Plotting
        plt.figure(figsize=(14, 10))
        nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes), node_size=node_sizes, node_color="skyblue")  # type: ignore
        # Draw each edge individually with its corresponding width
        for (u, v), width in zip(G.edges, edge_widths):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=width, alpha=0.6, edge_color="gray", arrows=True
            )
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title("Top 50 Email Senders Network")
        plt.axis("off")

        # Save plot
        path = os.path.join(self.output_dir, "email_sender_network.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {path}")

    def analyze_sentiment(self, df):
        """
        Performs sentiment analysis on email text (subject + body).
        Adds a 'sentiment' column to the DataFrame and visualizes sentiment distribution.
        Saves the plot as a bar chart.
        """

        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity  # type: ignore

        # Combine subject and body
        df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

        # Calculate polarity score
        df["polarity"] = df["text"].apply(get_sentiment)

        # Classify polarity
        def label_sentiment(p):
            if p > 0.1:
                return "positive"
            elif p < -0.1:
                return "negative"
            else:
                return "neutral"

        df["sentiment"] = df["polarity"].apply(label_sentiment)

        # Plot sentiment distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(
            data=df,
            x="sentiment",
            hue="sentiment",
            order=["positive", "neutral", "negative"],
            palette="coolwarm",
            legend=False,
        )
        plt.title("Email Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Emails")

        # Save plot
        path = os.path.join(self.output_dir, "email_sentiment_distribution.png")
        plt.savefig(path)
        plt.close()

        print(f"Saved: {path}")

        return df  # return updated DataFrame with sentiment

    def generate_sentiment_wordclouds(self, df):
        """
        Generates word clouds for each sentiment category (positive, neutral, negative).
        Saves images to the visualizations folder.
        """
        # Combine subject and body text
        df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

        for sentiment in ["positive", "neutral", "negative"]:
            subset = df[df["sentiment"] == sentiment]

            # Join all text entries into one big string
            combined_text = " ".join(subset["text"].dropna().tolist())

            if not combined_text.strip():
                print(f"No content found for {sentiment} sentiment. Skipping...")
                continue

            # Generate word cloud
            wordcloud = WordCloud(
                width=1000, height=600, background_color="white", colormap="coolwarm", max_words=200
            ).generate(combined_text)

            # Plot and save
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{sentiment.capitalize()} Email Word Cloud", fontsize=16)

            path = os.path.join(self.output_dir, f"wordcloud_{sentiment}.png")
            plt.savefig(path)
            plt.close()

        print(f"Saved: {path}")

    def visualize_wordcloud(self, df, text_column="body", output_name="wordcloud.png"):
        """
        Generate and save a word cloud from a DataFrame column.

        Args:
            df (DataFrame): Pandas DataFrame containing text data.
            text_column (str): Column in the DataFrame with the text (default: "body").
            output_name (str): Name of the output image file (default: "wordcloud.png").
        """
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found in DataFrame.")
            return

        # Combine all text into a single string
        text = " ".join(df[text_column].dropna().astype(str))

        # Create the word cloud
        wordcloud = WordCloud(
            width=1000, height=500, background_color="white", stopwords=STOPWORDS
        ).generate(text)

        # Save to file
        path = os.path.join(self.output_dir, output_name)
        wordcloud.to_file(path)
        print(f"saved: {path}")

    def plot_topic_visualization(self, df, text_column="body", num_topics=5, num_words=20):
        """
        Train LDA from DataFrame and plot a word cloud for each topic.
        """
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found in DataFrame.")
            return

        # Preprocess text
        def preprocess(text):
            text = str(text).lower()
            text = re.sub(r"[^a-z\s]", "", text)
            tokens = text.split()
            return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        texts = df[text_column].dropna().apply(preprocess).tolist()

        # Build dictionary and corpus
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Train LDA model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)

        # Generate word clouds for each topic
        for i, topic in lda_model.show_topics(formatted=False, num_words=num_words):
            word_freq = dict(topic)
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate_from_frequencies(word_freq)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Topic {i}")
            plt.tight_layout()

            # Save each wordcloud as a separate file
            path = os.path.join(self.output_dir, f"topic_{i}_wordcloud.png")
            wordcloud.to_file(path)
            print(f"Saved word cloud for topic {i} to: {path}")

    def plot_topic_wordcloud_for_keyword(
        self, df, keyword, text_column="body", num_topics=5, num_words=20
    ):
        filtered_df = df[df[text_column].str.contains(keyword, case=False, na=False)]

        if filtered_df.empty:
            print(f"No documents found with keyword '{keyword}'.")
            return

        print(
            f"Found {len(filtered_df)} documents with keyword '{keyword}'. Generating topic word clouds..."
        )
        self.plot_topic_visualization(
            filtered_df, text_column=text_column, num_topics=num_topics, num_words=num_words
        )

    def plot_cluster_visualization(self, df, text_column="body", num_clusters=5):
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found in DataFrame.")
            return

        # Get email texts and drop NaNs
        texts = df[text_column].dropna().astype(str).tolist()
        if not texts:
            print("No text data found to cluster.")
            return

        # Vectorize texts using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(texts)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        # Reduce dimensions for visualization
        pca = PCA(n_components=2, random_state=42)
        X_reduced = pca.fit_transform(X.toarray())

        # Plot clusters
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.title(f"KMeans Clustering of Emails ({num_clusters} clusters)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()

        # Save figure
        path = os.path.join(self.output_dir, "cluster_visualization.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved cluster visualization to: {path}")

    def plot_entity_relationships(
        self, df, sender_col="from", receiver_col="to", top_senders=10, top_receivers_per_sender=5
    ):
        if sender_col not in df.columns or receiver_col not in df.columns:
            print(f"Columns '{sender_col}' or '{receiver_col}' not found in DataFrame.")
            return

        # Count emails per sender
        sender_counts = df[sender_col].value_counts().head(top_senders).index.tolist()

        # Filter dataset to only top senders
        filtered_df = df[df[sender_col].isin(sender_counts)]

        G = nx.DiGraph()

        for sender in sender_counts:
            # Filter rows for this sender
            sender_emails = filtered_df[filtered_df[sender_col] == sender]

            # Count receivers for this sender
            receivers_list = []
            for recips in sender_emails[receiver_col].dropna():
                if isinstance(recips, str):
                    receivers_list.extend([r.strip() for r in recips.split(",")])
                elif isinstance(recips, list):
                    receivers_list.extend(recips)
            receiver_counts = (
                pd.Series(receivers_list).value_counts().head(top_receivers_per_sender)
            )

            # Add edges for top receivers
            for receiver, count in receiver_counts.items():
                if receiver:
                    G.add_edge(sender, receiver, weight=count)

        if len(G) == 0:
            print("No relationships found to plot.")
            return

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
        # Draw each edge individually with its corresponding width
        for (u, v), w in zip(G.edges(), weights):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                arrowstyle="->",
                arrowsize=10,
                edge_color="gray",
                width=w * 0.5,
            )
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title(f"Top {top_senders} Senders and their Top {top_receivers_per_sender} Receivers")
        plt.axis("off")

        path = os.path.join(self.output_dir, "entity_relationship_network_limited.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Saved limited entity relationship network to: {path}")

    def plot_som(self, df, text_column="body", som_x=10, som_y=10):
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found.")
            return

        texts = df[text_column].dropna().astype(str).tolist()
        if not texts:
            print("No text data available for SOM.")
            return

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out()

        # Initialize SOM
        som = MiniSom(
            x=som_x, y=som_y, input_len=X.shape[1], sigma=1, learning_rate=0.5, random_seed=42
        )
        som.train_random(X, num_iteration=100)

        # Map emails to SOM neurons
        neuron_to_texts = defaultdict(list)
        for i, x in enumerate(X):
            w = som.winner(x)
            neuron_to_texts[w].append(i)

        # Count hits per SOM neuron
        hit_map = np.zeros((som_x, som_y))
        for (x, y), idxs in neuron_to_texts.items():
            hit_map[x, y] = len(idxs)

        # Create labels for each neuron based on most important TF-IDF terms
        neuron_labels = {}
        for (x, y), idxs in neuron_to_texts.items():
            combined_text = " ".join([texts[i] for i in idxs])
            if combined_text.strip():
                tfidf_local = TfidfVectorizer(stop_words="english", max_features=1)
                top_term = tfidf_local.fit([combined_text]).get_feature_names_out()
                if len(top_term) > 0:
                    neuron_labels[(x, y)] = top_term[0]

        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(hit_map.T, origin="lower", cmap="Blues")
        plt.colorbar(label="Number of Emails")
        plt.title("SOM with Topic Labels")
        plt.xticks(np.arange(som_x))
        plt.yticks(np.arange(som_y))
        plt.grid(True, linestyle="--", linewidth=0.5)

        # Overlay labels
        for (x, y), label in neuron_labels.items():
            plt.text(
                x, y, label, ha="center", va="center", fontsize=8, color="black", weight="bold"
            )

        # Save the figure
        path = os.path.join(self.output_dir, "som_with_labels.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Saved SOM with topic labels to: {path}")

    def visualize_all(self):
        """
        Run all visualizations sequentially.
        """
        df = self.load_data()
        # self.plot_email_volume_over_time(df)
        # self.plot_sender_network(df)
        # self.analyze_sentiment(df)
        # self.generate_sentiment_wordclouds(df)
        # self.visualize_wordcloud(df)
        # self.plot_topic_visualization(df)  # Placeholder for LDA model visualization
        # self.plot_topic_wordcloud_for_keyword(df, keyword="accenture")
        # self.plot_cluster_visualization(df)
        # self.plot_entity_relationships(df, top_senders=4, top_receivers_per_sender=2)
        # self.plot_som(df)

        # 1. Plot how email traffic changes over time (e.g., daily/monthly volume)
        self.plot_email_volume_over_time(df)

        # 2. Visualize the communication network — who sends messages to whom
        self.plot_sender_network(df)

        # 3. Perform sentiment analysis (positive/negative/neutral) on email content
        self.analyze_sentiment(df)

        # 4. Generate separate word clouds for each sentiment category
        self.generate_sentiment_wordclouds(df)

        # 5. Generate a general word cloud showing the most frequent words in all emails
        self.visualize_wordcloud(df)

        # 6. Visualize discovered topics from an LDA model (e.g., using pyLDAvis or bar plots)
        self.plot_topic_visualization(df)  # Placeholder for LDA model visualization

        # 7. Show a word cloud of words related to a specific keyword in topic clusters
        self.plot_topic_wordcloud_for_keyword(df, keyword="accenture")

        # 8. Visualize clustered emails (e.g., using PCA/t-SNE) to show semantic groupings
        self.plot_cluster_visualization(df)

        # 9. Plot a limited sender-receiver relationship graph (entity interaction network)
        self.plot_entity_relationships(df, top_senders=4, top_receivers_per_sender=2)

        # 10. Self-Organizing Map (SOM): Clustering emails and labeling grid cells with top terms
        self.plot_som(df)

        print("All visualizations created.")


# Entry point
if __name__ == "__main__":
    visualizer = Visualization()
    visualizer.visualize_all()
