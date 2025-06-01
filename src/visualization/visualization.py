import json
import os
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
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

    def visualize_all(self):
        """
        Run all visualizations sequentially.
        """
        df = self.load_data()
        self.plot_email_volume_over_time(df)
        self.plot_sender_network(df)
        self.analyze_sentiment(df)
        self.generate_sentiment_wordclouds(df)
        self.visualize_wordcloud(df)

        # Future:
        # self.plot_topic_visualization()
        # self.plot_cluster_visualization()
        # self.plot_entity_relationships()
        # self.plot_som()

        print("All visualizations created.")


# Entry point
if __name__ == "__main__":
    visualizer = Visualization()
    visualizer.visualize_all()
