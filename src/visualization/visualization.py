import os
import pickle
import re
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from utils import save_to_postgresql, upload_to_supabase

try:
    import sompy
    from sompy.visualization.bmuhits import BmuHitsView
    from sompy.visualization.mapview import View2D

    SOMPY_AVAILABLE = True
except ImportError:
    SOMPY_AVAILABLE = False
    logger.warning("sompy package not available. Kohonen map visualization will be disabled.")


class Visualization:
    """
    Class responsible for visualization tasks:
    - Email volume over time
    - Network analysis of email communications
    - Topic visualization
    - Cluster visualization
    - Entity relationship visualization
    - Self-organizing maps (Kohonen maps)
    """

    def __init__(
        self,
        input_dir="./processed_data/",
        analysis_dir="./analysis_results/",
        output_dir="./visualizations/",
        save_to_supabase=False,
    ):
        """
        Initialize the Visualization class.

        Args:
            input_dir (str): Directory containing processed email data
            analysis_dir (str): Directory containing analysis results
            output_dir (str): Directory to store visualizations
        """
        self.input_dir = input_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.save_to_supabase = save_to_supabase

        # Create an output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set the default style for matplotlib
        plt.style.use("seaborn-v0_8-whitegrid")

        # Set the default figure size
        plt.rcParams["figure.figsize"] = (12, 8)

    def load_data(self, file_path=None):
        """
        Load processed email data from a file.

        Args:
            file_path (str, optional): Path to the processed data file.
                If not provided, the most recent file in input_dir will be used.

        Returns:
            pandas.DataFrame: DataFrame containing the processed email data
        """
        if file_path is None:
            # Find the most recent processed data file
            pkl_files = [
                f
                for f in os.listdir(self.input_dir)
                if f.startswith("processed_emails_") and f.endswith(".pkl")
            ]
            if not pkl_files:
                logger.error(f"No processed data files found in {self.input_dir}")
                return pd.DataFrame()

            # Sort by timestamp in filename
            pkl_files.sort(reverse=True)
            file_path = os.path.join(self.input_dir, pkl_files[0])

        try:
            df = pd.read_pickle(file_path)
            logger.info(f"Loaded data from {file_path}: {len(df)} emails")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()

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

            # Sort by timestamp in filename
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

    def visualize_email_volume(self, df):
        """
        Visualize email volume over time.

        Args:
            df (pandas.DataFrame): DataFrame containing email data

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Visualizing email volume over time")

        # Check if the date column exists and has valid data
        if "date" not in df.columns or df["date"].isna().all():
            logger.warning("No date information available for email volume visualization")
            return None

        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                return None

        # Drop rows with missing dates
        df_with_date = df.dropna(subset=["date"]).copy()
        if len(df_with_date) == 0:
            logger.warning("No valid dates available for email volume visualization")
            return None

        # Create time-based features
        df_with_date["year"] = df_with_date["date"].dt.year
        df_with_date["month"] = df_with_date["date"].dt.month
        df_with_date["day"] = df_with_date["date"].dt.day
        df_with_date["hour"] = df_with_date["date"].dt.hour
        df_with_date["weekday"] = df_with_date["date"].dt.weekday

        # Create a date string for grouping by month
        df_with_date["month_year"] = df_with_date["date"].dt.strftime("%Y-%m")

        # Create visualizations
        fig = plt.figure(figsize=(20, 15))

        # 1. Email volume by month
        plt.subplot(2, 2, 1)
        monthly_counts = df_with_date["month_year"].value_counts().sort_index()
        monthly_counts.plot(kind="line", marker="o")
        plt.title("Email Volume by Month")
        plt.xlabel("Month")
        plt.ylabel("Number of Emails")
        plt.xticks(rotation=45)
        plt.grid(True)

        # 2. Email volume by day of week
        plt.subplot(2, 2, 2)
        weekday_counts = df_with_date["weekday"].value_counts().sort_index()
        weekday_counts.index = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekday_counts.plot(kind="bar")
        plt.title("Email Volume by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Number of Emails")
        plt.grid(True)

        # 3. Email volume by hour of day
        plt.subplot(2, 2, 3)
        hour_counts = df_with_date["hour"].value_counts().sort_index()
        hour_counts.plot(kind="bar")
        plt.title("Email Volume by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Emails")
        plt.grid(True)

        # 4. Email volume heatmap (day of week vs. hour of day)
        plt.subplot(2, 2, 4)
        heatmap_data = pd.crosstab(df_with_date["weekday"], df_with_date["hour"])
        # Set y-axis tick labels to weekday names
        sns.heatmap(
            heatmap_data,
            cmap="viridis",
            annot=False,
            fmt="d",
            yticklabels=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        plt.title("Email Volume Heatmap (Day of Week vs. Hour of Day)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")

        plt.tight_layout()

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"email_volume_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved email volume visualization to {output_path}")

        res_obj = {
            "supabase_url": None,
            "output_path": output_path,
        }

        if self.save_to_supabase:
            # Upload to Supabase
            supabase_url = upload_to_supabase(output_path)
            if supabase_url:
                logger.info(f"Uploaded email volume visualization to Supabase: {supabase_url}")
                res_obj["supabase_url"] = supabase_url

        return res_obj

    def visualize_email_network(self, df, max_nodes=100):
        """
        Visualize the email communication network.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            max_nodes (int): Maximum number of nodes to include in the visualization

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Visualizing email communication network")

        # Extract email addresses
        email_pattern = r"[\w\.-]+@[\w\.-]+"

        # Create a directed graph
        G = nx.DiGraph()

        # Track email frequencies
        sender_counts = Counter()
        recipient_counts = Counter()
        edge_weights = Counter()

        # Process each email
        for _, row in df.iterrows():
            if row["from"] is not None:
                sender_emails = re.findall(email_pattern, row["from"])
            else:
                sender_emails = []
            if row["to"] is not None:
                recipient_emails = re.findall(email_pattern, row["to"])
            else:
                recipient_emails = []

            # Add edges from sender to recipients
            for sender in sender_emails:
                sender_counts[sender] += 1
                for recipient in recipient_emails:
                    recipient_counts[recipient] += 1
                    edge_weights[(sender, recipient)] += 1

        # Get the most frequent senders and recipients
        top_senders = [email for email, _ in sender_counts.most_common(max_nodes // 2)]
        top_recipients = [email for email, _ in recipient_counts.most_common(max_nodes // 2)]

        # Create a set of important nodes
        important_nodes = set(top_senders + top_recipients)

        # Add nodes and edges to the graph
        for (sender, recipient), weight in edge_weights.items():
            if sender in important_nodes and recipient in important_nodes:
                if not G.has_node(sender):
                    G.add_node(sender, type="sender", count=sender_counts[sender])
                if not G.has_node(recipient):
                    G.add_node(recipient, type="recipient", count=recipient_counts[recipient])
                G.add_edge(sender, recipient, weight=weight)

        # Create the visualization
        plt.figure(figsize=(15, 15))

        # Calculate node sizes based on frequency
        node_sizes = [G.nodes[node]["count"] * 10 for node in G.nodes]

        # Calculate edge widths based on weight
        edge_widths = [G[u][v]["weight"] / 5 for u, v in G.edges]

        # Set node colors based on type
        node_colors = [
            "skyblue" if G.nodes[node]["type"] == "sender" else "lightgreen" for node in G.nodes
        ]

        # Use spring layout for node positioning
        pos = nx.spring_layout(G, k=0.3, iterations=50)

        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            alpha=0.5,
            edge_color="gray",
            arrows=True,
            arrowsize=10,
        )

        # Add labels to the most important nodes
        top_nodes = sorted(G.nodes, key=lambda x: G.nodes[x]["count"], reverse=True)[:20]
        labels = {
            node: node.split("@")[0] for node in top_nodes
        }  # Show only username part of email
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")

        plt.title("Email Communication Network")
        plt.axis("off")

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"email_network_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved email network visualization to {output_path}")

        res_obj = {
            "supabase_url": None,
            "output_path": output_path,
        }

        if self.save_to_supabase:
            # Upload to Supabase
            supabase_url = upload_to_supabase(output_path)
            if supabase_url:
                logger.info(f"Uploaded email network visualization to Supabase: {supabase_url}")
                res_obj["supabase_url"] = supabase_url

        return res_obj

    def visualize_topics(self, analysis_results):
        """
        Visualize topics extracted from email content.

        Args:
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Visualizing topics")

        if (
            analysis_results is None
            or "topics" not in analysis_results
            or analysis_results["topics"] is None
        ):
            logger.warning("No topic modeling results available for visualization")
            return None

        topics = analysis_results["topics"]["topics"]

        # Create word clouds for each topic
        n_topics = len(topics)
        n_cols = 2
        n_rows = (n_topics + n_cols - 1) // n_cols  # Ceiling division

        fig = plt.figure(figsize=(15, 5 * n_rows))

        for i, (topic_name, keywords) in enumerate(topics.items()):
            plt.subplot(n_rows, n_cols, i + 1)

            # Create a word cloud
            word_freq = {word: 1.0 - 0.05 * j for j, word in enumerate(keywords)}
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                prefer_horizontal=1.0,
            ).generate_from_frequencies(word_freq)

            plt.imshow(wordcloud, interpolation="bilinear")
            plt.title(f"{topic_name}: {', '.join(keywords[:5])}")
            plt.axis("off")

        plt.tight_layout()

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"topics_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        res_obj = {
            "supabase_url_static": None,
            "supabase_url_html": None,
            "output_path": output_path,
        }

        if self.save_to_supabase:
            # Upload static visualization to Supabase
            supabase_url_static = upload_to_supabase(output_path)
            if supabase_url_static:
                logger.info(f"Uploaded topic visualization to Supabase: {supabase_url_static}")
                res_obj["supabase_url_static"] = supabase_url_static

        # Create an interactive visualization with Plotly
        html_path = None
        try:
            # Prepare data for visualization
            topic_data = []
            for topic_name, keywords in topics.items():
                for i, keyword in enumerate(keywords[:10]):  # Top 10 keywords
                    topic_data.append(
                        {
                            "Topic": topic_name,
                            "Keyword": keyword,
                            "Importance": 1.0 - 0.05 * i,  # Decreasing importance
                            "Rank": i + 1,
                        }
                    )

            topic_df = pd.DataFrame(topic_data)

            # Create a horizontal bar chart
            fig = px.bar(
                topic_df,
                y="Keyword",
                x="Importance",
                color="Topic",
                facet_col="Topic",
                facet_col_wrap=2,
                height=100 * n_rows,
                width=1000,
                labels={"Importance": "Keyword Importance", "Keyword": ""},
                title="Top Keywords by Topic",
            )

            # Update layout
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))

            # Save as HTML
            html_path = os.path.join(self.output_dir, f"topics_interactive_{timestamp}.html")
            fig.write_html(html_path)
            logger.info(f"Saved interactive topic visualization to {html_path}")

            if self.save_to_supabase:
                # Upload interactive visualization to Supabase
                supabase_url_html = upload_to_supabase(html_path)
                if supabase_url_html:
                    logger.info(
                        f"Uploaded interactive topic visualization to Supabase: {supabase_url_html}"
                    )
                    res_obj["supabase_url_html"] = supabase_url_html

        except Exception as e:
            logger.error(f"Error creating interactive topic visualization: {e}")

        logger.info(f"Saved topic visualization to {output_path}")

        return res_obj

    def visualize_clusters(self, df, analysis_results):
        """
        Visualize email clusters.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Visualizing email clusters")

        if (
            analysis_results is None
            or "clusters" not in analysis_results
            or analysis_results["clusters"] is None
        ):
            logger.warning("No clustering results available for visualization")
            return None

        cluster_analysis = analysis_results["clusters"]["cluster_analysis"]

        # Create a bar chart of cluster sizes
        plt.figure(figsize=(12, 6))

        cluster_sizes = {
            cluster_name: info["size"] for cluster_name, info in cluster_analysis.items()
        }
        cluster_df = pd.DataFrame(
            {
                "Cluster": list(cluster_sizes.keys()),
                "Size": list(cluster_sizes.values()),
            }
        )

        # Sort by cluster size
        cluster_df = cluster_df.sort_values("Size", ascending=False)

        # Plot
        sns.barplot(x="Cluster", y="Size", data=cluster_df)
        plt.title("Email Cluster Sizes")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Emails")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"clusters_size_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        res_obj = {
            "output_path": output_path,
            "supabase_url_size": None,
            "supabase_url_wordcloud": None,
        }

        if self.save_to_supabase:
            # Upload cluster size visualization to Supabase
            supabase_url_size = upload_to_supabase(output_path)
            if supabase_url_size:
                logger.info(f"Uploaded cluster size visualization to Supabase: {supabase_url_size}")
                res_obj["supabase_url_size"] = supabase_url_size

        # Create word clouds for each cluster
        n_clusters = len(cluster_analysis)
        n_cols = 2
        n_rows = (n_clusters + n_cols - 1) // n_cols  # Ceiling division

        fig = plt.figure(figsize=(15, 5 * n_rows))

        for i, (cluster_name, info) in enumerate(cluster_analysis.items()):
            plt.subplot(n_rows, n_cols, i + 1)

            # Create a word cloud
            word_freq = {word: 1.0 - 0.05 * j for j, word in enumerate(info["common_words"])}
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                prefer_horizontal=1.0,
            ).generate_from_frequencies(word_freq)

            plt.imshow(wordcloud, interpolation="bilinear")
            plt.title(f"{cluster_name} (Size: {info['size']})")
            plt.axis("off")

        plt.tight_layout()

        # Save the figure
        wordcloud_path = os.path.join(self.output_dir, f"clusters_wordcloud_{timestamp}.png")
        plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
        plt.close()

        if self.save_to_supabase:
            # Upload wordcloud visualization to Supabase
            supabase_url_wordcloud = upload_to_supabase(wordcloud_path)
            if supabase_url_wordcloud:
                logger.info(
                    f"Uploaded cluster wordcloud visualization to Supabase: {supabase_url_wordcloud}"
                )
                res_obj["supabase_url_wordcloud"] = supabase_url_wordcloud

        logger.info(f"Saved cluster visualizations to {output_path} and {wordcloud_path}")
        return res_obj

    def visualize_entities(self, analysis_results):
        """
        Visualize named entities extracted from email content.

        Args:
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Visualizing named entities")

        if (
            analysis_results is None
            or "entities" not in analysis_results
            or analysis_results["entities"] is None
        ):
            logger.warning("No entity extraction results available for visualization")
            return None

        entities = analysis_results["entities"]

        # Create a figure with subplots for each entity type
        entity_types = list(entities.keys())
        n_types = len(entity_types)

        fig = plt.figure(figsize=(15, 5 * n_types))

        for i, entity_type in enumerate(entity_types):
            plt.subplot(n_types, 1, i + 1)

            # Get the entity counts
            entity_counts = entities[entity_type]
            if not entity_counts:
                plt.text(
                    0.5,
                    0.5,
                    f"No {entity_type} entities found",
                    ha="center",
                    va="center",
                )
                plt.axis("off")
                continue

            # Create a DataFrame for plotting
            entity_df = pd.DataFrame(entity_counts, columns=["Entity", "Count"])

            # Sort by count
            entity_df = entity_df.sort_values("Count", ascending=False).head(20)

            # Plot
            sns.barplot(x="Count", y="Entity", data=entity_df)
            plt.title(f"Top {entity_type} Entities")
            plt.xlabel("Frequency")
            plt.ylabel(entity_type)
            plt.grid(True, axis="x")

        plt.tight_layout()

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"entities_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved entity visualization to {output_path}")

        res_obj = {
            "supabase_url": None,
            "output_path": output_path,
        }

        if self.save_to_supabase:
            # Upload to Supabase
            supabase_url = upload_to_supabase(output_path)
            if supabase_url:
                logger.info(f"Uploaded entity visualization to Supabase: {supabase_url}")
                res_obj["supabase_url"] = supabase_url

        return res_obj

    def visualize_sentiment(self, analysis_results):
        """
        Visualize sentiment analysis results.

        Args:
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Visualizing sentiment analysis")

        if (
            analysis_results is None
            or "sentiment" not in analysis_results
            or analysis_results["sentiment"] is None
        ):
            logger.warning("No sentiment analysis results available for visualization")
            return None

        sentiment_results = analysis_results["sentiment"]

        # Create a pie chart of sentiment distribution
        plt.figure(figsize=(10, 10))

        distribution = sentiment_results["distribution"]
        labels = list(distribution.keys())
        sizes = list(distribution.values())

        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["lightgreen", "lightcoral"],
        )
        plt.axis("equal")
        plt.title("Email Sentiment Distribution")

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"sentiment_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved sentiment visualization to {output_path}")

        res_obj = {
            "supabase_url": None,
            "output_path": output_path,
        }

        if self.save_to_supabase:
            # Upload to Supabase
            supabase_url = upload_to_supabase(output_path)
            if supabase_url:
                logger.info(f"Uploaded sentiment visualization to Supabase: {supabase_url}")
                res_obj["supabase_url"] = supabase_url

        return res_obj

    def create_kohonen_map(self, df, feature_cols=None, map_size=(20, 20)):
        """
        Create a Kohonen self-organizing map (SOM) visualization.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            feature_cols (list, optional): List of columns to use as features.
                If None, TF-IDF features will be generated from the email body.
            map_size (tuple): Size of the SOM grid

        Returns:
            str: Path to the saved visualization
        """
        logger.info("Creating Kohonen self-organizing map")

        try:
            if not SOMPY_AVAILABLE:
                logger.warning(
                    "sompy package not available. Kohonen map visualization is disabled."
                )
                return None

            # If feature columns are not provided, generate TF-IDF features
            if feature_cols is None:
                # Use clean_body column if available, otherwise use body
                text_column = "clean_body" if "clean_body" in df.columns else "body"

                # Create TF-IDF matrix
                from sklearn.feature_extraction.text import TfidfVectorizer

                vectorizer = TfidfVectorizer(
                    max_features=100, min_df=5, max_df=0.8, stop_words="english"
                )

                # Fit vectorizer and transform documents
                tfidf_matrix = vectorizer.fit_transform(df[text_column].fillna(""))

                # Convert to dense array
                X = tfidf_matrix.toarray()
                feature_names = vectorizer.get_feature_names_out()
            else:
                # Use provided feature columns
                X = df[feature_cols].values
                feature_names = feature_cols

            # Normalize the data
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Create and train the SOM
            som = sompy.SOMFactory.build(
                X_scaled,
                mapsize=map_size,
                normalization="var",
                initialization="pca",
                component_names=feature_names,
            )

            som.train(n_job=1, verbose="debug")

            # Create visualizations
            # 1. U-Matrix (unified distance matrix)
            view = View2D(sompy.umatrix, som.codebook.mapsize, "U-Matrix", text_size=8)
            view.show(som=som, col_sz=4)

            # 2. Component planes
            # Select a subset of components to visualize
            n_components = min(16, len(feature_names))
            selected_components = np.random.choice(len(feature_names), n_components, replace=False)

            view2 = View2D(
                som.codebook.matrix,
                som.codebook.mapsize,
                "Component Planes",
                text_size=8,
            )
            view2.show(som=som, what="codebook", col_sz=4)

            # 3. BMU (best matching unit) hits
            vhts = BmuHitsView(som, height=10, title="BMU Hits")
            vhts.show(som=som)

            # Save the figures
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            u_matrix_path = os.path.join(self.output_dir, f"som_umatrix_{timestamp}.png")
            component_path = os.path.join(self.output_dir, f"som_components_{timestamp}.png")
            hits_path = os.path.join(self.output_dir, f"som_hits_{timestamp}.png")

            view.save(u_matrix_path, dpi=300, bbox_inches="tight")
            view2.save(component_path, dpi=300, bbox_inches="tight")
            vhts.save(hits_path, dpi=300, bbox_inches="tight")

            plt.close("all")

            logger.info("Saved Kohonen map visualizations")

            res_obj = {
                "u_matrix_path": u_matrix_path,
                "component_path": component_path,
                "hits_path": hits_path,
                "supabase_url_components": None,
                "supabase_url_hits": None,
                "supabase_url_umatrix": None,
            }

            if self.save_to_supabase:
                # Upload to Supabase
                supabase_url_umatrix = upload_to_supabase(u_matrix_path)
                if supabase_url_umatrix:
                    logger.info(
                        f"Uploaded SOM U-Matrix visualization to Supabase: {supabase_url_umatrix}"
                    )
                    res_obj["supabase_url_umatrix"] = supabase_url_umatrix

                supabase_url_components = upload_to_supabase(component_path)
                if supabase_url_components:
                    logger.info(
                        f"Uploaded SOM Components visualization to Supabase: {supabase_url_components}"
                    )
                    res_obj["supabase_url_components"] = supabase_url_components

                supabase_url_hits = upload_to_supabase(hits_path)
                if supabase_url_hits:
                    logger.info(f"Uploaded SOM Hits visualization to Supabase: {supabase_url_hits}")
                    res_obj["supabase_url_hits"] = supabase_url_hits

            logger.info(
                f"Saved Kohonen map visualizations to {u_matrix_path}, {component_path}, and {hits_path}"
            )

            return res_obj

        except Exception as e:
            print(e.with_traceback(None))
            logger.error(f"Error creating Kohonen map: {e}")
            return None

    def create_dashboard(self, df, analysis_results):
        """
        Create an interactive dashboard with Plotly.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            str: Path to the saved dashboard
        """
        logger.info("Creating interactive dashboard")

        try:
            # Create a subplot figure
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "Email Volume Over Time",
                    "Top Senders",
                    "Email Sentiment Distribution",
                    "Cluster Sizes",
                    "Topic Distribution",
                    "Entity Counts",
                ),
                specs=[
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}],
                ],
            )

            # 1. Email Volume Over Time
            if "date" in df.columns and not df["date"].isna().all():
                # Convert date column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                    try:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    except Exception:
                        pass

                # Drop rows with missing dates
                df_with_date = df.dropna(subset=["date"]).copy()
                if len(df_with_date) > 0:
                    # Group by month
                    df_with_date["month_year"] = df_with_date["date"].dt.strftime("%Y-%m")
                    monthly_counts = df_with_date["month_year"].value_counts().sort_index()

                    fig.add_trace(
                        go.Scatter(
                            x=monthly_counts.index,
                            y=monthly_counts.values,
                            mode="lines+markers",
                            name="Email Volume",
                        ),
                        row=1,
                        col=1,
                    )

            # 2. Top Senders
            email_pattern = r"[\w\.-]+@[\w\.-]+"
            sender_counts = Counter()

            for _, row in df.iterrows():
                if row["from"] is not None:
                    sender_emails = re.findall(email_pattern, row["from"])
                else:
                    sender_emails = []
                for email in sender_emails:
                    sender_counts[email] += 1

            top_senders = pd.DataFrame(sender_counts.most_common(10), columns=["Sender", "Count"])

            fig.add_trace(
                go.Bar(
                    x=top_senders["Count"],
                    y=top_senders["Sender"],
                    orientation="h",
                    name="Top Senders",
                ),
                row=1,
                col=2,
            )

            # 3. Sentiment Distribution
            if (
                analysis_results
                and "sentiment" in analysis_results
                and analysis_results["sentiment"]
            ):
                distribution = analysis_results["sentiment"]["distribution"]
                labels = list(distribution.keys())
                sizes = list(distribution.values())

                fig.add_trace(go.Pie(labels=labels, values=sizes, name="Sentiment"), row=2, col=1)

            # 4. Cluster Sizes
            if analysis_results and "clusters" in analysis_results and analysis_results["clusters"]:
                cluster_analysis = analysis_results["clusters"]["cluster_analysis"]
                cluster_sizes = {
                    cluster_name: info["size"] for cluster_name, info in cluster_analysis.items()
                }
                cluster_df = pd.DataFrame(
                    {
                        "Cluster": list(cluster_sizes.keys()),
                        "Size": list(cluster_sizes.values()),
                    }
                ).sort_values("Size", ascending=False)

                fig.add_trace(
                    go.Bar(
                        x=cluster_df["Cluster"],
                        y=cluster_df["Size"],
                        name="Cluster Sizes",
                    ),
                    row=2,
                    col=2,
                )

            # 5. Topic Distribution
            if analysis_results and "topics" in analysis_results and analysis_results["topics"]:
                # Get document-topic distribution
                doc_topic_matrix = analysis_results["topics"]["doc_topic_matrix"]
                topic_counts = np.sum(doc_topic_matrix, axis=0)
                topic_names = [f"Topic {i + 1}" for i in range(len(topic_counts))]

                fig.add_trace(
                    go.Bar(x=topic_names, y=topic_counts, name="Topic Distribution"),
                    row=3,
                    col=1,
                )

            # 6. Entity Counts
            if analysis_results and "entities" in analysis_results and analysis_results["entities"]:
                entities = analysis_results["entities"]
                entity_counts = {}

                # Combine all entity types
                for entity_type, counts in entities.items():
                    for entity, count in counts[:5]:  # Take top 5 from each type
                        entity_counts[f"{entity} ({entity_type})"] = count

                # Convert to DataFrame
                entity_df = (
                    pd.DataFrame(
                        {
                            "Entity": list(entity_counts.keys()),
                            "Count": list(entity_counts.values()),
                        }
                    )
                    .sort_values("Count", ascending=False)
                    .head(10)
                )

                fig.add_trace(
                    go.Bar(
                        x=entity_df["Entity"],
                        y=entity_df["Count"],
                        name="Entity Counts",
                    ),
                    row=3,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                height=1200,
                width=1200,
                title_text="Enron Email Analysis Dashboard",
                showlegend=False,
            )

            # Save as HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")
            fig.write_html(output_path)

            logger.info(f"Saved interactive dashboard to {output_path}")

            # Upload to Supabase
            supabase_url = upload_to_supabase(output_path)
            if supabase_url:
                logger.info(f"Uploaded interactive dashboard to Supabase: {supabase_url}")

            return output_path

        except Exception as e:
            print(e.with_traceback(None))
            logger.error(f"Error creating dashboard: {e}")
            return None

    @staticmethod
    def save_file_url_to_database(file_dict: dict[str, str]):
        """
        Save the file URL to the database.
        Args:
            file_dict (dict): Dictionary containing file URLs
        """
        # create a DataFrame from the file_dict
        df = pd.DataFrame(file_dict.items(), columns=["file_type", "file_url"])
        # save the DataFrame to PostgreSQL
        table_name = "visualization_data"
        success_message = f"Saved data to PostgreSQL table: {table_name}"
        save_to_postgresql(
            df,
            table_name=table_name,
            success_message=success_message,
            if_exists="replace",
        )

    def visualize_all(self, df=None, analysis_results=None):
        """
        Create all visualizations.

        Args:
            df (pandas.DataFrame, optional): DataFrame containing email data.
                If not provided, data will be loaded from the most recent file.
            analysis_results (dict, optional): Dictionary containing analysis results.
                If not provided, results will be loaded from the most recent file.

        Returns:
            dict: Dictionary containing paths to all visualizations
        """

        try:
            # Load data if not provided
            if df is None or df.empty:
                df = self.load_data()
                if df.empty:
                    logger.error("No data available for visualization")
                    return {}

            # Load analysis results if not provided
            if analysis_results is None:
                analysis_results = self.load_analysis_results()

            logger.info(f"Creating all visualizations for {len(df)} emails")

            # Create all visualizations
            visualization_paths = {}
            supabase_paths = {}

            # 1. Email volume over time
            volume_path_res = self.visualize_email_volume(df)
            if volume_path_res is not None and volume_path_res.get("output_path"):
                visualization_paths["email_volume"] = volume_path_res["output_path"]
                supabase_paths["email_volume"] = volume_path_res["supabase_url"]

            # 2. Email network
            network_path = self.visualize_email_network(df)
            if network_path:
                visualization_paths["email_network"] = network_path.get("output_path")
                supabase_paths["email_network"] = network_path.get("supabase_url")

            # 3. Topics
            if analysis_results and "topics" in analysis_results and analysis_results["topics"]:
                topics_path = self.visualize_topics(analysis_results)
                if topics_path:
                    visualization_paths["topics"] = topics_path.get("output_path")
                    supabase_paths["topics"] = topics_path.get("supabase_url_static")

            # 4. Clusters
            if analysis_results and "clusters" in analysis_results and analysis_results["clusters"]:
                clusters_path = self.visualize_clusters(df, analysis_results)
                if clusters_path:
                    visualization_paths["clusters"] = clusters_path.get("output_path")
                    supabase_paths["clusters_size"] = clusters_path.get("supabase_url_size")
                    supabase_paths["clusters_wordcloud"] = clusters_path.get(
                        "supabase_url_wordcloud"
                    )

            # 5. Entities
            if analysis_results and "entities" in analysis_results and analysis_results["entities"]:
                entities_path = self.visualize_entities(analysis_results)
                if entities_path:
                    visualization_paths["entities"] = entities_path.get("output_path")
                    supabase_paths["entities"] = entities_path.get("supabase_url")

            # 6. Sentiment
            if (
                analysis_results
                and "sentiment" in analysis_results
                and analysis_results["sentiment"]
            ):
                sentiment_path = self.visualize_sentiment(analysis_results)
                if sentiment_path:
                    visualization_paths["sentiment"] = sentiment_path.get("output_path")
                    supabase_paths["sentiment"] = sentiment_path.get("supabase_url")

            # 7. Kohonen map
            kohonen_path = self.create_kohonen_map(df)
            if kohonen_path:
                visualization_paths["kohonen_map"] = kohonen_path.get("u_matrix_path")
                supabase_paths["kohonen_map"] = kohonen_path.get("supabase_url_umatrix")
                supabase_paths["kohonen_map_components"] = kohonen_path.get(
                    "supabase_url_components"
                )
                supabase_paths["kohonen_map_hits"] = kohonen_path.get("supabase_url_hits")

            # Save file URLs to the database
            if self.save_to_supabase:
                self.save_file_url_to_database(supabase_paths)

            # 8. Dashboard
            dashboard_path = self.create_dashboard(df, analysis_results)
            if dashboard_path:
                visualization_paths["dashboard"] = dashboard_path

            logger.info(f"Created {len(visualization_paths)} visualizations")
            return visualization_paths

        except Exception as e:
            print(e.with_traceback(None))
            logger.error(f"Error creating visualizations {e}")


if __name__ == "__main__":
    # Create Visualization instance
    visualizer = Visualization()

    # Load data
    df = visualizer.load_data()

    # Load analysis results
    analysis_results = visualizer.load_analysis_results()

    # Create all visualizations
    visualization_paths = visualizer.visualize_all(df, analysis_results)

    print("Done!")
