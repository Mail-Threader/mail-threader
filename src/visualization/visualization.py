import logging
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

from utils import upload_to_supabase  # save_to_postgresql,

logging.getLogger("matplotlib").setLevel(logging.ERROR)

SOMPY_AVAILABLE = False

try:
	import sompy
	from sompy.visualization.bmuhits import BmuHitsView
	from sompy.visualization.mapview import View2D

	SOMPY_AVAILABLE = True
except ImportError:
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
		batch_size=25,
		time_period_months=6,
	):
		"""
		Initialize the Visualization class.

		Args:
			input_dir (str): Directory containing processed email data
			analysis_dir (str): Directory containing analysis results
			output_dir (str): Directory to store visualizations
			save_to_supabase (bool): Whether to save visualizations to Supabase
			batch_size (int): Number of items to process in each batch for visualizations
			time_period_months (int): Time period in months for batching (1, 6, or 12)
		"""
		self.input_dir = input_dir
		self.analysis_dir = analysis_dir
		self.output_dir = output_dir
		self.save_to_supabase = save_to_supabase
		self.batch_size = batch_size
		self.time_period_months = time_period_months

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
				f for f in os.listdir(self.input_dir)
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
				f for f in os.listdir(self.analysis_dir)
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
				# Suppress date parsing warnings for cleaner output
				import warnings

				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", message=".*Parsing dates.*")
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

		# Map numeric weekdays to names for the ones present in data
		weekday_names = {
			0: "Monday",
			1: "Tuesday",
			2: "Wednesday",
			3: "Thursday",
			4: "Friday",
			5: "Saturday",
			6: "Sunday",
		}
		weekday_counts.index = [weekday_names.get(i, f"Day {i}") for i in weekday_counts.index]
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

	def visualize_email_network(self, df, max_nodes=50):
		"""
		Visualize the email communication network with batch processing.

		Args:
			df (pandas.DataFrame): DataFrame containing email data
			max_nodes (int): Maximum number of nodes to include in the visualization

		Returns:
			dict: Paths to the saved visualizations
		"""
		logger.info("Visualizing email communication network")

		# Sample data for better visualization
		df_sampled = self.sample_data_for_visualization(df, max_items=self.batch_size * 2)

		# Extract email addresses
		email_pattern = r"[\w\.-]+@[\w\.-]+"

		# Create a directed graph
		G = nx.DiGraph()

		# Track email frequencies
		sender_counts = Counter()
		recipient_counts = Counter()
		edge_weights = Counter()

		# Process each email
		for _, row in df_sampled.iterrows():
			if row["from"] is not None:
				sender_emails = re.findall(email_pattern, str(row["from"]))
			else:
				sender_emails = []
			if row["to"] is not None:
				recipient_emails = re.findall(email_pattern, str(row["to"]))
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

		# Calculate node sizes based on frequency (ensure minimum size)
		node_sizes = [max(50, G.nodes[node]["count"] * 20) for node in G.nodes]

		# Calculate edge widths based on weight (ensure minimum width)
		edge_widths = [max(0.5, G[u][v]["weight"] / 2) for u, v in G.edges]

		# Set node colors based on type
		node_colors = [
			"skyblue" if G.nodes[node]["type"] == "sender" else "lightgreen" for node in G.nodes
		]

		# Use spring layout for node positioning
		pos = nx.spring_layout(G, k=0.5, iterations=50)

		# Draw the network
		nx.draw_networkx_nodes(
			G,
			pos,
			node_size=node_sizes,  # type: ignore
			node_color=node_colors,  # type: ignore
			alpha=0.8,
			linewidths=1,
			edgecolors="black",
		)
		nx.draw_networkx_edges(
			G,
			pos,
			width=edge_widths,  # type: ignore
			alpha=0.6,
			edge_color="gray",
			arrows=True,
			arrowsize=15,
			arrowstyle="->",
		)

		# Add labels to the most important nodes
		top_nodes = sorted(G.nodes, key=lambda x: G.nodes[x]["count"], reverse=True)[:15]
		labels = {
			node: node.split("@")[0]
			for node in top_nodes
		}  # Show only username part of email
		nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")

		plt.title(f"Email Communication Network (Sample: {len(df_sampled)} emails)")
		plt.axis("off")

		# Save the figure
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join(self.output_dir, f"email_network_{timestamp}.png")
		plt.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close()

		logger.info(f"Saved email network visualization to {output_path}")

		res_obj = {
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
		Visualize topics extracted from email content with enhanced visualizations and batch processing.

		Creates multiple visualizations including word clouds, interactive bar charts,
		topic coherence metrics, topic relationships, and topic trends over time.

		Args:
			analysis_results (dict): Dictionary containing analysis results

		Returns:
			dict: Paths to the saved visualizations
		"""
		logger.info("Visualizing topics with enhanced visualizations and batch processing")

		if (analysis_results is None or "topics" not in analysis_results
			or analysis_results["topics"] is None):
			logger.warning("No topic modeling results available for visualization")
			return None

		# Extract all topic-related data
		topic_data = analysis_results["topics"]
		topics = topic_data.get("topics", {})

		# Limit number of topics for better visualization
		if (len(topics) > self.batch_size // 5):  # Assuming 5 keywords per topic on average
			# Select top topics by coherence or document frequency
			coherence_scores = topic_data.get("coherence_scores", {})
			if coherence_scores:
				# Sort topics by coherence score
				sorted_topics = sorted(
					[(k, v) for k, v in coherence_scores.items() if k != "overall"],
					key=lambda x: x[1],
					reverse=True,
				)
				top_topic_names = [
					topic_name for topic_name, _ in sorted_topics[:self.batch_size // 5]
				]
				topics = {k: v for k, v in topics.items() if k in top_topic_names}
				logger.info(f"Limited visualization to top {len(topics)} topics based on coherence")
			else:
				# Take first batch_size//5 topics
				topic_items = list(topics.items())[:self.batch_size // 5]
				topics = dict(topic_items)
				logger.info(f"Limited visualization to first {len(topics)} topics")

		# Check for enhanced topic data from improved extract_topics method
		topic_keywords_full = topic_data.get("topic_keywords_full", {})
		topic_labels = topic_data.get("topic_labels", {})
		coherence_scores = topic_data.get("coherence_scores", {})
		doc_topic_matrix = topic_data.get("doc_topic_matrix", None)

		# Create result object to store all visualization paths
		res_obj = {}

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

		# 1. Create enhanced word clouds for each topic
		n_topics = len(topics)
		n_cols = min(3, n_topics)  # Maximum 3 columns
		n_rows = (n_topics + n_cols - 1) // n_cols  # Ceiling division

		fig = plt.figure(figsize=(18, 6 * n_rows))

		for i, (topic_name, keywords) in enumerate(topics.items()):
			plt.subplot(n_rows, n_cols, i + 1)

			# Use topic labels if available
			topic_title = topic_labels.get(topic_name, topic_name)

			# Use keyword weights if available, otherwise use decreasing weights
			if topic_keywords_full and topic_name in topic_keywords_full:
				# Extract words and weights from full keyword data
				word_weight_pairs = topic_keywords_full[topic_name]
				word_freq = {word: weight for word, weight in word_weight_pairs}
			else:
				# Fallback to simple decreasing weights
				word_freq = {word: 1.0 - 0.05 * j for j, word in enumerate(keywords)}

			# Create a word cloud with enhanced settings
			wordcloud = WordCloud(
				width=800,
				height=500,
				background_color="white",
				max_words=150,
				prefer_horizontal=0.9,
				collocations=True,
				colormap="viridis",  # Use a more visually appealing colormap
				contour_width=1,
				contour_color="steelblue",
			).generate_from_frequencies(word_freq)

			plt.imshow(wordcloud, interpolation="bilinear")

			# Add coherence score if available
			if coherence_scores and topic_name in coherence_scores:
				coherence = coherence_scores[topic_name]
				plt.title(f"{topic_title}\nCoherence: {coherence:.3f}", fontsize=14)
			else:
				plt.title(topic_title, fontsize=14)

			plt.axis("off")

		plt.tight_layout()

		# Save the figure
		wordcloud_path = os.path.join(self.output_dir, f"topics_wordcloud_{timestamp}.png")
		plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
		plt.close()
		res_obj["wordcloud_path"] = wordcloud_path
		res_obj["output_path"] = wordcloud_path  # For backward compatibility

		if self.save_to_supabase:
			# Upload static visualization to Supabase
			supabase_url = upload_to_supabase(wordcloud_path)
			if supabase_url:
				logger.info(f"Uploaded topic wordcloud to Supabase: {supabase_url}")
				res_obj["supabase_url_static"] = supabase_url

		# 2. Create an enhanced interactive visualization with Plotly
		try:
			# Prepare data for visualization with actual weights if available
			topic_data = []

			for topic_name, keywords in topics.items():
				topic_label = topic_labels.get(topic_name, topic_name)

				# Use actual weights if available
				if topic_keywords_full and topic_name in topic_keywords_full:
					for i, (keyword, weight) in enumerate(
						topic_keywords_full[topic_name][:15]):  # Top 15 keywords
						topic_data.append({
							"Topic ID": topic_name,
							"Topic": topic_label,
							"Keyword": keyword,
							"Weight": weight,
							"Rank": i + 1,
						})
				else:
					# Fallback to simple decreasing weights
					for i, keyword in enumerate(keywords[:15]):  # Top 15 keywords
						topic_data.append({
							"Topic ID": topic_name,
							"Topic": topic_label,
							"Keyword": keyword,
							"Weight": 1.0 - 0.05 * i,
							"Rank": i + 1,
						})

			topic_df = pd.DataFrame(topic_data)

			# Create a more informative horizontal bar chart
			fig = px.bar(
				topic_df,
				y="Keyword",
				x="Weight",
				color="Topic",
				facet_col="Topic",
				facet_col_wrap=2,
				height=max(120 * n_rows, 600),
				width=1200,
				labels={
				"Weight": "Keyword Importance",
				"Keyword": ""
				},
				title="Top Keywords by Topic",
				hover_data=["Rank", "Topic ID"],
				color_discrete_sequence=px.colors.qualitative.Bold,
				facet_row_spacing=0.05,
			)

			# Update layout for better readability
			fig.update_layout(
				showlegend=False,
				margin=dict(l=10, r=10, t=60, b=10),
				title_font=dict(size=20),
				font=dict(size=12),
			)

			# Customize hover template
			fig.update_traces(hovertemplate=
				"<b>%{y}</b><br>Weight: %{x:.3f}<br>Rank: %{customdata[0]}<br>Topic: %{customdata[1]}"
								)

			# Save as HTML
			interactive_path = os.path.join(self.output_dir, f"topics_interactive_{timestamp}.html")
			fig.write_html(interactive_path, include_plotlyjs="cdn")
			logger.info(f"Saved enhanced interactive topic visualization to {interactive_path}")
			res_obj["interactive_path"] = interactive_path

			if self.save_to_supabase:
				# Upload interactive visualization to Supabase
				supabase_url = upload_to_supabase(interactive_path)
				if supabase_url:
					logger.info(
						f"Uploaded interactive topic visualization to Supabase: {supabase_url}")
					res_obj["supabase_url_html"] = supabase_url

			# 3. Create topic coherence visualization if coherence scores are available
			if coherence_scores:
				# Filter out the 'overall' score if present
				topic_coherence = {k: v for k, v in coherence_scores.items() if k != "overall"}

				if topic_coherence:
					# Create a DataFrame for the coherence scores
					coherence_df = pd.DataFrame({
						"Topic":
						[topic_labels.get(topic, topic) for topic in topic_coherence.keys()],
						"Topic ID":
						list(topic_coherence.keys()),
						"Coherence":
						list(topic_coherence.values()),
					})

					# Sort by coherence score
					coherence_df = coherence_df.sort_values("Coherence", ascending=False)

					# Create a bar chart
					fig = px.bar(
						coherence_df,
						x="Topic",
						y="Coherence",
						color="Coherence",
						color_continuous_scale="Viridis",
						title="Topic Coherence Scores (Higher is Better)",
						hover_data=["Topic ID"],
						height=600,
						width=1000,
					)

					# Update layout
					fig.update_layout(
						xaxis_title="Topic",
						yaxis_title="Coherence Score",
						xaxis={"categoryorder": "total descending"},
						font=dict(size=12),
					)

					# Add a horizontal line for the overall coherence if available
					if "overall" in coherence_scores:
						fig.add_shape(
							type="line",
							x0=-0.5,
							y0=coherence_scores["overall"],
							x1=len(topic_coherence) - 0.5,
							y1=coherence_scores["overall"],
							line=dict(color="red", width=2, dash="dash"),
						)
						fig.add_annotation(
							x=len(topic_coherence) / 2,
							y=coherence_scores["overall"] * 1.05,
							text=f"Overall Coherence: {coherence_scores['overall']:.3f}",
							showarrow=False,
							font=dict(color="red", size=14),
						)

					# Save as HTML
					coherence_path = os.path.join(self.output_dir,
						f"topic_coherence_{timestamp}.html")
					fig.write_html(coherence_path, include_plotlyjs="cdn")
					logger.info(f"Saved topic coherence visualization to {coherence_path}")
					res_obj["coherence_path"] = coherence_path

			# 4. Create topic relationship visualization if doc-topic matrix is available
			if doc_topic_matrix is not None and isinstance(doc_topic_matrix, np.ndarray):
				try:
					# Calculate topic correlation matrix
					topic_corr = np.corrcoef(doc_topic_matrix.T)

					# Create a heatmap
					plt.figure(figsize=(12, 10))

					# Use topic labels if available
					if topic_labels:
						labels = [
							topic_labels.get(f"Topic {i+1}", f"Topic {i+1}")
							for i in range(topic_corr.shape[0])
						]
						# Truncate long labels
						labels = [
							label[:20] + "..." if len(label) > 20 else label for label in labels
						]
					else:
						labels = [f"Topic {i+1}" for i in range(topic_corr.shape[0])]

					# Create heatmap
					sns.heatmap(
						topic_corr,
						annot=True,
						cmap="coolwarm",
						fmt=".2f",
						linewidths=0.5,
						xticklabels=labels,
						yticklabels=labels,
					)

					plt.title("Topic Correlation Matrix", fontsize=16)
					plt.tight_layout()

					# Save the figure
					relationship_path = os.path.join(self.output_dir,
						f"topic_relationships_{timestamp}.png")
					plt.savefig(relationship_path, dpi=300, bbox_inches="tight")
					plt.close()
					res_obj["relationship_path"] = relationship_path
				except Exception as e:
					logger.error(f"Error creating topic relationship visualization: {e}")

		except Exception as e:
			logger.error(f"Error creating enhanced topic visualizations: {e}")
			import traceback

			logger.error(traceback.format_exc())

		logger.info(f"Saved topic visualizations to {self.output_dir}")

		return res_obj

	def visualize_clusters(self, df, analysis_results):
		"""
		Visualize email clusters with enhanced batch processing and multiple views.

		Args:
			df (pandas.DataFrame): DataFrame containing email data
			analysis_results (dict): Dictionary containing analysis results

		Returns:
			dict: Paths to the saved visualizations
		"""
		logger.info("Visualizing email clusters with batch processing")

		if (analysis_results is None or "clusters" not in analysis_results
			or analysis_results["clusters"] is None):
			logger.warning("No clustering results available for visualization")
			return None

		cluster_analysis = analysis_results["clusters"]["cluster_analysis"]

		# Limit number of clusters for better visualization
		if (len(cluster_analysis)
			> self.batch_size // 3):  # Assuming 3 items per cluster on average
			# Sort clusters by size and take top ones
			sorted_clusters = sorted(cluster_analysis.items(),
				key=lambda x: x[1]["size"],
				reverse=True)
			top_clusters = dict(sorted_clusters[:self.batch_size // 3])
			cluster_analysis = top_clusters
			logger.info(f"Limited cluster visualization to top {len(cluster_analysis)} clusters")

		# Create comprehensive cluster visualization
		fig, axes = plt.subplots(2, 2, figsize=(20, 16))

		# 1. Cluster sizes bar chart
		cluster_sizes = {
			cluster_name: info["size"]
			for cluster_name, info in cluster_analysis.items()
		}
		cluster_df = pd.DataFrame({
			"Cluster": list(cluster_sizes.keys()),
			"Size": list(cluster_sizes.values()),
		})

		# Sort by cluster size
		cluster_df = cluster_df.sort_values("Size", ascending=False)

		# Plot cluster sizes
		axes[0, 0].bar(
			cluster_df["Cluster"],
			cluster_df["Size"],
			color="skyblue",
			edgecolor="black",
		)
		axes[0, 0].set_title("Email Cluster Sizes")
		axes[0, 0].set_xlabel("Cluster")
		axes[0, 0].set_ylabel("Number of Emails")
		axes[0, 0].tick_params(axis="x", rotation=45)
		axes[0, 0].grid(True, axis="y", alpha=0.3)

		# 2. Cluster size distribution
		sizes = list(cluster_sizes.values())
		axes[0, 1].hist(
			sizes,
			bins=min(10, len(sizes)),
			alpha=0.7,
			color="lightgreen",
			edgecolor="black",
		)
		axes[0, 1].set_title("Cluster Size Distribution")
		axes[0, 1].set_xlabel("Cluster Size")
		axes[0, 1].set_ylabel("Frequency")
		axes[0, 1].grid(True, alpha=0.3)

		# 3. Top keywords per cluster (if available)
		if any("keywords" in info or "common_words" in info for info in cluster_analysis.values()):
			cluster_keywords = {}
			for cluster_name, info in cluster_analysis.items():
				# Support both 'keywords' and 'common_words' field names
				keywords = info.get("keywords", info.get("common_words", []))
				if keywords:
					# Take top 5 keywords per cluster
					cluster_keywords[cluster_name] = keywords[:5]

			if cluster_keywords:
				# Create a word frequency plot
				all_keywords = []
				cluster_labels = []
				for cluster, keywords in cluster_keywords.items():
					all_keywords.extend(keywords)
					cluster_labels.extend([cluster] * len(keywords))

				keyword_counts = Counter(all_keywords)
				top_keywords = dict(keyword_counts.most_common(15))

				axes[1, 0].barh(
					list(top_keywords.keys()),
					list(top_keywords.values()),
					color="coral",
					alpha=0.7,
				)
				axes[1, 0].set_title("Most Frequent Keywords Across Clusters")
				axes[1, 0].set_xlabel("Frequency")
				axes[1, 0].grid(True, alpha=0.3)

		# 4. Cluster quality metrics (if available)
		if any("quality_metrics" in info for info in cluster_analysis.values()):
			metrics = {}
			for cluster_name, info in cluster_analysis.items():
				if "quality_metrics" in info:
					metrics[cluster_name] = info["quality_metrics"]

			if metrics:
				# Plot silhouette scores or other quality metrics
				cluster_names = list(metrics.keys())
				silhouette_scores = [
					metrics[name].get("silhouette_score", 0) for name in cluster_names
				]

				axes[1, 1].bar(
					cluster_names,
					silhouette_scores,
					color="gold",
					alpha=0.7,
					edgecolor="black",
				)
				axes[1, 1].set_title("Cluster Quality (Silhouette Scores)")
				axes[1, 1].set_xlabel("Cluster")
				axes[1, 1].set_ylabel("Silhouette Score")
				axes[1, 1].tick_params(axis="x", rotation=45)
				axes[1, 1].grid(True, alpha=0.3)
		else:
			# Show cluster statistics
			stats_text = f"Total Clusters: {len(cluster_analysis)}\n"
			stats_text += f"Total Emails: {sum(cluster_sizes.values())}\n"
			stats_text += f"Largest Cluster: {max(cluster_sizes.values())} emails\n"
			stats_text += f"Smallest Cluster: {min(cluster_sizes.values())} emails\n"
			stats_text += (f"Average Cluster Size: {np.mean(list(cluster_sizes.values())):.1f}")

			axes[1, 1].text(
				0.5,
				0.5,
				stats_text,
				ha="center",
				va="center",
				fontsize=14,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
			)
			axes[1, 1].set_title("Cluster Statistics")
			axes[1, 1].axis("off")

		plt.tight_layout()

		# Save the figure
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join(self.output_dir, f"clusters_comprehensive_{timestamp}.png")
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
			# Support both 'keywords' and 'common_words' field names
			keywords = info.get("keywords", info.get("common_words", []))
			if keywords:
				word_freq = {word: 1.0 - 0.05 * j for j, word in enumerate(keywords)}
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
			else:
				# No keywords available, show cluster info
				plt.text(
					0.5,
					0.5,
					f"{cluster_name}\nSize: {info['size']}\nNo keywords available",
					ha="center",
					va="center",
					fontsize=14,
					bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
				)
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
		Visualize named entities extracted from email content with batch processing.

		Args:
			analysis_results (dict): Dictionary containing analysis results

		Returns:
			dict: Path to the saved visualization
		"""
		logger.info("Visualizing named entities with batch processing")

		if (analysis_results is None or "entities" not in analysis_results
			or analysis_results["entities"] is None):
			logger.warning("No entity extraction results available for visualization")
			return None

		entities = analysis_results["entities"]

		# Limit entities for better visualization
		entity_types = list(entities.keys())

		# Create a comprehensive entity visualization
		fig, axes = plt.subplots(2, 2, figsize=(20, 16))
		fig.suptitle("Named Entity Analysis", fontsize=16, fontweight="bold")

		# 1. Entity counts by type
		type_counts = {
			entity_type: len(entity_counts)
			for entity_type, entity_counts in entities.items()
		}

		axes[0, 0].bar(
			type_counts.keys(),
			type_counts.values(),
			color="lightblue",
			edgecolor="black",
		)
		axes[0, 0].set_title("Entity Types Distribution")
		axes[0, 0].set_xlabel("Entity Type")
		axes[0, 0].set_ylabel("Number of Unique Entities")
		axes[0, 0].tick_params(axis="x", rotation=45)
		axes[0, 0].grid(True, alpha=0.3)

		# 2. Top entities across all types
		all_entities = []
		for entity_type, entity_counts in entities.items():
			for entity, count in entity_counts[:self.batch_size //
				len(entity_types)]:  # Limit per type
				all_entities.append((entity, count, entity_type))

		# Sort by count and take top entities
		all_entities.sort(key=lambda x: x[1], reverse=True)
		top_entities = all_entities[:15]  # Top 15 entities overall

		if top_entities:
			entity_names = [
				f"{entity} ({entity_type})" for entity, count, entity_type in top_entities
			]
			entity_counts_list = [count for entity, count, entity_type in top_entities]

			axes[0, 1].barh(range(len(entity_names)), entity_counts_list, color="lightcoral")
			axes[0, 1].set_yticks(range(len(entity_names)))
			axes[0, 1].set_yticklabels(entity_names, fontsize=10)
			axes[0, 1].set_title("Top 15 Entities (All Types)")
			axes[0, 1].set_xlabel("Frequency")
			axes[0, 1].grid(True, alpha=0.3)

		# 3. Entity type heatmap showing frequency distribution
		if len(entity_types) > 1:
			# Create a matrix of entity frequencies
			max_entities_per_type = max(5, self.batch_size // len(entity_types))
			heatmap_data = []
			entity_labels = []

			for entity_type in entity_types:
				entity_counts = entities[entity_type][:max_entities_per_type]
				type_counts = [count for entity, count in entity_counts]

				# Pad with zeros if needed
				while len(type_counts) < max_entities_per_type:
					type_counts.append(0)

				heatmap_data.append(type_counts)

				# Get entity names for this type
				type_labels = [entity for entity, count in entity_counts]
				while len(type_labels) < max_entities_per_type:
					type_labels.append("")

				if not entity_labels:  # First iteration
					entity_labels = type_labels

			if heatmap_data:
				import seaborn as sns

				heatmap_df = pd.DataFrame(heatmap_data, index=entity_types, columns=entity_labels)

				sns.heatmap(
					heatmap_df,
					annot=True,
					fmt="d",
					cmap="YlOrRd",
					ax=axes[1, 0],
					cbar_kws={"label": "Frequency"},
				)
				axes[1, 0].set_title("Entity Frequency Heatmap")
				axes[1, 0].set_xlabel("Entities")
				axes[1, 0].set_ylabel("Entity Types")

		# 4. Entity statistics and word cloud
		total_entities = sum(
			sum(count for _, count in entity_counts) for entity_counts in entities.values())
		unique_entities = sum(len(entity_counts) for entity_counts in entities.values())

		# Create a word cloud of all entities
		try:
			from wordcloud import WordCloud

			# Combine all entities with their frequencies
			all_entity_freq = {}
			for entity_type, entity_counts in entities.items():
				for entity, count in entity_counts[:self.batch_size // len(entity_types)]:
					all_entity_freq[entity] = count

			if all_entity_freq:
				wordcloud = WordCloud(
					width=600,
					height=400,
					background_color="white",
					max_words=50,
					colormap="viridis",
				).generate_from_frequencies(all_entity_freq)

				axes[1, 1].imshow(wordcloud, interpolation="bilinear")
				axes[1, 1].set_title("Entity Word Cloud")
				axes[1, 1].axis("off")
			else:
				# Show statistics instead
				stats_text = f"Total Entity Mentions: {total_entities}\n"
				stats_text += f"Unique Entities: {unique_entities}\n"
				stats_text += f"Entity Types: {len(entity_types)}\n\n"
				stats_text += "Entity Types:\n" + "\n".join(
					[f"• {et}: {len(entities[et])} unique" for et in entity_types])

				axes[1, 1].text(
					0.5,
					0.5,
					stats_text,
					ha="center",
					va="center",
					fontsize=12,
					bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
				)
				axes[1, 1].set_title("Entity Statistics")
				axes[1, 1].axis("off")

		except ImportError:
			# Fallback to statistics if WordCloud is not available
			stats_text = f"Total Entity Mentions: {total_entities}\n"
			stats_text += f"Unique Entities: {unique_entities}\n"
			stats_text += f"Entity Types: {len(entity_types)}\n\n"
			stats_text += "Entity Types:\n" + "\n".join(
				[f"• {et}: {len(entities[et])} unique" for et in entity_types])

			axes[1, 1].text(
				0.5,
				0.5,
				stats_text,
				ha="center",
				va="center",
				fontsize=12,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
			)
			axes[1, 1].set_title("Entity Statistics")
			axes[1, 1].axis("off")

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
		Visualize sentiment analysis results with enhanced batch processing.

		Args:
			analysis_results (dict): Dictionary containing analysis results

		Returns:
			dict: Path to the saved visualization
		"""
		logger.info("Visualizing sentiment analysis")

		if (analysis_results is None or "sentiment" not in analysis_results
			or analysis_results["sentiment"] is None):
			logger.warning("No sentiment analysis results available for visualization")
			return None

		sentiment_results = analysis_results["sentiment"]

		# Create a comprehensive sentiment visualization
		fig, axes = plt.subplots(2, 2, figsize=(16, 12))

		# 1. Sentiment distribution pie chart
		distribution = sentiment_results["distribution"]
		labels = list(distribution.keys())
		sizes = list(distribution.values())

		colors = ["lightgreen", "lightcoral", "lightyellow", "lightblue"][:len(labels)]

		axes[0, 0].pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
		axes[0, 0].set_title("Email Sentiment Distribution")

		# 2. Sentiment scores histogram (if available)
		if "scores" in sentiment_results and sentiment_results["scores"]:
			scores = sentiment_results["scores"]
			# Sample scores for better visualization
			sampled_scores = (scores[:self.batch_size] if len(scores) > self.batch_size else scores)

			axes[0, 1].hist(sampled_scores, bins=20, alpha=0.7, edgecolor="black", color="skyblue")
			axes[0, 1].set_title(f"Sentiment Scores Distribution (Sample: {len(sampled_scores)})")
			axes[0, 1].set_xlabel("Sentiment Score")
			axes[0, 1].set_ylabel("Frequency")
			axes[0, 1].grid(True, alpha=0.3)

		# 3. Sentiment over time (if timestamps available)
		if "timestamps" in sentiment_results and sentiment_results["timestamps"]:
			timestamps = sentiment_results["timestamps"]
			sentiments = sentiment_results.get("labels", [])

			if len(timestamps) == len(sentiments):
				# Convert to DataFrame for easier processing
				import warnings

				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", message=".*Parsing dates.*")
					timestamps_parsed = pd.to_datetime(timestamps, errors="coerce")

				df_sentiment = pd.DataFrame({
					"timestamp": timestamps_parsed,
					"sentiment": sentiments,
				})

				# Drop invalid timestamps
				df_sentiment = df_sentiment.dropna(subset=["timestamp"])

				if len(df_sentiment) > 0:
					# Sample data for visualization
					if len(df_sentiment) > self.batch_size:
						df_sentiment = df_sentiment.sample(n=self.batch_size, random_state=42)

					# Group by date and calculate sentiment percentages
					df_sentiment["date"] = df_sentiment["timestamp"].dt.date
					sentiment_by_date = (df_sentiment.groupby(["date",
						"sentiment"]).size().unstack(fill_value=0))

					# Calculate percentages
					sentiment_by_date_pct = (
						sentiment_by_date.div(sentiment_by_date.sum(axis=1), axis=0) * 100)

					# Plot stacked area chart
					sentiment_by_date_pct.plot(kind="area", stacked=True, ax=axes[1, 0], alpha=0.7)
					axes[1, 0].set_title("Sentiment Trends Over Time")
					axes[1, 0].set_xlabel("Date")
					axes[1, 0].set_ylabel("Percentage")
					axes[1, 0].legend(title="Sentiment")
					axes[1, 0].grid(True, alpha=0.3)

		# 4. Confidence distribution (if available)
		if "confidence" in sentiment_results and sentiment_results["confidence"]:
			confidence_scores = sentiment_results["confidence"]
			# Sample confidence scores
			sampled_confidence = (confidence_scores[:self.batch_size]
				if len(confidence_scores) > self.batch_size else confidence_scores)

			axes[1, 1].hist(
				sampled_confidence,
				bins=15,
				alpha=0.7,
				edgecolor="black",
				color="orange",
			)
			axes[1, 1].set_title(
				f"Sentiment Confidence Distribution (Sample: {len(sampled_confidence)})")
			axes[1, 1].set_xlabel("Confidence Score")
			axes[1, 1].set_ylabel("Frequency")
			axes[1, 1].grid(True, alpha=0.3)
		else:
			# If no confidence data, show sentiment statistics
			axes[1, 1].text(
				0.5,
				0.5,
				f"Total Emails Analyzed: {sum(sizes)}\n\nSentiment Breakdown:\n" + "\n".join([
				f"{label}: {size} ({size/sum(sizes)*100:.1f}%)"
				for label, size in zip(labels, sizes)
				]),
				ha="center",
				va="center",
				fontsize=12,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
			)
			axes[1, 1].set_title("Sentiment Analysis Summary")
			axes[1, 1].axis("off")

		plt.tight_layout()

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
					"sompy package not available. Kohonen map visualization is disabled.")
				return None

			# If feature columns are not provided, generate TF-IDF features
			if feature_cols is None:
				# Use clean_body column if available, otherwise use body
				text_column = "clean_body" if "clean_body" in df.columns else "body"

				# Create TF-IDF matrix
				from sklearn.feature_extraction.text import TfidfVectorizer

				vectorizer = TfidfVectorizer(max_features=100,
					min_df=5,
					max_df=0.8,
					stop_words="english")

				# Fit vectorizer and transform documents
				tfidf_matrix = vectorizer.fit_transform(df[text_column].fillna(""))

				# Convert to dense array
				X = tfidf_matrix.toarray() if hasattr(tfidf_matrix,
					"toarray") else tfidf_matrix  # type: ignore
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
						f"Uploaded SOM U-Matrix visualization to Supabase: {supabase_url_umatrix}")
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

	def visualize_summaries(self, analysis_results):
		"""
		Visualize summarization results with multiple charts and graphs.

		Args:
			analysis_results (dict): Dictionary containing analysis results

		Returns:
			dict: Dictionary containing paths to all summary visualizations
		"""
		logger.info("Visualizing summarization results")

		if (analysis_results is None or "summaries" not in analysis_results
			or analysis_results["summaries"] is None):
			logger.warning("No summarization results available for visualization")
			return None

		summaries = analysis_results["summaries"]
		metadata = analysis_results.get("metadata", {})

		# Enhance metrics if they're missing
		for subject, styles in summaries.items():
			for style, summary in styles.items():
				if "metrics" not in summary:
					summary["metrics"] = {}

				metrics = summary["metrics"]

				# Add missing metrics with default values
				if "word_count" not in metrics:
					metrics["word_count"] = len(summary.get("summary", "").split())

				if "sentence_count" not in metrics:
					metrics["sentence_count"] = len(summary.get("summary", "").split("."))

				if "entity_count" not in metrics:
					entities = summary.get("key_information", {}).get("entities", {})
					if not entities:
						entities = summary.get("entities", {})
					metrics["entity_count"] = (sum(
						len(entity_list) if isinstance(entity_list, list) else 1
						for entity_list in entities.values()) if entities else 0)

				if "action_item_count" not in metrics:
					action_items = summary.get("key_information", {}).get("action_items", [])
					metrics["action_item_count"] = (len(action_items) if action_items else 0)

		# Create a timestamp for file naming
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		visualization_paths = {}

		try:
			# 1. Summary Length Distribution by Style
			plt.figure(figsize=(12, 6))
			style_lengths = {}
			for subject, styles in summaries.items():
				for style, summary in styles.items():
					if style not in style_lengths:
						style_lengths[style] = []
					style_lengths[style].append(summary["metrics"]["word_count"])

			# Create box plot
			box_data = [lengths for lengths in style_lengths.values()]
			box_labels = list(style_lengths.keys())

			bp = plt.boxplot(box_data)
			plt.xticks(range(1, len(box_labels) + 1), box_labels)
			plt.title("Summary Length Distribution by Style")
			plt.xlabel("Summary Style")
			plt.ylabel("Word Count")
			plt.xticks(rotation=45)
			plt.grid(True, axis="y")

			# Save the figure
			length_path = os.path.join(self.output_dir, f"summary_lengths_{timestamp}.png")
			plt.savefig(length_path, dpi=300, bbox_inches="tight")
			plt.close()
			visualization_paths["summary_lengths"] = length_path

			# 2. Entity Distribution by Style
			plt.figure(figsize=(12, 6))
			style_entities = {}
			for subject, styles in summaries.items():
				for style, summary in styles.items():
					if style not in style_entities:
						style_entities[style] = Counter()
					# Handle different possible structures for entities
					entities = summary.get("key_information", {}).get("entities", {})
					if not entities:
						# Try alternative structure
						entities = summary.get("entities", {})

					if entities:
						for entity_type, entity_list in entities.items():
							if isinstance(entity_list, list):
								style_entities[style][entity_type] += len(entity_list)
							else:
								style_entities[style][entity_type] += 1

			# Create stacked bar chart
			entity_types = set()
			for counts in style_entities.values():
				entity_types.update(counts.keys())

			if len(entity_types) == 0:
				logger.warning("No entities found for summary visualization")
				# Create a simple placeholder plot
				plt.figure(figsize=(8, 6))
				plt.text(
					0.5,
					0.5,
					"No entities found in summaries",
					horizontalalignment="center",
					verticalalignment="center",
					transform=plt.gca().transAxes,
					fontsize=16,
				)
				plt.title("Entity Distribution by Summary Style")
				plt.xlabel("Summary Style")
				plt.ylabel("Number of Entities")
			else:
				x = np.arange(len(style_entities))
				width = 0.8 / len(entity_types)

				for i, entity_type in enumerate(sorted(entity_types)):
					values = [
						style_entities[style].get(entity_type, 0)
						for style in style_entities.keys()
					]
					plt.bar(x + i * width, values, width, label=entity_type)

				plt.title("Entity Distribution by Summary Style")
				plt.xlabel("Summary Style")
				plt.ylabel("Number of Entities")
				plt.xticks(
					x + width * (len(entity_types) - 1) / 2,
					list(style_entities.keys()),
					rotation=45,
				)
				plt.legend()
				plt.grid(True, axis="y")

			# Save the figure
			entity_path = os.path.join(self.output_dir, f"summary_entities_{timestamp}.png")
			plt.savefig(entity_path, dpi=300, bbox_inches="tight")
			plt.close()
			visualization_paths["summary_entities"] = entity_path

			# 3. Action Items by Style
			plt.figure(figsize=(12, 6))
			style_actions = {}
			for subject, styles in summaries.items():
				for style, summary in styles.items():
					if style not in style_actions:
						style_actions[style] = []
					style_actions[style].append(len(summary["key_information"]["action_items"]))

			# Create bar chart
			plt.bar(
				list(style_actions.keys()),
				[sum(actions) for actions in style_actions.values()],
			)
			plt.title("Total Action Items by Summary Style")
			plt.xlabel("Summary Style")
			plt.ylabel("Number of Action Items")
			plt.xticks(rotation=45)
			plt.grid(True, axis="y")

			# Save the figure
			action_path = os.path.join(self.output_dir, f"summary_actions_{timestamp}.png")
			plt.savefig(action_path, dpi=300, bbox_inches="tight")
			plt.close()
			visualization_paths["summary_actions"] = action_path

			# 4. Interactive Summary Dashboard using Plotly
			fig = make_subplots(
				rows=2,
				cols=2,
				subplot_titles=(
				"Summary Length Distribution",
				"Entity Distribution",
				"Action Items Distribution",
				"Summary Metrics Overview",
				),
				specs=[
				[{
				"type": "box"
				}, {
				"type": "bar"
				}],
				[{
				"type": "bar"
				}, {
				"type": "scatter"
				}],
				],
			)

			# Add box plot for summary lengths
			for style, lengths in style_lengths.items():
				fig.add_trace(go.Box(y=lengths, name=style), row=1, col=1)

			# Add stacked bar chart for entities
			for entity_type in sorted(entity_types):
				values = [
					style_entities[style].get(entity_type, 0) for style in style_entities.keys()
				]
				fig.add_trace(
					go.Bar(name=entity_type, x=list(style_entities.keys()), y=values),
					row=1,
					col=2,
				)

			# Add bar chart for action items
			fig.add_trace(
				go.Bar(
				x=list(style_actions.keys()),
				y=[sum(actions) for actions in style_actions.values()],
				name="Action Items",
				),
				row=2,
				col=1,
			)

			# Add scatter plot for metrics overview
			metrics = [
				"word_count",
				"sentence_count",
				"entity_count",
				"action_item_count",
			]
			for metric in metrics:
				values = []
				for subject, styles in summaries.items():
					for style, summary in styles.items():
						values.append(summary["metrics"][metric])
				fig.add_trace(
					go.Scatter(
					x=list(range(len(values))),
					y=values,
					mode="lines+markers",
					name=metric.replace("_", " ").title(),
					),
					row=2,
					col=2,
				)

			# Update layout
			fig.update_layout(
				height=1000,
				width=1200,
				title_text="Summary Analysis Dashboard",
				showlegend=True,
				barmode="stack",
			)

			# Save as HTML
			dashboard_path = os.path.join(self.output_dir, f"summary_dashboard_{timestamp}.html")
			fig.write_html(dashboard_path)
			visualization_paths["summary_dashboard"] = dashboard_path

			# 5. Word Clouds for each summary style
			plt.figure(figsize=(15, 5 * len(style_lengths)))
			for i, (style, _) in enumerate(style_lengths.items()):
				plt.subplot(len(style_lengths), 1, i + 1)

				# Combine all summaries of this style
				all_text = " ".join([
					summaries[subject][style]["summary"] for subject in summaries
					if style in summaries[subject]
				])

				# Create word cloud
				wordcloud = WordCloud(
					width=800,
					height=400,
					background_color="white",
					max_words=100,
					prefer_horizontal=1.0,
				).generate(all_text)

				plt.imshow(wordcloud, interpolation="bilinear")
				plt.title(f"Word Cloud - {style} Style")
				plt.axis("off")

			plt.tight_layout()

			# Save the figure
			wordcloud_path = os.path.join(self.output_dir, f"summary_wordclouds_{timestamp}.png")
			plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
			plt.close()
			visualization_paths["summary_wordclouds"] = wordcloud_path

			# Upload to Supabase if enabled
			if self.save_to_supabase:
				supabase_paths = {}
				for viz_type, path in visualization_paths.items():
					supabase_url = upload_to_supabase(path)
					if supabase_url:
						supabase_paths[viz_type] = supabase_url

				# Save URLs to database
				self.save_file_url_to_database(supabase_paths)

			return visualization_paths

		except Exception as e:
			logger.error(f"Error creating summary visualizations: {e}")
			return None

	def create_dashboard(self, df, analysis_results):
		"""
		Create an enhanced interactive dashboard with Plotly.

		Args:
			df (pandas.DataFrame): DataFrame containing email data
			analysis_results (dict): Dictionary containing analysis results

		Returns:
			str: Path to the saved dashboard
		"""
		logger.info("Creating interactive dashboard")

		try:
			# Create a more organized subplot figure
			fig = make_subplots(
				rows=4,
				cols=3,
				subplot_titles=(
				"Email Volume Over Time",
				"Top Senders",
				"Email Sentiment Distribution",
				"Cluster Sizes",
				"Topic Distribution",
				"Entity Counts",
				"Summary Length Distribution",
				"Action Items by Style",
				"Email Traffic Heatmap",
				),
				specs=[
				[{
				"type": "xy"
				}, {
				"type": "xy"
				}, {
				"type": "pie"
				}],
				[{
				"type": "bar"
				}, {
				"type": "bar"
				}, {
				"type": "bar"
				}],
				[{
				"type": "box"
				}, {
				"type": "bar"
				}, {
				"type": "xy"
				}],
				[{
				"colspan": 3
				}, None, None],  # Full width for summary
				],
				vertical_spacing=0.08,
				horizontal_spacing=0.05,
			)

			# Helper function to safely get data
			def safe_get(obj, keys, default=None):
				"""Safely get nested dictionary values"""
				try:
					result = obj
					for key in keys if isinstance(keys, list) else [keys]:
						result = result[key]
					return result
				except (KeyError, TypeError, AttributeError):
					return default

			# 1. Email Volume Over Time
			if "date" in df.columns and not df["date"].isna().all():
				try:
					# Convert date column to datetime if it's not already
					if not pd.api.types.is_datetime64_any_dtype(df["date"]):
						import warnings

						with warnings.catch_warnings():
							warnings.filterwarnings("ignore", message=".*Parsing dates.*")
							df["date"] = pd.to_datetime(df["date"], errors="coerce")

					# Drop rows with missing dates
					df_with_date = df.dropna(subset=["date"]).copy()
					if len(df_with_date) > 0:
						# Group by month
						df_with_date["month_year"] = df_with_date["date"].dt.strftime("%Y-%m")
						monthly_counts = (df_with_date["month_year"].value_counts().sort_index())

						fig.add_trace(
							go.Scatter(
							x=list(monthly_counts.index),
							y=list(monthly_counts.values),
							mode="lines+markers",
							name="Email Volume",
							line=dict(color="blue", width=3),
							marker=dict(size=8),
							),
							row=1,
							col=1,
						)
					else:
						# Add placeholder for no data
						fig.add_annotation(
							text="No date data available",
							xref="x",
							yref="y",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=1,
							col=1,
						)
				except Exception as e:
					logger.warning(f"Error creating email volume chart: {e}")

			# 2. Top Senders
			try:
				email_pattern = r"[\w\.-]+@[\w\.-]+"
				sender_counts = Counter()

				for _, row in df.iterrows():
					if pd.notna(row.get("from")):
						sender_emails = re.findall(email_pattern, str(row["from"]))
						for email in sender_emails:
							sender_counts[email] += 1

				if sender_counts:
					top_senders = sender_counts.most_common(10)
					senders, counts = zip(*top_senders) if top_senders else ([], [])

					fig.add_trace(
						go.Bar(
						x=list(counts),
						y=[sender.split("@")[0] for sender in senders],  # Show only username
						orientation="h",
						name="Top Senders",
						marker=dict(color="lightblue"),
						),
						row=1,
						col=2,
					)
				else:
					fig.add_annotation(
						text="No sender data available",
						xref="x2",
						yref="y2",
						x=0.5,
						y=0.5,
						showarrow=False,
						row=1,
						col=2,
					)
			except Exception as e:
				logger.warning(f"Error creating top senders chart: {e}")

			# 3. Sentiment Distribution
			sentiment_data = safe_get(analysis_results, ["sentiment", "distribution"])
			if sentiment_data:
				try:
					labels = list(sentiment_data.keys())
					sizes = list(sentiment_data.values())

					fig.add_trace(
						go.Pie(
						labels=labels,
						values=sizes,
						name="Sentiment",
						marker=dict(colors=["green", "red", "gray"]),
						),
						row=1,
						col=3,
					)
				except Exception as e:
					logger.warning(f"Error creating sentiment chart: {e}")
			else:
				# For pie chart subplots, we need to use paper coordinates
				fig.add_annotation(
					text="No sentiment data available",
					xref="paper",
					yref="paper",
					x=0.83,  # Approximate position for column 3
					y=0.85,  # Approximate position for row 1
					showarrow=False,
					font=dict(size=12),
				)

			# 4. Cluster Sizes
			cluster_data = safe_get(analysis_results, ["clusters", "cluster_analysis"])
			if cluster_data:
				try:
					cluster_sizes = {
						cluster_name: info.get("size", 0)
						for cluster_name, info in cluster_data.items()
					}

					if cluster_sizes:
						cluster_names = list(cluster_sizes.keys())
						cluster_values = list(cluster_sizes.values())

						fig.add_trace(
							go.Bar(
							x=cluster_names,
							y=cluster_values,
							name="Cluster Sizes",
							marker=dict(color="orange"),
							),
							row=2,
							col=1,
						)
					else:
						fig.add_annotation(
							text="No cluster size data",
							xref="x4",
							yref="y4",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=2,
							col=1,
						)
				except Exception as e:
					logger.warning(f"Error creating cluster chart: {e}")
			else:
				fig.add_annotation(
					text="No cluster data available",
					xref="x4",
					yref="y4",
					x=0.5,
					y=0.5,
					showarrow=False,
					row=2,
					col=1,
				)

			# 5. Topic Distribution
			topic_data = safe_get(analysis_results, ["topics"])
			if topic_data:
				try:
					# Get document-topic distribution
					doc_topic_matrix = topic_data.get("doc_topic_matrix")

					# Check if doc_topic_matrix is valid
					topic_names = []
					topic_counts = []

					try:
						if (doc_topic_matrix is not None and hasattr(doc_topic_matrix, "sum")
							and hasattr(doc_topic_matrix, "shape")
							and len(doc_topic_matrix.shape) == 2 and doc_topic_matrix.size > 0):
							topic_counts = doc_topic_matrix.sum(axis=0)
							# Convert to list to avoid numpy array boolean ambiguity
							topic_counts = (topic_counts.tolist() if hasattr(
								topic_counts, "tolist") else list(topic_counts))
							topic_names = [f"Topic {i + 1}" for i in range(len(topic_counts))]
					except Exception:
						# Fallback if matrix operations fail
						doc_topic_matrix = None

					if not topic_names or not topic_counts:
						# Fallback to topic keywords
						topics = topic_data.get("topics", {})
						if topics:
							topic_names = list(topics.keys())
							topic_counts = [
								len(keywords) if isinstance(keywords, list) else 1
								for keywords in topics.values()
							]

					if (topic_names and topic_counts and len(topic_names) == len(topic_counts)):
						fig.add_trace(
							go.Bar(
							x=topic_names,
							y=topic_counts,
							name="Topic Distribution",
							marker=dict(color="purple"),
							),
							row=2,
							col=2,
						)
					else:
						fig.add_annotation(
							text="No topic distribution data",
							xref="x5",
							yref="y5",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=2,
							col=2,
						)
				except Exception as e:
					logger.warning(f"Error creating topic chart: {e}")
			else:
				fig.add_annotation(
					text="No topic data available",
					xref="x5",
					yref="y5",
					x=0.5,
					y=0.5,
					showarrow=False,
					row=2,
					col=2,
				)

			# 6. Entity Counts
			entities_data = safe_get(analysis_results, ["entities"])
			if entities_data:
				try:
					entity_counts = {}
					# Combine all entity types
					for entity_type, counts in entities_data.items():
						if isinstance(counts, list):
							for item in counts[:5]:  # Take top 5 from each type
								if isinstance(item, (list, tuple)) and len(item) >= 2:
									entity, count = item[0], item[1]
									entity_counts[f"{entity} ({entity_type})"] = count

					if entity_counts:
						# Sort by count and take top 10
						sorted_entities = sorted(entity_counts.items(),
							key=lambda x: x[1],
							reverse=True)[:10]
						entity_names, entity_values = (zip(*sorted_entities) if sorted_entities else
							([], []))

						fig.add_trace(
							go.Bar(
							x=list(entity_names),
							y=list(entity_values),
							name="Entity Counts",
							marker=dict(color="cyan"),
							),
							row=2,
							col=3,
						)
					else:
						fig.add_annotation(
							text="No entity count data",
							xref="x6",
							yref="y6",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=2,
							col=3,
						)
				except Exception as e:
					logger.warning(f"Error creating entity chart: {e}")
			else:
				fig.add_annotation(
					text="No entity data available",
					xref="x6",
					yref="y6",
					x=0.5,
					y=0.5,
					showarrow=False,
					row=2,
					col=3,
				)

			# 7. Summary Length Distribution
			summaries_data = safe_get(analysis_results, ["summaries"])
			if summaries_data:
				try:
					style_lengths = {}
					for subject, styles in summaries_data.items():
						for style, summary in styles.items():
							if style not in style_lengths:
								style_lengths[style] = []
							word_count = safe_get(summary, ["metrics", "word_count"], 0)
							style_lengths[style].append(word_count)

					if style_lengths:
						for style, lengths in style_lengths.items():
							fig.add_trace(
								go.Box(y=lengths, name=style, boxmean=True),
								row=3,
								col=1,
							)
					else:
						fig.add_annotation(
							text="No summary length data",
							xref="x7",
							yref="y7",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=3,
							col=1,
						)
				except Exception as e:
					logger.warning(f"Error creating summary length chart: {e}")
			else:
				fig.add_annotation(
					text="No summary data available",
					xref="x7",
					yref="y7",
					x=0.5,
					y=0.5,
					showarrow=False,
					row=3,
					col=1,
				)

			# 8. Action Items by Style
			if summaries_data:
				try:
					style_actions = {}
					for subject, styles in summaries_data.items():
						for style, summary in styles.items():
							if style not in style_actions:
								style_actions[style] = []
							action_items = safe_get(summary, ["key_information", "action_items"],
								[])
							style_actions[style].append(len(action_items) if action_items else 0)

					if style_actions:
						fig.add_trace(
							go.Bar(
							x=list(style_actions.keys()),
							y=[sum(actions) for actions in style_actions.values()],
							name="Action Items",
							marker=dict(color="red"),
							),
							row=3,
							col=2,
						)
					else:
						fig.add_annotation(
							text="No action items data",
							xref="x8",
							yref="y8",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=3,
							col=2,
						)
				except Exception as e:
					logger.warning(f"Error creating action items chart: {e}")
			else:
				fig.add_annotation(
					text="No action items data available",
					xref="x8",
					yref="y8",
					x=0.5,
					y=0.5,
					showarrow=False,
					row=3,
					col=2,
				)

			# 9. Email Traffic Heatmap (day vs hour)
			if "date" in df.columns and not df["date"].isna().all():
				try:
					df_copy = df.copy()
					if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
						import warnings

						with warnings.catch_warnings():
							warnings.filterwarnings("ignore", message=".*Parsing dates.*")
							df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")

					df_with_date = df_copy.dropna(subset=["date"]).copy()
					if len(df_with_date) > 0:
						df_with_date["weekday"] = df_with_date["date"].dt.day_name()
						df_with_date["hour"] = df_with_date["date"].dt.hour

						# Create heatmap data
						heatmap_data = pd.crosstab(df_with_date["weekday"], df_with_date["hour"])

						fig.add_trace(
							go.Heatmap(
							z=heatmap_data.values,
							x=list(heatmap_data.columns),
							y=list(heatmap_data.index),
							colorscale="Viridis",
							name="Email Traffic",
							),
							row=3,
							col=3,
						)
					else:
						fig.add_annotation(
							text="No date data for heatmap",
							xref="x9",
							yref="y9",
							x=0.5,
							y=0.5,
							showarrow=False,
							row=3,
							col=3,
						)
				except Exception as e:
					logger.warning(f"Error creating heatmap: {e}")

			# Update layout with better styling
			fig.update_layout(
				height=1800,
				width=1400,
				title={
				"text": "📧 Enron Email Analysis Dashboard",
				"x": 0.5,
				"xanchor": "center",
				"font": {
				"size": 24,
				"color": "darkblue"
				},
				},
				showlegend=True,
				font=dict(size=12),
				plot_bgcolor="white",
				paper_bgcolor="#f8f9fa",
				margin=dict(l=50, r=50, t=100, b=50),
			)

			# Update axes labels
			fig.update_xaxes(title_font=dict(size=14))
			fig.update_yaxes(title_font=dict(size=14))

			# Save as HTML with better styling
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			output_path = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")

			# Create enhanced HTML with custom CSS
			html_string = fig.to_html(include_plotlyjs="cdn", config={"displayModeBar": True})

			# Add custom CSS for better styling
			custom_css = """
			<style>
				body {
					font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
					background-color: #f8f9fa;
					margin: 0;
					padding: 20px;
				}
				.plotly-graph-div {
					box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
					border-radius: 8px;
					background-color: white;
					margin: 10px 0;
				}
				.main-svg {
					border-radius: 8px;
				}
			</style>
			"""

			# Insert custom CSS into the HTML
			html_string = html_string.replace("<head>", f"<head>{custom_css}")

			with open(output_path, "w", encoding="utf-8") as f:
				f.write(html_string)

			logger.info(f"Saved interactive dashboard to {output_path}")

			# Upload to Supabase
			res_obj = {
				"supabase_url": None,
				"output_path": output_path,
			}

			if self.save_to_supabase:
				supabase_url = upload_to_supabase(output_path)
				if supabase_url:
					logger.info(f"Uploaded interactive dashboard to Supabase: {supabase_url}")
					res_obj["supabase_url"] = supabase_url

			return res_obj

		except Exception as e:
			logger.error(f"Error creating dashboard: {e}")
			import traceback

			logger.error(traceback.format_exc())
			return None

	def visualize_all(self, df=None, analysis_results=None):
		"""
		Create all visualizations with enhanced batch processing.

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

			logger.info(
				f"Creating all visualizations for {len(df)} emails using batch processing (batch_size={self.batch_size}, time_period={self.time_period_months} months)"
			)

			# Create all visualizations
			visualization_paths = {}
			supabase_paths = {}

			# 1. Email volume over time
			try:
				volume_path_res = self.visualize_email_volume(df)
				if volume_path_res is not None and volume_path_res.get("output_path"):
					visualization_paths["email_volume"] = volume_path_res["output_path"]
					supabase_paths["email_volume"] = volume_path_res["supabase_url"]
					logger.info("✓ Email volume visualization completed")
			except Exception as e:
				logger.error(f"Error creating email volume visualization: {e}")

			# 2. Email network
			try:
				network_path = self.visualize_email_network(df)
				if network_path:
					visualization_paths["email_network"] = network_path.get("output_path")
					supabase_paths["email_network"] = network_path.get("supabase_url")
					logger.info("✓ Email network visualization completed")
			except Exception as e:
				logger.error(f"Error creating email network visualization: {e}")

			# 3. Email threading
			try:
				threading_path = self.visualize_email_threading(df)
				if threading_path:
					visualization_paths["email_threading"] = threading_path.get("output_path")
					supabase_paths["email_threading"] = threading_path.get("supabase_url")
					logger.info("✓ Email threading visualization completed")
			except Exception as e:
				logger.error(f"Error creating email threading visualization: {e}")

			# 4. Topics
			if (analysis_results and "topics" in analysis_results and analysis_results["topics"]):
				try:
					topics_path = self.visualize_topics(analysis_results)
					if topics_path:
						output_path = topics_path.get("output_path")
						if output_path:
							visualization_paths["topics"] = output_path
						supabase_url = topics_path.get("supabase_url_static")
						if supabase_url:
							supabase_paths["topics"] = supabase_url
						logger.info("✓ Topics visualization completed")
				except Exception as e:
					logger.error(f"Error creating topics visualization: {e}")

			# 5. Clusters
			if (analysis_results and "clusters" in analysis_results
				and analysis_results["clusters"]):
				try:
					clusters_path = self.visualize_clusters(df, analysis_results)
					if clusters_path:
						output_path = clusters_path.get("output_path")
						if output_path:
							visualization_paths["clusters"] = output_path
						supabase_url = clusters_path.get("supabase_url_size")
						if supabase_url:
							supabase_paths["clusters"] = supabase_url
						logger.info("✓ Clusters visualization completed")
				except Exception as e:
					logger.error(f"Error creating clusters visualization: {e}")

			# 6. Entities
			if (analysis_results and "entities" in analysis_results
				and analysis_results["entities"]):
				try:
					entities_path = self.visualize_entities(analysis_results)
					if entities_path:
						output_path = entities_path.get("output_path")
						if output_path:
							visualization_paths["entities"] = output_path
						supabase_url = entities_path.get("supabase_url")
						if supabase_url:
							supabase_paths["entities"] = supabase_url
						logger.info("✓ Entities visualization completed")
				except Exception as e:
					logger.error(f"Error creating entities visualization: {e}")

			# 7. Sentiment
			if (analysis_results and "sentiment" in analysis_results
				and analysis_results["sentiment"]):
				try:
					sentiment_path = self.visualize_sentiment(analysis_results)
					if sentiment_path:
						output_path = sentiment_path.get("output_path")
						if output_path:
							visualization_paths["sentiment"] = output_path
						supabase_url = sentiment_path.get("supabase_url")
						if supabase_url:
							supabase_paths["sentiment"] = supabase_url
						logger.info("✓ Sentiment visualization completed")
				except Exception as e:
					logger.error(f"Error creating sentiment visualization: {e}")

			# 8. Summaries (if available)
			if (analysis_results and "summaries" in analysis_results
				and analysis_results["summaries"]):
				try:
					summaries_path = self.visualize_summaries(analysis_results)
					if summaries_path:
						for key, path in summaries_path.items():
							if "path" in key and path:
								visualization_paths[f"summaries_{key}"] = path
						logger.info("✓ Summaries visualization completed")
				except Exception as e:
					logger.error(f"Error creating summaries visualization: {e}")

			# 9. Interactive dashboard
			try:
				dashboard_result = self.create_dashboard(df, analysis_results)
				if dashboard_result:
					if isinstance(dashboard_result, dict):
						dashboard_path = dashboard_result.get("output_path")
						if dashboard_path:
							visualization_paths["dashboard"] = dashboard_path
					elif isinstance(dashboard_result, str):
						visualization_paths["dashboard"] = dashboard_result
					logger.info("✓ Interactive dashboard completed")
			except Exception as e:
				logger.error(f"Error creating dashboard: {e}")

			# Create batch information summary
			batch_info = {
				"total_emails": len(df),
				"batch_size": self.batch_size,
				"time_period_months": self.time_period_months,
				"visualizations_created": len(visualization_paths),
				"visualization_types": list(visualization_paths.keys()),
			}

			logger.info(
				f"Completed all visualizations. Created {len(visualization_paths)} visualizations:")
			for viz_type in visualization_paths.keys():
				logger.info(f"  - {viz_type}")

			return {
				"visualization_paths": visualization_paths,
				"supabase_paths": supabase_paths,
				"batch_info": batch_info,
			}

		except Exception as e:
			logger.error(f"Error in visualize_all: {e}")
			return {
				"visualization_paths": {},
				"supabase_paths": {},
				"batch_info": {
				"error": str(e)
				},
			}

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
		# save_to_postgresql(
		#     df,
		#     table_name=table_name,
		#     success_message=success_message,
		#     if_exists="replace",
		# )

	def create_time_batches(self, df):
		"""
		Create time-based batches from the DataFrame.

		Args:
			df (pandas.DataFrame): DataFrame containing email data with date column

		Returns:
			list: List of DataFrames, each representing a time batch
		"""
		if "date" not in df.columns or df["date"].isna().all():
			logger.warning("No date information available for time-based batching")
			return [df.head(self.batch_size)]

		# Convert date column to datetime
		df = df.copy()
		if not pd.api.types.is_datetime64_any_dtype(df["date"]):
			import warnings

			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", message=".*Parsing dates.*")
				df["date"] = pd.to_datetime(df["date"], errors="coerce")

		# Drop rows with invalid dates
		df = df.dropna(subset=["date"])

		if len(df) == 0:
			logger.warning("No valid dates available for batching")
			return []

		# Sort by date
		df = df.sort_values("date")

		# Create time-based groupings
		if self.time_period_months == 1:
			df["time_group"] = df["date"].dt.to_period("M")
		elif self.time_period_months == 6:
			# Create 6-month periods (2 quarters)
			quarter_periods = df["date"].dt.to_period("Q")
			df["time_group"] = (quarter_periods.astype(str).str[:4] + "_H" +
				((quarter_periods.dt.quarter - 1) // 2 + 1).astype(str))
		elif self.time_period_months == 12:
			df["time_group"] = df["date"].dt.to_period("Y")
		else:
			# Default to 6 months
			quarter_periods = df["date"].dt.to_period("Q")
			df["time_group"] = (quarter_periods.astype(str).str[:4] + "_H" +
				((quarter_periods.dt.quarter - 1) // 2 + 1).astype(str))

		# Group by time periods and create batches
		batches = []
		for time_group, group_df in df.groupby("time_group"):
			# Further split large time groups into smaller batches
			if len(group_df) > self.batch_size:
				for i in range(0, len(group_df), self.batch_size):
					batch = group_df.iloc[i:i + self.batch_size].copy()
					batch["batch_info"] = f"{time_group}_batch_{i//self.batch_size + 1}"
					batches.append(batch)
			else:
				group_df = group_df.copy()
				group_df["batch_info"] = f"{time_group}_full"
				batches.append(group_df)

		logger.info(f"Created {len(batches)} time-based batches")
		return batches

	def sample_data_for_visualization(self, df, max_items=None):
		"""
		Sample data to prevent overcrowded visualizations.

		Args:
			df (pandas.DataFrame): DataFrame to sample from
			max_items (int, optional): Maximum number of items to include

		Returns:
			pandas.DataFrame: Sampled DataFrame
		"""
		if max_items is None:
			max_items = self.batch_size

		if len(df) <= max_items:
			return df

		# Stratified sampling if date column exists
		if "date" in df.columns and not df["date"].isna().all():
			# Sample evenly across time periods
			df_sampled = df.copy()
			if not pd.api.types.is_datetime64_any_dtype(df_sampled["date"]):
				import warnings

				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", message=".*Parsing dates.*")
					df_sampled["date"] = pd.to_datetime(df_sampled["date"], errors="coerce")

			df_sampled = df_sampled.dropna(subset=["date"])

			# Group by month and sample proportionally
			df_sampled["month_year"] = df_sampled["date"].dt.strftime("%Y-%m")
			groups = df_sampled.groupby("month_year")

			samples_per_group = max(1, max_items // len(groups))
			sampled_dfs = []

			for name, group in groups:
				if len(group) <= samples_per_group:
					sampled_dfs.append(group)
				else:
					sampled_dfs.append(group.sample(n=samples_per_group, random_state=42))

			result = pd.concat(sampled_dfs, ignore_index=True)

			# If still too many, do final random sampling
			if len(result) > max_items:
				result = result.sample(n=max_items, random_state=42)

			return result.drop(columns=["month_year"])
		else:
			# Random sampling if no date column
			return df.sample(n=max_items, random_state=42)

	def visualize_email_threading(self, df):
		"""
		Visualize email threading and conversation flow.

		Args:
			df (pandas.DataFrame): DataFrame containing email data

		Returns:
			dict: Paths to the saved visualizations
		"""
		logger.info("Visualizing email threading and conversation flow")

		# Sample data for better visualization
		df_sampled = self.sample_data_for_visualization(df, max_items=self.batch_size)

		# Create threading visualization
		fig, axes = plt.subplots(2, 2, figsize=(20, 16))

		# 1. Thread length distribution
		if "subject" in df_sampled.columns:
			# Group by normalized subject (remove Re:, Fwd:, etc.)
			df_sampled["normalized_subject"] = df_sampled["subject"].apply(lambda x:
				(re.sub(r"^(Re:|Fwd:|RE:|FWD:)\s*", "",
				str(x).strip()) if pd.notna(x) else "No Subject"))

			thread_counts = df_sampled["normalized_subject"].value_counts()
			thread_lengths = thread_counts.values

			axes[0, 0].hist(
				thread_lengths,
				bins=min(20, len(thread_lengths)),
				alpha=0.7,
				edgecolor="black",
			)
			axes[0, 0].set_title("Thread Length Distribution")
			axes[0, 0].set_xlabel("Number of Emails in Thread")
			axes[0, 0].set_ylabel("Frequency")
			axes[0, 0].grid(True, alpha=0.3)

		# 2. Top conversation threads
		if "subject" in df_sampled.columns:
			top_threads = thread_counts.head(10)

			# Truncate long subject lines for better visualization
			truncated_subjects = [
				subj[:40] + "..." if len(subj) > 40 else subj for subj in top_threads.index
			]

			axes[0, 1].barh(range(len(top_threads)), top_threads.values)
			axes[0, 1].set_yticks(range(len(top_threads)))
			axes[0, 1].set_yticklabels(truncated_subjects)
			axes[0, 1].set_title("Top 10 Conversation Threads")
			axes[0, 1].set_xlabel("Number of Emails")
			axes[0, 1].grid(True, alpha=0.3)

		# 3. Response time analysis (if date column exists)
		if "date" in df_sampled.columns and not df_sampled["date"].isna().all():
			# Convert to datetime
			import warnings

			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", message=".*Parsing dates.*")
				df_sampled["date"] = pd.to_datetime(df_sampled["date"], errors="coerce")
			df_with_date = df_sampled.dropna(subset=["date"])

			if len(df_with_date) > 1:
				# Calculate time differences within threads
				response_times = []
				for subject in df_with_date["normalized_subject"].unique():
					thread_emails = df_with_date[df_with_date["normalized_subject"] ==
						subject].sort_values("date")
					if len(thread_emails) > 1:
						time_diffs = thread_emails["date"].diff().dropna()
						response_times.extend([diff.total_seconds() / 3600
							for diff in time_diffs])  # Convert to hours

				if response_times:
					# Filter out extreme values for better visualization
					response_times = [rt for rt in response_times if rt <= 168]  # Max 1 week

					axes[1, 0].hist(response_times, bins=20, alpha=0.7, edgecolor="black")
					axes[1, 0].set_title("Response Time Distribution")
					axes[1, 0].set_xlabel("Response Time (hours)")
					axes[1, 0].set_ylabel("Frequency")
					axes[1, 0].grid(True, alpha=0.3)

		# 4. Thread network graph
		if ("subject" in df_sampled.columns and "from" in df_sampled.columns
			and "to" in df_sampled.columns):
			# Create a simplified thread network
			G = nx.Graph()
			email_pattern = r"[\w\.-]+@[\w\.-]+"

			# Select top 5 threads for network visualization
			top_5_threads = thread_counts.head(5).index
			thread_data = df_sampled[df_sampled["normalized_subject"].isin(top_5_threads)]

			for _, row in thread_data.iterrows():
				if pd.notna(row["from"]) and pd.notna(row["to"]):
					sender_emails = re.findall(email_pattern, str(row["from"]))
					recipient_emails = re.findall(email_pattern, str(row["to"]))

					for sender in sender_emails[:1]:  # Take first sender only
						for recipient in recipient_emails[:2]:  # Take first 2 recipients
							if sender != recipient:
								thread_subject = (row["normalized_subject"][:20] + "..." if len(
									row["normalized_subject"]) > 20 else row["normalized_subject"])

								if not G.has_edge(sender, recipient):
									G.add_edge(sender, recipient, threads=set(), weight=0)

								G[sender][recipient]["threads"].add(thread_subject)
								G[sender][recipient]["weight"] += 1

			if G.nodes():
				pos = nx.spring_layout(G, k=1, iterations=50)

				# Node sizes based on degree
				degrees = dict(G.degree())  # type: ignore
				node_sizes = [degrees[node] * 100 + 50 for node in G.nodes()]

				# Edge widths based on weight
				edge_widths = [G[u][v]["weight"] for u, v in G.edges()]

				# Different colors for different users - use simple color list
				color_list = [
					"red",
					"blue",
					"green",
					"orange",
					"purple",
					"brown",
					"pink",
					"gray",
					"olive",
					"cyan",
				]
				colors = [color_list[i % len(color_list)] for i in range(len(G.nodes()))]

				nx.draw_networkx_nodes(
					G,
					pos,
					node_size=node_sizes,  # type: ignore
					node_color=colors,  # type: ignore
					alpha=0.8,
					ax=axes[1, 1],
				)
				nx.draw_networkx_edges(
					G,
					pos,
					width=edge_widths,  # type: ignore
					alpha=0.6,
					edge_color="gray",
					ax=axes[1, 1],
				)

				# Add labels (email usernames only)
				labels = {node: node.split("@")[0] for node in G.nodes()}
				nx.draw_networkx_labels(
					G,
					pos,
					labels=labels,
					font_size=8,
					font_weight="bold",
					ax=axes[1, 1],
				)

				axes[1, 1].set_title("Thread Network (Top 5 Conversations)")
				axes[1, 1].axis("off")

		plt.tight_layout()

		# Save the figure
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join(self.output_dir, f"email_threading_{timestamp}.png")
		plt.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close()

		logger.info(f"Saved email threading visualization to {output_path}")

		res_obj = {
			"supabase_url": None,
			"output_path": output_path,
		}

		if self.save_to_supabase:
			supabase_url = upload_to_supabase(output_path)
			if supabase_url:
				logger.info(f"Uploaded email threading visualization to Supabase: {supabase_url}")
				res_obj["supabase_url"] = supabase_url

		return res_obj
