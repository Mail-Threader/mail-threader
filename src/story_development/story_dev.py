import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor as FutureExecutor
from datetime import timedelta
from typing import Optional

import networkx as nx
import numpy as np
import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

from utils import custom_stop_words, load_processed_df, save_error_log


class StoryDevelopment:

	def __init__(
		self,
		processed_data_dir="./processed_data/",
		analysis_results_dir="./analysis_results/",
		output_dir="./stories/",
	):
		"""
		Initialize the StoryDevelopment class.

		Args:
			processed_data_dir (str): Directory containing processed email data
			analysis_results_dir (str): Directory containing analysis results
			output_dir (str): Directory to store generated stories
		"""

		self.processed_data_dir = processed_data_dir
		self.analysis_results_dir = analysis_results_dir
		self.output_dir = output_dir

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

	def generate_stories_from_threads(
			self,
			df: pd.DataFrame,
			limit: Optional[int] = None,
			style="factual",  # "factual" or "creative"
	):
		"""
		Generate stories from email threads.

		Args:
			df (pd.DataFrame): DataFrame containing email data with thread_id.
			limit (Optional[int]): Maximum number of stories to generate. If None, generate all.
			style (str): Style of the story to generate, either "factual" or "creative".

		Returns:
			pd.DataFrame: DataFrame with generated stories.
		"""

		if "thread_id" not in df.columns:
			save_error_log("DataFrame does not contain 'thread_id' column.")
			logger.error("DataFrame does not contain 'thread_id' column.")
			return []

		logger.info("Generating stories from email threads...")

		stories = []
		grouped = df.groupby("thread_id")

		i = 0

		for thread_id, thread_df in tqdm(
			grouped,
			total=len(grouped),
			desc="Generating stories from threads",
		):
			if pd.isna(thread_id):
				continue

			prompt = ((
				"Based on the following email thread, write a concise and factual narrative story. "
				"The story should strictly adhere to the information provided in the emails,"
				"capturing only the key events, decisions and emotions showed in input data."
				"Avoid introducing any external details or creative interpretations not explicitly mentioned."
				"Focus on summarizing the thread's progression objectively and include relevant details from all emails.\n\n"
				"Understand the entities, names and their nicknames, relationships, and events in the emails"
				"In response, only provide the story without any additional commentary or explanations.\n\n"
			) if style == "factual" else (
				"Based on the following email thread, write a compelling and creative narrative story. "
				"The story should capture the key events, decisions, and emotions of the people involved. "
				"Feel free to add creative elements while staying true to the essence of the emails.\n\n"
				# Based on the following email thread, write a compelling factual narrative story. The story should capture the key events, decisions, and emotions of the people involved. \n\n
			))
			for _, row in thread_df.iterrows():
				prompt += f"From: {row['from']}\nTo: {row['to']}\nSubject: {row['subject']}\nDate: {row['date']}\n\n{row['body']}\n\n---\n\n"
			try:

				result = ollama.generate(
					model="llama3.2:3b",
					prompt=prompt,
					options={
					"temperature": (0.2 if style == "factual" else
					0.5),  # Lower temperature for less creativity, more factual output
					"top_p": 0.5,  # Lower top_p to focus on highly probable tokens
					},
				)

				response = result.get("response", "")
				if not response:
					logger.warning(f"No response generated for thread {thread_id}. Skipping...")
					continue

				stories.append({
					"thread_id": thread_id,
					"story": response.strip(),
				})

				i += 1
				if limit is not None and i >= limit:
					logger.info(f"Reached limit of {limit} stories. Stopping generation.")
					break
			except Exception as e:
				logger.error(f"Error generating story for thread {thread_id}: {e}")
				save_error_log(f"Error generating story for thread {thread_id}: {e}")
				continue

		return stories

	def generate_non_threaded_stories(
			self,
			df: pd.DataFrame,
			limit: Optional[int] = None,
			style="factual",  # "factual" or "creative"
	):
		"""
		Generate stories from email data.

		Args:
						df (pd.DataFrame): DataFrame containing email data with thread_id.
						limit (Optional[int]): Maximum number of stories to generate. If None, generate all.
						style (str): Style of the story to generate, either "factual" or "creative".

		Returns:
						pd.DataFrame: DataFrame with generated stories.
		"""

		logger.info("Generating stories from dataframe which don't have thread_id...")

		stories = []
		df_no_thread = df[df["thread_id"].isna()]

		i = 0

		for idx, row in tqdm(
			df_no_thread.iterrows(),
			total=len(df_no_thread),
			desc="Generating stories from non-threaded emails",
		):
			prompt = (("Based on the following email, write a concise and factual narrative story. "
				"The story should strictly adhere to the information provided in the email,"
				"capturing only the key events, decisions and emotions showed in input data."
				"Avoid introducing any external details or creative interpretations not explicitly mentioned."
				"Focus on summarizing the email's content objectively and include relevant details.\n\n"
				"Understand the entities, names and their nicknames, relationships, and events in the email"
				"In response, only provide the story without any additional commentary or explanations.\n\n"
						) if style == "factual" else
				("Based on the following email, write a compelling and creative narrative story. "
				"The story should capture the key events, decisions, and emotions of the people involved. "
				"Feel free to add creative elements while staying true to the essence of the email.\n\n"
					))
			prompt += f"From: {row['from']}\nTo: {row['to']}\nSubject: {row['subject']}\nDate: {row['date']}\n\n{row['body']}\n\n---\n\n"

			try:
				result = ollama.generate(
					model="llama3.2:3b",
					prompt=prompt,
					options={
					"temperature": (0.2 if style == "factual" else
					0.5),  # Lower temperature for less creativity, more factual output
					"top_p": 0.5,  # Lower top_p to focus on highly probable tokens
					},
				)

				response = result.get("response", "")
				if not response:
					logger.warning(
						f"No response generated for email with subject '{row['subject']}'. Skipping..."
					)
					continue

				stories.append({
					"message_id": row["message_id"],
					"story": response.strip(),
				})

				i += 1
				if limit is not None and i >= limit:
					logger.info(f"Reached limit of {limit} stories. Stopping generation.")
					break
			except Exception as e:
				logger.error(
					f"Error generating story for email with subject '{row['subject']}': {e}")
				save_error_log(
					f"Error generating story for email with subject '{row['subject']}': {e}")

	@staticmethod
	def identify_key_actors(df: pd.DataFrame):
		"""
		Identify key actors in the email data.

		Args:
						df (pd.DataFrame): DataFrame containing email data.

		Returns:
						list: List of key actors identified in the emails.
		"""
		logger.info("Identifying key actors...")

		top_n = min(20, 10) if len(df) < 100 else max(20, round(len(df) / 10))

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
			sender_emails = (re.findall(email_pattern, row["from"])
				if row["from"] is not None else [])
			recipient_emails = (re.findall(email_pattern, row["to"])
				if row["to"] is not None else [])

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

			return {"top_actors": top_actors}

		except Exception as e:
			logger.error(f"Error calculating network metrics: {e}")
			return {
				"top_actors": {
				actor: {
				"sent": count,
				"received": recipient_counts.get(actor, 0)
				}
				for actor, count in sender_counts.most_common(top_n)
				},
			}

	def detect_significant_events(self, df: pd.DataFrame):
		"""
		Detect significant events in the email data.

		Args:
						df (pd.DataFrame): DataFrame containing email data.

		Returns:
						list: List of significant events detected in the emails.
		"""
		logger.info("Detecting significant events...")

		# Count emails per day
		logger.info("Counting emails per day...")
		daily_counts = df.groupby(df["date"].dt.date).size()

		# Calculate rolling statistics
		logger.info("Calculating rolling statistics...")
		rolling_mean = daily_counts.rolling(window=7, min_periods=1).mean()
		rolling_std = daily_counts.rolling(window=7, min_periods=1).std()

		# Identify spikes
		logger.info("Identifying email volume spikes...")
		threshold = 2
		spikes = daily_counts[daily_counts > (rolling_mean + threshold * rolling_std)]

		# Extract events
		events = []
		logger.info("Analyzing detected events...")
		for date, count in tqdm(spikes.items(), desc="Processing events"):
			# Convert date to datetime for filtering
			date = str(date)
			event_date = pd.to_datetime(date)

			# Get emails from the spike day
			_date = pd.to_datetime(df["date"].dt.date)
			event_emails = df[(_date >= event_date) & (_date < event_date + timedelta(days=1))]

			# Extract common words from these emails
			if len(event_emails) > 0:
				# Combine all text
				all_text = " ".join(event_emails["body"].fillna(""))

				# Tokenize and count words
				words = re.findall(r"\b\w+\b", all_text.lower())
				words = [word for word in words if word not in custom_stop_words and len(word) > 2]
				common_words = Counter(words).most_common(10)

				# Get sample subjects
				sample_subjects = event_emails["subject"].head(5).tolist()

				events.append({
					"date":
					date,
					"email_count":
					count,
					"normal_level":
					rolling_mean[date],
					"std_dev":
					rolling_std[date],
					"deviation": ((count - rolling_mean[date]) /
					rolling_std[date] if rolling_std[date] > 0 else 0),
					"common_words":
					common_words,
					"sample_subjects":
					sample_subjects,
					"email_ids":
					event_emails.index.tolist(),
				})

		# Sort events by deviation
		events.sort(key=lambda x: x["deviation"], reverse=True)

		return events

	@staticmethod
	def track_topics_over_time(analysis_results: pd.DataFrame):
		"""
		Track how topics evolve over time.

		Args:
						analysis_results (pd.DataFrame): DataFrame containing analysis results

		Returns:
						dict: Dictionary containing topic evolution data
		"""
		logger.info("Tracking topics over time")

		# check for "topic_keywords" column in analysis_results
		if "topic_label" not in analysis_results.columns:
			logger.warning("No topic_label column found in analysis_results")
			return {"time_periods": [], "topic_counts": {}, "topic_keywords": {}}

		# Create time-based features
		analysis_results["date"] = pd.to_datetime(analysis_results["date"], errors="coerce")
		analysis_results["year"] = analysis_results["date"].dt.year
		analysis_results["month"] = analysis_results["date"].dt.month
		analysis_results["week"] = analysis_results["date"].dt.isocalendar().week

		# Create a date string for grouping by month
		analysis_results["month_year"] = analysis_results["date"].dt.strftime("%Y-%m")

		# Group by month and topic
		topic_time_counts = (analysis_results.groupby(["month_year",
			"topic_label"]).size().unstack(fill_value=0))

		topics = analysis_results["topic_label"].unique()

		if topics is not None:
			topics = [str(topic) for topic in topics if pd.notna(topic)]
			topics = list(set(topics))
			topics.sort()

		# Create a dictionary with topic evolution data
		topic_evolution = {
			"time_periods": list(topic_time_counts.index),
			"topic_counts": topic_time_counts.to_dict(),
			"topic_keywords": topics,
		}

		logger.info(
			f"Topic evolution analysis complete: {len(topic_evolution['time_periods'])} time periods, {len(topics)} topics"
		)
		return topic_evolution

	@staticmethod
	def _generate_actor_summary(
		actor,
		metrics,
		common_words,
		busiest_day,
		busiest_day_count,
		avg_response_time,
	):
		"""Generate a summary for a key actor story."""
		# Create an engaging hook
		name = actor.split("@")[0].replace(".", " ").title()
		summary = f"Unveiling the Digital Footprint: The Enron Emails of {name}\n\n"

		# Add role and influence context
		influence_score = (metrics.get("degree_centrality", 0) +
			metrics.get("betweenness_centrality", 0) + metrics.get("pagerank", 0)) / 3
		influence_level = ("highly influential" if influence_score > 0.1 else
			"moderately influential" if influence_score > 0.05 else "influential")
		summary += f"In the intricate web of Enron's corporate communications, {name} emerges as a {influence_level} figure, "
		summary += f"having sent {metrics['sent']} emails and received {metrics['received']} emails. "

		# Add communication patterns with context
		summary += f"\n\nCommunication Patterns:\n"
		summary += f"• Peak Activity: {name} is most active on {busiest_day}s, sending {busiest_day_count} emails. "
		if avg_response_time:
			response_context = ("remarkably quick"
				if avg_response_time < 4 else "moderate" if avg_response_time < 8 else "deliberate")
			summary += f"This suggests a {response_context} response pattern, with an average response time of {avg_response_time:.1f} hours. "

		# Add topic analysis with deeper context
		summary += f"\n\nKey Focus Areas:\n"
		top_topics = [word for word, _ in common_words[:5]]
		topic_counts = [count for _, count in common_words[:5]]
		topic_analysis = []
		for topic, count in zip(top_topics, topic_counts):
			if topic.lower() in ["energy", "power", "gas"]:
				topic_analysis.append(f"energy sector operations ({count} mentions)")
			elif topic.lower() in ["state", "regulatory", "policy"]:
				topic_analysis.append(f"regulatory affairs ({count} mentions)")
			elif topic.lower() in ["market", "trading", "price"]:
				topic_analysis.append(f"market activities ({count} mentions)")
			else:
				topic_analysis.append(f"{topic} ({count} mentions)")

		summary += f"• Primary Focus: {', '.join(topic_analysis[:-1])}, and {topic_analysis[-1]}. "

		# Add network analysis
		summary += f"\n\nNetwork Impact:\n"
		summary += f"• Influence Score: {influence_score:.3f} (combining centrality measures)\n"
		summary += f"• Communication Reach: {metrics.get('degree_centrality', 0):.3f} (direct connections)\n"
		summary += f"• Information Flow: {metrics.get('betweenness_centrality', 0):.3f} (brokerage role)\n"
		summary += f"• Overall Importance: {metrics.get('pagerank', 0):.3f} (network-wide significance)"

		# Add temporal analysis
		summary += f"\n\nTemporal Patterns:\n"
		if busiest_day_count > 10:
			summary += f"• High Activity: {name} shows intense engagement on {busiest_day}s, suggesting this day may be crucial for weekly operations or reporting.\n"
		if avg_response_time and avg_response_time < 4:
			summary += f"• Quick Response: The rapid response time indicates a key operational role or high-priority communications.\n"

		# Add concluding insights
		summary += f"\n\nKey Insights:\n"
		if influence_score > 0.1:
			summary += f"• {name} plays a central role in Enron's communication network, acting as a key information hub.\n"
		if any("energy" in topic.lower() for topic in top_topics):
			summary += f"• Strong focus on energy sector operations suggests involvement in core business activities.\n"
		if avg_response_time and avg_response_time < 4:
			summary += f"• Quick response times indicate a hands-on role in critical communications.\n"

		return summary

	@staticmethod
	def _generate_event_summary(
		date,
		email_count,
		deviation,
		common_words,
		event_metrics,
	):
		"""Generate a summary for a significant event story."""
		# Create an engaging hook
		summary = f"Unusual Activity Detected: Email Surge on {date}\n\n"

		# Add event significance
		significance_level = ("extremely significant"
			if deviation > 3 else "highly significant" if deviation > 2 else "significant")
		summary += f"A {significance_level} spike in email activity was detected, with {email_count} emails exchanged "
		summary += f"({deviation:.1f} standard deviations above normal). "

		# Add participant analysis
		summary += f"\n\nParticipant Analysis:\n"
		summary += f"• Scale: {event_metrics['participant_count']} individuals were involved in this communication surge\n"
		summary += f"• Engagement: Average email length of {event_metrics['avg_email_length']:.0f} characters suggests "
		summary += f"{'detailed discussions' if event_metrics['avg_email_length'] > 500 else 'brief exchanges'}\n"
		summary += (f"• Interaction: Reply rate of {event_metrics['reply_rate']:.1%} indicates ")
		summary += f"{'highly interactive' if event_metrics['reply_rate'] > 0.5 else 'moderate'} communication patterns"

		# Add topic analysis
		summary += f"\n\nKey Topics:\n"
		top_words = [word for word, _ in common_words[:5]]
		word_counts = [count for _, count in common_words[:5]]
		topic_analysis = []
		for word, count in zip(top_words, word_counts):
			if word.lower() in ["urgent", "emergency", "critical"]:
				topic_analysis.append(f"urgent matters ({count} mentions)")
			elif word.lower() in ["meeting", "conference", "call"]:
				topic_analysis.append(f"coordination activities ({count} mentions)")
			elif word.lower() in ["report", "update", "status"]:
				topic_analysis.append(f"status updates ({count} mentions)")
			else:
				topic_analysis.append(f"{word} ({count} mentions)")

		summary += f"• Primary Focus: {', '.join(topic_analysis[:-1])}, and {topic_analysis[-1]}\n"

		# Add temporal context
		summary += f"\n\nTemporal Context:\n"
		if deviation > 3:
			summary += f"• Exceptional Activity: This spike represents one of the most significant communication events in the dataset\n"
		if event_metrics["reply_rate"] > 0.7:
			summary += f"• Rapid Response: The high reply rate suggests urgent or time-sensitive matters were being discussed\n"
		if event_metrics["avg_email_length"] > 1000:
			summary += f"• Detailed Communication: The lengthy emails indicate complex or important discussions\n"

		# Add potential implications
		summary += f"\n\nPotential Implications:\n"
		if any(word.lower() in ["urgent", "emergency", "critical"] for word in top_words):
			summary += f"• This event may represent a critical business situation requiring immediate attention\n"
		if event_metrics["participant_count"] > 10:
			summary += f"• The large number of participants suggests a company-wide or department-wide communication event\n"
		if event_metrics["reply_rate"] > 0.5:
			summary += f"• The high level of interaction indicates active problem-solving or decision-making\n"

		return summary

	@staticmethod
	def _analyze_topic_trend(topic_counts, topic_num):
		"""Analyze the trend of a topic over time."""
		if not topic_counts:
			return {"trend": "unknown", "peak_period": None, "peak_count": 0}

		# Extract counts for this topic
		topic_data = {period: counts.get(topic_num, 0) for period, counts in topic_counts.items()}

		if not topic_data:
			return {"trend": "unknown", "peak_period": None, "peak_count": 0}

		# Find peak period
		peak_period = max(topic_data.items(), key=lambda x: x[1])

		# Determine trend
		values = list(topic_data.values())
		if len(values) < 2:
			trend = "stable"
		else:
			slope = np.polyfit(range(len(values)), values, 1)[0]
			if slope > 0.1:
				trend = "increasing"
			elif slope < -0.1:
				trend = "decreasing"
			else:
				trend = "stable"

		return {
			"trend": trend,
			"peak_period": peak_period[0],
			"peak_count": peak_period[1],
		}

	@staticmethod
	def _generate_topic_summary(
		topic_id,
		keywords,
		topic_trend,
	):
		"""Generate a summary for a topic evolution story."""
		# Create an engaging hook
		summary = f"Topic Evolution: {topic_id}\n\n"

		# Add topic overview
		summary += (f"Analysis of communication patterns reveals the evolution of {topic_id}, ")
		summary += f"characterized by keywords such as {', '.join(keywords[:3])}. "

		# Add trend analysis
		summary += f"\n\nTrend Analysis:\n"
		trend_character = {
			"increasing": "growing",
			"decreasing": "declining",
			"stable": "consistent",
		}.get(topic_trend["trend"], "variable")

		summary += f"• Overall Trend: {trend_character} interest in this topic\n"
		if topic_trend["peak_period"]:
			summary += f"• Peak Activity: {topic_trend['peak_count']} mentions during {topic_trend['peak_period']}\n"

		# Add keyword analysis
		summary += f"\n\nKeyword Analysis:\n"
		keyword_categories = {
			"energy": "energy sector",
			"power": "power operations",
			"gas": "gas operations",
			"market": "market activities",
			"trading": "trading operations",
			"price": "pricing",
			"state": "regulatory affairs",
			"regulatory": "regulatory matters",
			"policy": "policy issues",
		}

		categorized_keywords = []
		for keyword in keywords[:5]:
			category = next((v for k, v in keyword_categories.items() if k in keyword.lower()),
				None)
			if category:
				categorized_keywords.append(f"{category} ({keyword})")
			else:
				categorized_keywords.append(keyword)

		summary += f"• Primary Focus: {', '.join(categorized_keywords[:-1])}, and {categorized_keywords[-1]}\n"

		# Add temporal patterns
		summary += f"\n\nTemporal Patterns:\n"
		if topic_trend["trend"] == "increasing":
			summary += (f"• Growing Interest: The topic shows increasing relevance over time\n")
		elif topic_trend["trend"] == "decreasing":
			summary += f"• Declining Focus: The topic shows decreasing prominence\n"
		else:
			summary += f"• Stable Presence: The topic maintains consistent attention\n"

		# Add potential implications
		summary += f"\n\nPotential Implications:\n"
		if topic_trend["trend"] == "increasing":
			summary += (f"• This topic may represent an emerging area of focus or concern\n")
		if topic_trend["peak_count"] > 50:
			summary += (f"• The high peak activity suggests significant business impact\n")
		if any(kw.lower() in ["urgent", "critical", "emergency"] for kw in keywords):
			summary += (f"• The presence of urgent keywords indicates time-sensitive matters\n")

		return summary

	def generate_summaries(
		self,
		analysis_result: pd.DataFrame,
		key_actors: dict,
		significant_events: list,
		topic_evolution: dict,
	):
		"""
		Generate summaries based on key actors and significant events.

		Args:
						analysis_result (pd.DataFrame): DataFrame containing analysis results.
						key_actors (dict): Dictionary of key actors identified in the emails.
						significant_events (list): List of significant events detected in the emails.
						topic_evolution (dict): Dictionary containing topic evolution data.

		Returns:
						list: List of summaries generated from the email data.
		"""
		logger.info("Generating summaries...")

		summaries = []

		# 1. Stories based on key actors
		if key_actors and "top_actors" in key_actors:
			for actor, metrics in list(key_actors["top_actors"].items())[:5]:
				# Get emails sent by this actor
				actor_emails = analysis_result[analysis_result["from"].str.contains(actor,
					na=False)]

				if len(actor_emails) > 0:
					# Get related emails (emails in threads where actor participated)
					related_emails = []
					for _, email in actor_emails.iterrows():
						thread_emails = analysis_result[analysis_result["subject"].str.contains(
							email["subject"], na=False)]
						for _, thread_email in thread_emails.iterrows():
							related_emails.append({
								"subject":
								thread_email["subject"],
								"date":
								thread_email["date"],
								"from":
								thread_email["from"],
								"to":
								thread_email["to"],
								"body_preview": (thread_email["body"][:200] +
								"..." if len(thread_email["body"]) > 200 else thread_email["body"]),
							})

					# Use clean_body column if available, otherwise use body
					text_column = ("clean_body" if "clean_body" in actor_emails.columns else "body")
					all_text = " ".join(actor_emails[text_column].fillna(""))
					words = re.findall(r"\b\w+\b", all_text.lower())
					words = [
						word for word in words if word not in custom_stop_words and len(word) > 2
					]
					common_words = Counter(words).most_common(20)
					sample_subjects = actor_emails["subject"].head(5).tolist()

					# Calculate communication patterns
					daily_patterns = actor_emails.groupby(actor_emails["date"].dt.day_name()).size()
					busiest_day = daily_patterns.idxmax()
					busiest_day_count = daily_patterns.max()

					# Calculate average response time
					response_times = []
					for _, email in actor_emails.iterrows():
						if pd.notna(email["date"]):
							replies = analysis_result[(
								analysis_result["subject"].str.contains(email["subject"], na=False))
								& (analysis_result["date"] > email["date"])]
							if not replies.empty:
								response_time = (replies["date"].min() -
									email["date"]).total_seconds() / 3600
								response_times.append(response_time)

					avg_response_time = (np.mean(response_times) if response_times else None)

					story = {
						"title":
						f"The Story of {actor}",
						"type":
						"key_actor",
						"actor":
						actor,
						"metrics":
						metrics,
						"common_topics":
						common_words,
						"sample_subjects":
						sample_subjects,
						"communication_patterns": {
						"busiest_day":
						busiest_day,
						"busiest_day_count":
						int(busiest_day_count),
						"avg_response_time":
						(f"{avg_response_time:.1f} hours" if avg_response_time else "N/A"),
						},
						"related_emails":
						related_emails,
						"summary":
						self._generate_actor_summary(
						actor,
						metrics,
						common_words,
						busiest_day,
						busiest_day_count,
						avg_response_time,
						),
					}
					summaries.append(story)

		# 2. Stories based on significant events
		for event in significant_events[:5]:
			event_date = event["date"]
			event_emails = analysis_result[
				(pd.to_datetime(analysis_result["date"].dt.date) >= pd.to_datetime(event_date))
				& (pd.to_datetime(analysis_result["date"].dt.date) < pd.to_datetime(event_date) +
				timedelta(days=1))]

			# Get all related emails (including replies and forwards)
			related_emails = []
			for _, email in event_emails.iterrows():
				# Get emails in the same thread
				thread_emails = analysis_result[analysis_result["subject"].str.contains(
					email["subject"], na=False)]
				for _, thread_email in thread_emails.iterrows():
					related_emails.append({
						"subject":
						thread_email["subject"],
						"date":
						thread_email["date"],
						"from":
						thread_email["from"],
						"to":
						thread_email["to"],
						"body_preview": (thread_email["body"][:200] +
						"..." if len(thread_email["body"]) > 200 else thread_email["body"]),
					})

			# Analyze event participants
			participants = set()
			for _, email in event_emails.iterrows():
				if pd.notna(email["from"]):
					participants.update(re.findall(r"[\w\.-]+@[\w\.-]+", email["from"]))
				if pd.notna(email["to"]):
					participants.update(re.findall(r"[\w\.-]+@[\w\.-]+", email["to"]))

			# Calculate event metrics
			event_metrics = {
				"participant_count":
				len(participants),
				"avg_email_length":
				event_emails["body"].str.len().mean(),
				"reply_rate":
				len(event_emails[event_emails["subject"].str.contains("Re:", na=False)]) /
				len(event_emails),
			}

			story = {
				"title":
				f"Significant Event on {event_date}",
				"type":
				"significant_event",
				"date":
				event_date,
				"email_count":
				event["email_count"],
				"common_words":
				event["common_words"],
				"sample_subjects":
				event["sample_subjects"],
				"event_metrics":
				event_metrics,
				"participants":
				list(participants),
				"related_emails":
				related_emails,
				"summary":
				self._generate_event_summary(
				event_date,
				event["email_count"],
				event["deviation"],
				event["common_words"],
				event_metrics,
				),
			}
			summaries.append(story)

		# 3. Stories based on topic evolution
		if topic_evolution and "topic_keywords" in topic_evolution:
			for topic_id, keywords in list(topic_evolution["topic_keywords"].items())[:5]:
				# Get emails related to this topic
				topic_emails = []
				for keyword in keywords[:5]:  # Use top 5 keywords
					keyword_emails = analysis_result[analysis_result["body"].str.contains(keyword,
						case=False,
						na=False)]
					for _, email in keyword_emails.iterrows():
						topic_emails.append({
							"subject":
							email["subject"],
							"date":
							email["date"],
							"from":
							email["from"],
							"to":
							email["to"],
							"body_preview": (email["body"][:200] +
							"..." if len(email["body"]) > 200 else email["body"]),
							"matching_keyword":
							keyword,
						})

				# Extract topic number
				topic_num = int(topic_id.split()[-1])

				# Get topic evolution data
				topic_counts = topic_evolution.get("topic_counts", {})
				topic_trend = self._analyze_topic_trend(topic_counts, topic_num)

				story = {
					"title": f"The Evolution of {topic_id}",
					"type": "topic_evolution",
					"topic_id": topic_id,
					"keywords": keywords,
					"topic_metrics": {
					"trend": topic_trend["trend"],
					"peak_period": topic_trend["peak_period"],
					"peak_count": topic_trend["peak_count"],
					},
					"related_emails": topic_emails,
					"summary": self._generate_topic_summary(
					topic_id,
					keywords,
					topic_trend,
					),
				}
				summaries.append(story)

		return summaries

	def develop_story(
		self,
		processed_data: Optional[pd.DataFrame],
		analysis_results: Optional[pd.DataFrame],
		limit: Optional[int] = None,
	):
		"""
		Generate a story based on processed data and analysis results.

		Args:
			processed_data (Optional[pd.DataFrame]): DataFrame containing processed email data
			analysis_results (Optional[pd.DataFrame]): DataFrame containing analysis results.
			limit (Optional[int]): Maximum number of stories to generate. If None, generate all.
		"""

		try:
			logger.info("Starting story generation...")

			if (processed_data is None or processed_data.empty) or (analysis_results is None
				or analysis_results.empty):
				logger.warning(
					"No processed data or analysis results provided. Extracting from directories.")

				if processed_data is None or processed_data.empty:
					logger.warning(
						"No processed data provided. Attempting to load from processed data directory."
					)
					processed_data = load_processed_df(
						search_dir=self.processed_data_dir,
						search_file_name="processed_data_",
					)

				if analysis_results is None:
					logger.warning(
						"No analysis results provided. Attempting to load from analysis directory.")
					analysis_results = load_processed_df(
						search_dir=self.analysis_results_dir,
						search_file_name="analysis_results_",
					)

			if (processed_data is None or processed_data.empty or analysis_results is None
				or analysis_results.empty):
				logger.error("Failed to load necessary data for story generation.")
				return None

			logger.info(f"Developing stories from {len(analysis_results)} analysis results.")

			num_threads = max(os.cpu_count() or 1, 4)
			with FutureExecutor(max_workers=num_threads) as executor:
				futures = {
					"stories_from_threads":
					executor.submit(
					self.generate_stories_from_threads,
					analysis_results,
					limit=limit,
					style="creative",
					),
					"stories_from_non_threads":
					executor.submit(
					self.generate_non_threaded_stories,
					analysis_results,
					limit=limit,
					style="factual",
					),
					"identify_key_actors":
					executor.submit(self.identify_key_actors, processed_data),
					"detect_significant_events":
					executor.submit(self.detect_significant_events, processed_data),
					"track_topics_over_time":
					executor.submit(self.track_topics_over_time, analysis_results),
				}

			results = {}

			for task_name, future in tqdm(futures.items(), desc="Processing All Story Tasks"):
				try:
					result = future.result(timeout=600)  # Set a timeout for each task
					results[task_name] = result
				except Exception as e:
					save_error_log(f"Error in task {task_name}: {e}")
					logger.error(f"Error in task {task_name}: {e}")

			threaded_stories = results.get("stories_from_threads", [])
			non_threaded_stories = results.get("stories_from_non_threads", [])
			key_actors = results.get("identify_key_actors", {})
			significant_events = results.get("detect_significant_events", [])
			topic_evolution = results.get("track_topics_over_time", {})

			logger.info(
				f"Generated {len(threaded_stories)} threaded stories and {len(non_threaded_stories)} non-threaded stories."
			)
			logger.info(f"Identified {len(key_actors.get('top_actors', []))} key actors.")
			logger.info(f"Detected {len(significant_events)} significant events.")
			logger.info(
				f"Tracked topic evolution with {len(topic_evolution.get('topic_keywords', []))} topics."
			)

			# Generate summaries based on the results
			summaries = self.generate_summaries(
				analysis_result=analysis_results,
				key_actors=key_actors,
				significant_events=significant_events,
				topic_evolution=topic_evolution,
			)
			logger.info(f"Generated {len(summaries)} summaries.")

		except Exception as e:
			logger.error(f"Error generating story: {e}")
