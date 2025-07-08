import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor as FutureExecutor
from datetime import timedelta, datetime
from typing import Optional
import networkx as nx
import numpy as np
import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm
from utils import custom_stop_words, load_processed_df, save_error_log, sort_emails_by_date
import json
import uuid
from database.database_manager import DatabaseManager


class StoryDevelopment:
	"""
	Author - Siddhant Dalvi
	Matriculation Number - 319413
	"""

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
			total=len(grouped) if limit is None else limit,
			desc="Generating stories from threads",
		):
			if pd.isna(thread_id):
				continue

			# Construct the prompt for the LLM, instructing it to return JSON with title and story
			if style == "factual":
				prompt = (
					"Based on the following email thread, write a concise and factual narrative story. "
					"The story should strictly adhere to the information provided in the emails, "
					"capturing only the key events, decisions, and emotions showed in input data. "
					"Avoid introducing any external details or creative interpretations not explicitly mentioned. "
					"Focus on summarizing the thread's progression objectively and include relevant details from all emails. "
					"Understand the entities, names and their nicknames, relationships, and events in the emails. "
					"In response, provide a JSON object with two keys: 'title' for a concise title "
					"summarizing the story, and 'story' for the narrative itself. "
					"Do not include any additional commentary or explanations outside the JSON.\n\n"
				)
			else:  # style == "creative"
				prompt = (
					"Based on the following email thread, write a compelling and creative narrative story. "
					"The story should capture the key events, decisions, and emotions of the people involved. "
					"Feel free to add creative elements while staying true to the essence of the emails. "
					"In response, provide a JSON object with two keys: 'title' for a creative and engaging title "
					"summarizing the story, and 'story' for the narrative itself. "
					"Do not include any additional commentary or explanations outside the JSON.\n\n"
				)
			for _, row in thread_df.iterrows():
				prompt += f"From: {row['from']}\nTo: {row['to']}\nSubject: {row['subject']}\nDate: {row['date']}\n\n{row['body']}\n\n---\n\n"

			try:
				result = ollama.generate(
					model="llama3.2:3b",
					prompt=prompt,
					format="json",  # Request JSON output from Ollama
					options={
					"temperature": (0.2 if style == "factual" else 0.5),
					"top_p": 0.5,
					})

				response_text = result.get("response", "")

				if not response_text:
					logger.warning(f"No response generated for thread {thread_id}. Skipping...")
					return None

				try:
					# Attempt to parse the response as JSON
					parsed_response = json.loads(response_text)
					if not parsed_response or "title" not in parsed_response or ("story"
						not in parsed_response and "description" not in parsed_response):
						continue
					story_title = parsed_response.get("title", "No Title Generated")
					story_content = parsed_response.get("story",
						response_text).strip()  # Fallback to raw text if 'story' key is missing
				except json.JSONDecodeError as e:
					logger.error(
						f"Failed to parse JSON for thread {thread_id}: {e}. Raw response: {response_text[:200]}..."
					)
					save_error_log(f"JSON parsing error for thread {thread_id}: {e}")
					story_title = "Parsing Error - See Logs"
					story_content = response_text.strip(
					)  # Use raw response as story if parsing fails

				# get message_ids in a list
				message_ids = thread_df["message_id"].dropna().tolist()

				stories.append({
					"thread_id": thread_id,
					"title": story_title,
					"story": story_content,
					"related_emails": message_ids,
					"email_count": len(thread_df),
					"style": style,
				})

				if limit is not None and i >= limit:
					logger.info(f"Reached limit of {limit} stories. Stopping generation.")
					break
				i += 1
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
			total=len(df_no_thread) if limit is None else limit,
			desc="Generating stories from non-threaded emails",
		):
			# Construct the prompt for the LLM, instructing it to return JSON with title and story
			if style == "factual":
				prompt = (
					"Based on the following email, write a concise and factual narrative story. "
					"The story should strictly adhere to the information provided in the email, "
					"capturing only the key events, decisions, and emotions showed in input data. "
					"Avoid introducing any external details or creative interpretations not explicitly mentioned. "
					"Focus on summarizing the email's content objectively and include relevant details. "
					"Understand the entities, names and their nicknames, relationships, and events in the email. "
					"In response, provide a JSON object with two keys: 'title' for a concise title "
					"summarizing the story, and 'story' for the narrative itself. "
					"Do not include any additional commentary or explanations outside the JSON.\n\n"
				)
			else:  # style == "creative"
				prompt = (
					"Based on the following email, write a compelling and creative narrative story. "
					"The story should capture the key events, decisions, and emotions of the people involved. "
					"Feel free to add creative elements while staying true to the essence of the email. "
					"In response, provide a JSON object with two keys: 'title' for a creative and engaging title "
					"summarizing the story, and 'story' for the narrative itself. "
					"Do not include any additional commentary or explanations outside the JSON.\n\n"
				)
			prompt += f"From: {row['from']}\nTo: {row['to']}\nSubject: {row['subject']}\nDate: {row['date']}\n\n{row['body']}\n\n---\n\n"

			try:
				result = ollama.generate(
					model="llama3.2:3b",
					prompt=prompt,
					format="json",  # Request JSON output from Ollama
					options={
					"temperature": (0.2 if style == "factual" else 0.5),
					"top_p": 0.5,
					})

				response_text = result.get("response", "")

				if not response_text:
					return None

				try:
					# Attempt to parse the response as JSON
					parsed_response = json.loads(response_text)
					if not parsed_response or "title" not in parsed_response or ("story"
						not in parsed_response and "description" not in parsed_response):
						continue
					story_title = parsed_response.get("title", "No Title Generated")
					if "story" in parsed_response:
						story_content = parsed_response["story"].strip()
					elif "description" in parsed_response:
						story_content = parsed_response["description"].strip()
					else:
						story_content = response_text.strip()
				except json.JSONDecodeError as e:
					story_title = "Title could not be generated"
					story_content = response_text.strip(
					)  # Use raw response as story if parsing fails

				stories.append({
					"message_id": row["message_id"],
					"title": story_title,
					"story": story_content,
					"related_emails": [row["message_id"]],
					"email_count": 1,
					"style": style,
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

		return stories

	@staticmethod
	def identify_key_actors(df: pd.DataFrame, limit: Optional[int] = None):
		"""
		Identify key actors in the email data.

		Args:
			df (pd.DataFrame): DataFrame containing email data.
			limit (Optional[int]): Maximum number of emails to process. If None, process all.

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

		i = 0

		# Process each email with progress bar
		logger.info("Processing emails to build actor network...")
		for _, row in tqdm(df.iterrows(), total=len(df), desc="Building actor network"):
			if limit is not None and i >= limit:
				break
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

	def detect_significant_events(self, df: pd.DataFrame, limit: Optional[int] = None):
		"""
		Detect significant events in the email data.

		Args:
			df (pd.DataFrame): DataFrame containing email data.
			limit (Optional[int]): Maximum number of emails to process. If None, process all.

		Returns:
			list: List of significant events detected in the emails.
		"""
		try:
			logger.info("Detecting significant events...")

			# Count emails per day
			logger.info("Counting emails per day...")
			df = df.dropna(subset=["date"])
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
			i = 0
			for date, count in tqdm(spikes.items(), desc="Processing events"):
				try:
					if limit is not None and i >= limit:
						break
					date = str(date)
					event_date = pd.to_datetime(date)

					# Get emails from the spike day
					_date = pd.to_datetime(df["date"].dt.date)
					event_emails = df[(_date >= event_date)
						and (_date < event_date + timedelta(days=1))]

					# Extract common words from these emails
					if len(event_emails) > 0:
						# Combine all text
						all_text = " ".join(event_emails["body"].fillna(""))

						# Tokenize and count words
						words = re.findall(r"\b\w+\b", all_text.lower())
						words = [
							word for word in words
							if word not in custom_stop_words and len(word) > 2
						]
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
				except Exception as e:
					continue

			# Sort events by deviation
			events.sort(key=lambda x: x["deviation"], reverse=True)

			return events
		except Exception as e:
			logger.error(f"Error detecting significant events: {e}")
			save_error_log(f"Error detecting significant events: {e}")
			return []

	@staticmethod
	def track_topics_over_time(analysis_results: pd.DataFrame, limit: Optional[int] = None):
		"""
		Track how topics evolve over time.

		Args:
			analysis_results (pd.DataFrame): DataFrame containing analysis results
			limit (Optional[int]): Maximum number of time periods to process. If None, process all.

		Returns:
			dict: Dictionary containing topic evolution data
		"""
		logger.info("Tracking topics over time")

		# check for "topic_keywords" column in analysis_results
		if "topic_label" not in analysis_results.columns:
			logger.warning("No topic_label column found in analysis_results")
			return {"time_periods": [], "topic_counts": {}, "topic_keywords": {}}

		if limit is not None:
			# Limit the analysis results to the specified number of rows
			analysis_results = analysis_results.head(limit)

		# Create time-based features
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
		"""Generate a summary for a key actor story using an LLM."""
		name = actor.split("@")[0].replace(".", " ").title()
		prompt = (
			"Generate a concise and insightful summary about a key person based on the following data. "
			"The summary should be a narrative that highlights their role, communication patterns, and key areas of focus. "
			"Provide the output in a JSON object with a single key: 'summary'.\n\n"
			f"Person's Name: {name}\n"
			f"Email Address: {actor}\n"
			f"Metrics: {json.dumps(metrics, indent=2)}\n"
			f"Commonly Used Words: {json.dumps(common_words, indent=2)}\n"
			f"Busiest Day: {busiest_day} (with {busiest_day_count} emails)\n"
			f"Average Response Time: {avg_response_time} hours\n\n"
			"Generate the JSON summary now.")

		try:
			result = ollama.generate(
				model="llama3.2:3b",
				prompt=prompt,
				format="json",
				options={
				"temperature": 0.3,
				"top_p": 0.5
				},
			)
			response_text = result.get("response", "")
			if response_text:
				parsed_response = json.loads(response_text)
				return parsed_response.get("summary", "Summary could not be generated.")
			return "Summary could not be generated."
		except Exception as e:
			logger.error(f"Error generating actor summary for {actor}: {e}")
			save_error_log(f"Error generating actor summary for {actor}: {e}")
			return f"Error generating summary for {actor}. See logs for details."

	@staticmethod
	def _generate_event_summary(
		date,
		email_count,
		deviation,
		common_words,
		event_metrics,
	):
		"""Generate a summary for a significant event story using an LLM."""
		prompt = (
			"Generate a concise and insightful summary about a significant event based on the following data. "
			"The summary should be a narrative that explains the event's significance, participant engagement, and key topics. "
			"Provide the output in a JSON object with a single key: 'summary'.\n\n"
			f"Event Date: {date}\n"
			f"Email Count: {email_count} (Deviation: {deviation:.1f} std devs above normal)\n"
			f"Commonly Used Words: {json.dumps(common_words, indent=2)}\n"
			f"Event Metrics: {json.dumps(event_metrics, indent=2)}\n\n"
			"Generate the JSON summary now.")

		try:
			result = ollama.generate(
				model="llama3.2:3b",
				prompt=prompt,
				format="json",
				options={
				"temperature": 0.3,
				"top_p": 0.5
				},
			)
			response_text = result.get("response", "")
			if response_text:
				parsed_response = json.loads(response_text)
				return parsed_response.get("summary", "Summary could not be generated.")
			return "Summary could not be generated."
		except Exception as e:
			logger.error(f"Error generating event summary for {date}: {e}")
			save_error_log(f"Error generating event summary for {date}: {e}")
			return f"Error generating summary for {date}. See logs for details."

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
		"""Generate a summary for a topic evolution story using an LLM."""
		prompt = (
			"Generate a concise and insightful summary about a topic's evolution based on the following data. "
			"The summary should be a narrative that explains the topic's trend, peak activity, and key characteristics. "
			"Provide the output in a JSON object with a single key: 'summary'.\n\n"
			f"Topic ID: {topic_id}\n"
			f"Keywords: {json.dumps(keywords, indent=2)}\n"
			f"Topic Trend: {json.dumps(topic_trend, indent=2)}\n\n"
			"Generate the JSON summary now.")

		try:
			result = ollama.generate(
				model="llama3.2:3b",
				prompt=prompt,
				format="json",
				options={
				"temperature": 0.3,
				"top_p": 0.5
				},
			)
			response_text = result.get("response", "")
			if response_text:
				parsed_response = json.loads(response_text)
				return parsed_response.get("summary", "Summary could not be generated.")
			return "Summary could not be generated."
		except Exception as e:
			logger.error(f"Error generating topic summary for {topic_id}: {e}")
			save_error_log(f"Error generating topic summary for {topic_id}: {e}")
			return f"Error generating summary for {topic_id}. See logs for details."

	@staticmethod
	def _clean_for_json(obj):
		"""Recursively clean data for JSON serialization."""
		if isinstance(obj, dict):
			return {k: StoryDevelopment._clean_for_json(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [StoryDevelopment._clean_for_json(i) for i in obj]
		if pd.isna(obj):
			return None
		if isinstance(obj, (datetime, pd.Timestamp)):
			return obj.isoformat()
		if isinstance(obj, timedelta):
			return str(obj)
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return obj

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
				try:
					# Get emails sent by this actor
					actor_emails = analysis_result[analysis_result["from"].str.contains(actor,
						na=False)]

					if len(actor_emails) > 0:
						# Get related emails (emails in threads where actor participated)
						related_emails = []
						for _, email in actor_emails.iterrows():
							try:
								thread_emails = analysis_result[
									analysis_result["subject"].str.contains(email["subject"],
									na=False,
									regex=False)]
								for _, thread_email in thread_emails.iterrows():
									message_id = thread_email.get("message_id", None)
									if message_id is not None and message_id not in related_emails:
										related_emails.append(message_id)
							except Exception as e:
								logger.error(
									f"Error processing email thread for {email['subject']}: {e}")
								save_error_log(
									f"Error processing email thread for {email['subject']}: {e}")

						# Use clean_body column if available, otherwise use body
						text_column = ("clean_body"
							if "clean_body" in actor_emails.columns else "body")
						all_text = " ".join(actor_emails[text_column].fillna(""))
						words = re.findall(r"\b\w+\b", all_text.lower())
						words = [
							word for word in words
							if word not in custom_stop_words and len(word) > 2
						]
						common_words = Counter(words).most_common(20)
						sample_subjects = actor_emails["subject"].head(5).tolist()

						daily_patterns = actor_emails.groupby(
							actor_emails["date"].dt.day_name()).size()
						busiest_day = daily_patterns.idxmax()
						busiest_day_count = daily_patterns.max()

						# Calculate average response time
						response_times = []
						for _, email in actor_emails.iterrows():
							try:
								if pd.notna(email["date"]):
									subject_to_match = (email["subject"][0]
										if isinstance(email["subject"], list)
										and len(email["subject"]) > 0 else email["subject"])
									replies = analysis_result[(analysis_result["subject"].str.
										contains(subject_to_match, na=False, regex=False))
										& (analysis_result["date"] > email["date"])]
									if not replies.empty:
										response_time = (replies["date"].min() -
											email["date"]).total_seconds() / 3600
										response_times.append(response_time)
							except Exception as e:
								logger.error(
									f"Error calculating response time for email {email['subject']}: {e}"
								)
								save_error_log(
									f"Error calculating response time for email {email['subject']}: {e}"
								)

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
							self._clean_for_json(metrics),
							self._clean_for_json(common_words),
							busiest_day,
							int(busiest_day_count),
							avg_response_time,
							),
						}
						summaries.append(story)
				except Exception as e:
					logger.error(f"Error processing key actors: {e}")
					save_error_log(f"Error processing key actors: {e}")

			# 2. Stories based on significant events
			for event in significant_events[:5]:
				try:
					event_date = event["date"]
					event_emails = analysis_result[(pd.to_datetime(analysis_result["date"].dt.date)
						>= pd.to_datetime(event_date))
						and (pd.to_datetime(analysis_result["date"].dt.date) <
						pd.to_datetime(event_date) + timedelta(days=1))]

					# Get all related emails (including replies and forwards)
					related_emails = []
					for _, email in event_emails.iterrows():
						# Get emails in the same thread
						thread_emails = analysis_result[analysis_result["subject"].str.contains(
							email["subject"], na=False)]
						for _, thread_email in thread_emails.iterrows():
							message_id = thread_email.get("message_id", None)
							if message_id is not None and message_id not in related_emails:
								related_emails.append(message_id)

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
						self._clean_for_json(event["common_words"]),
						self._clean_for_json(event_metrics),
						),
					}
					summaries.append(story)
				except Exception as e:
					logger.error(f"Error processing significant event {event['date']}: {e}")

			# 3. Stories based on topic evolution
			if topic_evolution and "topic_keywords" in topic_evolution:
				for topic_id, keywords in list(topic_evolution["topic_keywords"].items())[:5]:
					# Get emails related to this topic
					topic_emails = []
					try:
						for keyword in keywords[:5]:  # Use top 5 keywords
							keyword_emails = analysis_result[analysis_result["body"].str.contains(
								keyword, case=False, na=False)]
							for _, email in keyword_emails.iterrows():
								topic_emails.append({
									"subject": email["subject"],
									"date": email["date"],
									"from": email["from"],
									"to": email["to"],
									"body": email["body"],
									"matching_keyword": keyword,
								})

						# Extract topic number
						topic_num = int(topic_id.split()[-1])

						# Get topic evolution data
						topic_counts = topic_evolution.get("topic_counts", {})
						topic_trend = self._analyze_topic_trend(topic_counts, topic_num)

						story = {
							"title":
							f"The Evolution of {topic_id}",
							"type":
							"topic_evolution",
							"topic_id":
							topic_id,
							"keywords":
							keywords,
							"topic_metrics": {
							"trend": topic_trend["trend"],
							"peak_period": topic_trend["peak_period"],
							"peak_count": topic_trend["peak_count"],
							},
							"related_emails":
							topic_emails,
							"summary":
							self._generate_topic_summary(
							topic_id,
							self._clean_for_json(keywords),
							self._clean_for_json(topic_trend),
							),
						}
						summaries.append(story)
					except Exception as e:
						logger.error(f"Error processing topic {topic_id}: {e}")
						save_error_log(f"Error processing topic {topic_id}: {e}")

			return summaries

	def develop_story(
		self,
		processed_data: Optional[pd.DataFrame],
		analysis_results: Optional[pd.DataFrame],
		limit: Optional[int] = None,
		db_manager: Optional[DatabaseManager] = None,
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

			processed_data = sort_emails_by_date(processed_data)
			analysis_results = sort_emails_by_date(analysis_results)

			processed_data["date"] = pd.to_datetime(processed_data["date"], errors="coerce")
			analysis_results["date"] = pd.to_datetime(analysis_results["date"], errors="coerce")

			# check if analysis_results has NaN values in 'date' column
			if analysis_results["date"].isna().any():
				logger.warning(
					"Analysis results contain NaN values in 'date' column. Attempting to clean data."
				)
				analysis_results = analysis_results.dropna(subset=["date"])

			num_threads = max(os.cpu_count() or 1, 4)
			with FutureExecutor(max_workers=num_threads) as executor:
				futures = {
					"stories_from_threads":
					executor.submit(
					self.generate_stories_from_threads,
					analysis_results,
					limit=limit,
					style="factual",
					),
					"stories_from_non_threads":
					executor.submit(
					self.generate_non_threaded_stories,
					analysis_results,
					limit=limit,
					style="factual",
					),
					"stories_from_threads_creative":
					executor.submit(
					self.generate_stories_from_threads,
					analysis_results,
					limit=limit,
					style="creative",
					),
					"stories_from_non_threads_creative":
					executor.submit(
					self.generate_non_threaded_stories,
					analysis_results,
					limit=limit,
					style="creative",
					),
					"identify_key_actors":
					executor.submit(self.identify_key_actors, processed_data, limit=limit),
					"detect_significant_events":
					executor.submit(self.detect_significant_events, processed_data, limit=limit),
					"track_topics_over_time":
					executor.submit(self.track_topics_over_time, analysis_results, limit=limit),
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
			threaded_stories_creative = results.get("stories_from_threads_creative", [])
			non_threaded_stories_creative = results.get("stories_from_non_threads_creative", [])

			# Save stories to JSON files
			story_outputs = {
				"threaded_stories": threaded_stories,
				"non_threaded_stories": non_threaded_stories,
				"threaded_stories_creative": threaded_stories_creative,
				"non_threaded_stories_creative": non_threaded_stories_creative,
			}

			for name, story_data in story_outputs.items():
				if story_data:
					file_path = os.path.join(
						self.output_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
					with open(file_path, "w", encoding="utf-8") as f:
						json.dump(self._clean_for_json(story_data), f, ensure_ascii=False, indent=4)
					logger.info(f"{name.replace('_', ' ').title()} saved to {file_path}")

			# Save stories to database
			all_threaded_stories = threaded_stories + threaded_stories_creative
			if all_threaded_stories:
				threaded_stories_df = pd.DataFrame(all_threaded_stories)
				threaded_stories_df['related_emails'] = threaded_stories_df['related_emails'].apply(
					lambda x: json.dumps(self._clean_for_json(x)) if x else None)
				if db_manager is not None:
					db_manager.insert_from_dataframe(threaded_stories_df,
						"threaded_stories",
						columns=[
						"thread_id",
						"style",
						"title",
						"story",
						"related_emails",
						"email_count",
						])
					logger.info(
						f"Inserted {len(threaded_stories_df)} threaded stories into the database.")

			all_non_threaded_stories = non_threaded_stories + non_threaded_stories_creative
			if all_non_threaded_stories:
				non_threaded_stories_df = pd.DataFrame(all_non_threaded_stories)
				non_threaded_stories_df['related_emails'] = non_threaded_stories_df[
					'related_emails'].apply(lambda x: json.dumps(self._clean_for_json(x))
					if x else None)
				if db_manager is not None:
					db_manager.insert_from_dataframe(non_threaded_stories_df,
						"non_threaded_stories",
						columns=[
						"message_id",
						"style",
						"title",
						"story",
						"related_emails",
						"email_count",
						])
					logger.info(
						f"Inserted {len(non_threaded_stories_df)} non-threaded stories into the database."
					)

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

			# Save summaries to JSON file
			if summaries:
				summaries_file = os.path.join(
					self.output_dir, f"summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
				with open(summaries_file, "w", encoding="utf-8") as f:
					json.dump(self._clean_for_json(summaries), f, ensure_ascii=False, indent=4)
				logger.info(f"Summaries saved to {summaries_file}")

			# Save summaries to database
			if summaries and db_manager is not None:
				summaries_for_db = []
				for s in summaries:
					summary_id = str(uuid.uuid4())
					original_id = None
					summary_metadata = {}

					if s["type"] == "key_actor":
						original_id = s.get("actor")
						summary_metadata = {
							"metrics": s.get("metrics"),
							"common_topics": s.get("common_topics"),
							"sample_subjects": s.get("sample_subjects"),
							"communication_patterns": s.get("communication_patterns"),
							"related_emails": s.get("related_emails"),
						}
					elif s["type"] == "significant_event":
						original_id = s.get("date")
						summary_metadata = {
							"email_count": s.get("email_count"),
							"common_words": s.get("common_words"),
							"sample_subjects": s.get("sample_subjects"),
							"event_metrics": s.get("event_metrics"),
							"participants": s.get("participants"),
							"related_emails": s.get("related_emails"),
						}
					elif s["type"] == "topic_evolution":
						original_id = s.get("topic_id")
						summary_metadata = {
							"keywords": s.get("keywords"),
							"topic_metrics": s.get("topic_metrics"),
							"related_emails": s.get("related_emails"),
						}

					summaries_for_db.append({
						"summary_id":
						summary_id,
						"original_id":
						original_id,
						"summary_title":
						s.get("title"),
						"summary_content":
						s.get("summary"),
						"summary_type":
						s.get("type"),
						"summary_metadata":
						json.dumps(self._clean_for_json(summary_metadata)),
					})

				summaries_df = pd.DataFrame(summaries_for_db)
				db_manager.insert_from_dataframe(
					summaries_df,
					"summaries",
					columns=[
					"summary_id",
					"original_id",
					"summary_title",
					"summary_content",
					"summary_type",
					"summary_metadata",
					],
				)
				logger.info(f"Inserted {len(summaries_df)} summaries into the database.")

			return {
				"threaded_stories": threaded_stories,
				"non_threaded_stories": non_threaded_stories,
				"threaded_stories_creative": threaded_stories_creative,
				"non_threaded_stories_creative": non_threaded_stories_creative,
				"key_actors": key_actors,
				"significant_events": significant_events,
				"topic_evolution": topic_evolution,
				"summaries": summaries,
			}

		except Exception as e:
			logger.error(f"Error generating story: {e}")


if __name__ == "__main__":
	# Example usage
	story_dev = StoryDevelopment(
		processed_data_dir="path/to/processed/data",
		analysis_results_dir="path/to/analysis/results",
		output_dir="path/to/output",
	)
	story_dev.develop_story(None, None, limit=100)
