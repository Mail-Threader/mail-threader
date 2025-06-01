import json
import os
import pickle
import re
from collections import Counter
from datetime import datetime, timedelta

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")


class StoryDevelopment:
    """
    Class responsible for story development tasks:
    - Identifying key actors and their relationships
    - Tracking topics over time
    - Detecting significant events
    - Constructing narratives from email threads
    - Generating story summaries
    """

    def __init__(
        self,
        input_dir="./processed_data/",
        analysis_dir="./analysis_results/",
        output_dir="./stories/",
    ):
        """
        Initialize the StoryDevelopment class.

        Args:
            input_dir (str): Directory containing processed email data
            analysis_dir (str): Directory containing analysis results
            output_dir (str): Directory to store generated stories
        """
        self.input_dir = input_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize stopwords
        self.stop_words = set(stopwords.words("english"))

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

    def identify_key_actors(self, df, top_n=20):
        """
        Identify key actors in the email dataset based on email frequency and centrality.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            top_n (int): Number of top actors to identify

        Returns:
            dict: Dictionary containing key actors and their metrics
        """
        logger.info("Identifying key actors")

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
            sender_emails = (
                re.findall(email_pattern, row["from"]) if row["from"] is not None else []
            )
            recipient_emails = re.findall(email_pattern, row["to"]) if row["to"] is not None else []

            # Add edges from sender to recipients
            for sender in sender_emails:
                sender_counts[sender] += 1
                for recipient in recipient_emails:
                    recipient_counts[recipient] += 1
                    edge_weights[(sender, recipient)] += 1

        # Add nodes and edges to the graph
        for (sender, recipient), weight in edge_weights.items():
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
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)

            # Betweenness centrality (who connects different groups)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G)))

            # PageRank (importance based on connections)
            pagerank = nx.pagerank(G)

            # Combine metrics
            actor_metrics = {}
            for actor in G.nodes():
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

            return {"top_actors": top_actors, "graph": G}

        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            return {
                "top_actors": {
                    actor: {"sent": count, "received": recipient_counts.get(actor, 0)}
                    for actor, count in sender_counts.most_common(top_n)
                },
                "graph": G,
            }

    def track_topics_over_time(self, df, analysis_results):
        """
        Track how topics evolve over time.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results

        Returns:
            dict: Dictionary containing topic evolution data
        """
        logger.info("Tracking topics over time")

        if (
            analysis_results is None
            or "topics" not in analysis_results
            or analysis_results["topics"] is None
        ):
            logger.warning("No topic modeling results available for tracking")
            return None

        # Check if date column exists and has valid data
        if "date" not in df.columns or df["date"].isna().all():
            logger.warning("No date information available for topic tracking")
            return None

        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                return None

        # Drop rows with missing dates
        df_with_date = df.dropna(subset=["date"]).copy()
        if len(df_with_date) == 0:
            logger.warning("No valid dates available for topic tracking")
            return None

        # Get topic assignments and document-topic matrix
        if "dominant_topic" not in df.columns and "doc_topic_matrix" in analysis_results["topics"]:
            # Get indices of rows with valid dates
            valid_indices = df.dropna(subset=["date"]).index

            # Assign dominant topic to each document
            doc_topic_matrix = analysis_results["topics"]["doc_topic_matrix"]

            # Filter doc_topic_matrix to match the rows in df_with_date
            if len(doc_topic_matrix) != len(df_with_date):
                logger.info(
                    f"Filtering doc_topic_matrix from {len(doc_topic_matrix)} to {len(df_with_date)} rows"
                )
                # Get the indices in the original dataframe
                valid_indices_list = valid_indices.tolist()
                # Filter doc_topic_matrix to only include rows with valid dates
                if len(valid_indices_list) <= len(doc_topic_matrix):
                    doc_topic_matrix = doc_topic_matrix[valid_indices_list]
                else:
                    logger.warning(
                        f"Cannot filter doc_topic_matrix: valid indices ({len(valid_indices_list)}) > matrix rows ({len(doc_topic_matrix)})"
                    )
                    return None

            df_with_date["dominant_topic"] = np.argmax(doc_topic_matrix, axis=1) + 1

        # Create time-based features
        df_with_date["year"] = df_with_date["date"].dt.year
        df_with_date["month"] = df_with_date["date"].dt.month
        df_with_date["week"] = df_with_date["date"].dt.isocalendar().week

        # Create a date string for grouping by month
        df_with_date["month_year"] = df_with_date["date"].dt.strftime("%Y-%m")

        # Group by month and topic
        topic_time_counts = (
            df_with_date.groupby(["month_year", "dominant_topic"]).size().unstack(fill_value=0)
        )

        # Get topic keywords
        topics = analysis_results["topics"]["topics"]

        # Create a dictionary with topic evolution data
        topic_evolution = {
            "time_periods": list(topic_time_counts.index),
            "topic_counts": topic_time_counts.to_dict(),
            "topic_keywords": topics,
        }

        return topic_evolution

    def detect_significant_events(self, df, window_size=7, threshold=2):
        """
        Detect significant events based on email volume spikes and content.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            window_size (int): Size of the rolling window in days
            threshold (float): Number of standard deviations above mean to consider as a spike

        Returns:
            list: List of detected events with dates and related emails
        """
        logger.info("Detecting significant events")

        # Check if date column exists and has valid data
        if "date" not in df.columns or df["date"].isna().all():
            logger.warning("No date information available for event detection")
            return []

        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                return []

        # Drop rows with missing dates
        df_with_date = df.dropna(subset=["date"]).copy()
        if len(df_with_date) == 0:
            logger.warning("No valid dates available for event detection")
            return []

        # Sort by date
        df_with_date = df_with_date.sort_values("date")

        # Count emails per day
        daily_counts = df_with_date.groupby(df_with_date["date"].dt.date).size()

        # Calculate rolling statistics
        rolling_mean = daily_counts.rolling(window=window_size, min_periods=1).mean()
        rolling_std = daily_counts.rolling(window=window_size, min_periods=1).std()

        # Identify spikes
        spikes = daily_counts[daily_counts > (rolling_mean + threshold * rolling_std)]

        # Extract events
        events = []
        for date, count in spikes.items():
            # Convert date to datetime for filtering
            event_date = pd.to_datetime(date)

            # Get emails from the spike day
            event_emails = df_with_date[
                (pd.to_datetime(df_with_date["date"].dt.date) >= event_date)
                & (pd.to_datetime(df_with_date["date"].dt.date) < event_date + timedelta(days=1))
            ]

            # Extract common words from these emails
            if len(event_emails) > 0:
                # Use clean_body column if available, otherwise use body
                text_column = "clean_body" if "clean_body" in event_emails.columns else "body"

                # Combine all text
                all_text = " ".join(event_emails[text_column].fillna(""))

                # Tokenize and count words
                words = re.findall(r"\b\w+\b", all_text.lower())
                words = [word for word in words if word not in self.stop_words and len(word) > 2]
                common_words = Counter(words).most_common(10)

                # Get sample subjects
                sample_subjects = event_emails["subject"].head(5).tolist()

                events.append(
                    {
                        "date": date,
                        "email_count": count,
                        "normal_level": rolling_mean[date],
                        "std_dev": rolling_std[date],
                        "deviation": (
                            (count - rolling_mean[date]) / rolling_std[date]
                            if rolling_std[date] > 0
                            else 0
                        ),
                        "common_words": common_words,
                        "sample_subjects": sample_subjects,
                        "email_ids": event_emails.index.tolist(),
                    }
                )

        # Sort events by deviation
        events.sort(key=lambda x: x["deviation"], reverse=True)

        return events

    def construct_email_threads(self, df):
        """
        Construct email threads based on subject lines and timestamps.

        Args:
            df (pandas.DataFrame): DataFrame containing email data

        Returns:
            dict: Dictionary of email threads
        """
        logger.info("Constructing email threads")

        # Check if required columns exist
        if "subject" not in df.columns or "date" not in df.columns:
            logger.warning("Required columns missing for thread construction")
            return {}

        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception:
                pass

        # Create a copy of the dataframe with only the necessary columns
        thread_df = df[["subject", "date", "from", "to", "body"]].copy()

        # Clean subject lines for better matching
        def clean_subject(subject):
            # Remove 'Re:', 'Fwd:', etc.
            cleaned = re.sub(r"^(re|fwd|fw):\s*", "", subject.lower(), flags=re.IGNORECASE)
            # Remove other common prefixes/suffixes
            cleaned = re.sub(r"\[.*?\]", "", cleaned)
            # Remove extra whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        thread_df["clean_subject"] = thread_df["subject"].fillna("").apply(clean_subject)

        # Group by cleaned subject
        threads = {}
        for subject, group in thread_df.groupby("clean_subject"):
            if not subject:  # Skip empty subjects
                continue

            # Sort by date
            sorted_group = group.sort_values("date")

            # Only consider as thread if there are multiple emails
            if len(sorted_group) > 1:
                threads[subject] = sorted_group.to_dict("records")

        # Sort threads by size (number of emails)
        sorted_threads = {
            k: v for k, v in sorted(threads.items(), key=lambda item: len(item[1]), reverse=True)
        }

        return sorted_threads

    def generate_story_summaries(
        self,
        df,
        analysis_results,
        key_actors,
        topic_evolution,
        significant_events,
        email_threads,
    ):
        """
        Generate story summaries based on all available data.

        Args:
            df (pandas.DataFrame): DataFrame containing email data
            analysis_results (dict): Dictionary containing analysis results
            key_actors (dict): Dictionary containing key actors data
            topic_evolution (dict): Dictionary containing topic evolution data
            significant_events (list): List of significant events
            email_threads (dict): Dictionary of email threads

        Returns:
            list: List of story summaries
        """
        logger.info("Generating story summaries")

        stories = []

        # 1. Stories based on key actors
        if key_actors and "top_actors" in key_actors:
            for actor, metrics in list(key_actors["top_actors"].items())[:5]:  # Top 5 actors
                # Get emails sent by this actor
                actor_emails = df[df["from"].str.contains(actor, na=False)]

                if len(actor_emails) > 0:
                    # Use clean_body column if available, otherwise use body
                    text_column = "clean_body" if "clean_body" in actor_emails.columns else "body"

                    # Combine all text
                    all_text = " ".join(actor_emails[text_column].fillna(""))

                    # Tokenize and count words
                    words = re.findall(r"\b\w+\b", all_text.lower())
                    words = [
                        word for word in words if word not in self.stop_words and len(word) > 2
                    ]
                    common_words = Counter(words).most_common(20)

                    # Get sample subjects
                    sample_subjects = actor_emails["subject"].head(5).tolist()

                    # Create a story
                    story = {
                        "title": f"The Story of {actor}",
                        "type": "key_actor",
                        "actor": actor,
                        "metrics": metrics,
                        "common_topics": common_words,
                        "sample_subjects": sample_subjects,
                        "summary": f"{actor} is a key actor in the Enron email dataset, having sent {metrics['sent']} emails and received {metrics['received']} emails. They frequently discuss topics related to {', '.join([word for word, _ in common_words[:5]])}.",
                    }

                    stories.append(story)

        # 2. Stories based on significant events
        for event in significant_events[:5]:  # Top 5 events
            event_date = event["date"]

            # Create a story
            story = {
                "title": f"Significant Event on {event_date}",
                "type": "significant_event",
                "date": event_date,
                "email_count": event["email_count"],
                "common_words": event["common_words"],
                "sample_subjects": event["sample_subjects"],
                "summary": f"On {event_date}, there was a significant spike in email activity with {event['email_count']} emails, which is {event['deviation']:.1f} standard deviations above normal. The emails frequently mention {', '.join([word for word, _ in event['common_words'][:5]])}.",
            }

            stories.append(story)

        # 3. Stories based on email threads
        for subject, thread in list(email_threads.items())[:5]:  # Top 5 threads
            if len(thread) >= 3:  # Only consider threads with at least 3 emails
                # Extract participants
                participants = set()
                for email in thread:
                    if "sender" in email and email["sender"]:
                        sender_emails = re.findall(r"[\w\.-]+@[\w\.-]+", email["sender"])
                        participants.update(sender_emails)

                # Create a story
                story = {
                    "title": f"Email Thread: {subject}",
                    "type": "email_thread",
                    "subject": subject,
                    "num_emails": len(thread),
                    "participants": list(participants),
                    "start_date": thread[0]["date"] if "date" in thread[0] else None,
                    "end_date": thread[-1]["date"] if "date" in thread[-1] else None,
                    "summary": f"This email thread contains {len(thread)} messages about '{subject}', involving {len(participants)} participants over a period of time.",
                }

                stories.append(story)

        # 4. Stories based on topic evolution
        if topic_evolution and "topic_keywords" in topic_evolution:
            for topic_id, keywords in list(topic_evolution["topic_keywords"].items())[
                :5
            ]:  # Top 5 topics
                # Extract topic number
                topic_num = int(topic_id.split()[-1])

                # Create a story
                story = {
                    "title": f"The Evolution of {topic_id}",
                    "type": "topic_evolution",
                    "topic_id": topic_id,
                    "keywords": keywords,
                    "summary": f"{topic_id} is characterized by keywords like {', '.join(keywords[:5])}. This topic shows how discussions about {keywords[0]} and {keywords[1]} evolved over time in the Enron email dataset.",
                }

                stories.append(story)

        return stories

    def save_stories(self, stories, save_to_db=True):
        """
        Save generated stories to files and database.

        Args:
            stories (list): List of story summaries

        Returns:
            str: Path to the saved stories
        """
        if not stories:
            logger.warning("No stories to save")
            return None

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = os.path.join(self.output_dir, f"stories_{timestamp}.json")
        try:
            with open(json_path, "w") as f:
                json.dump(stories, f, indent=2, default=str)
            logger.info(f"Saved stories to {json_path}")
        except Exception as e:
            logger.error(f"Error saving stories to JSON: {e}")

        # Save as HTML
        html_path = os.path.join(self.output_dir, f"stories_{timestamp}.html")
        try:
            html_content = self._generate_html_report(stories)
            with open(html_path, "w") as f:
                f.write(html_content)
            logger.info(f"Saved stories to {html_path}")
        except Exception as e:
            logger.error(f"Error saving stories to HTML: {e}")

        if save_to_db:
            # Save to the database
            try:
                # Convert stories list to DataFrame
                stories_df = pd.DataFrame(stories)

                # Add timestamp column
                stories_df["timestamp"] = timestamp

                # Save to PostgreSQL database
                save_to_postgresql(
                    stories_df,
                    table_name="stories",
                    if_exists="append",
                    success_message=f"Saved {len(stories)} stories to database",
                )
            except Exception as e:
                logger.error(f"Error saving stories to database: {e}")

        return json_path

    def _generate_html_report(self, stories):
        """
        Generate an HTML report from the stories.

        Args:
            stories (list): List of story summaries

        Returns:
            str: HTML content
        """
        html = (
            """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Enron Email Stories</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #2c3e50; }
                        h2 { color: #3498db; margin-top: 30px; }
                        .story { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                        .story-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
                        .story-summary { margin-bottom: 15px; }
                        .story-details { font-size: 0.9em; color: #555; }
                        .key-actor { background-color: #e8f4f8; }
                        .significant-event { background-color: #f8f4e8; }
                        .email-thread { background-color: #f4f8e8; }
                        .topic-evolution { background-color: #f8e8f4; }
                    </style>
                </head>
                <body>
                    <h1>Enron Email Stories</h1>
                    <p>Generated on """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
        """
        )

        # Group stories by type
        story_types = {
            "key_actor": [],
            "significant_event": [],
            "email_thread": [],
            "topic_evolution": [],
        }

        for story in stories:
            story_type = story.get("type", "other")
            if story_type in story_types:
                story_types[story_type].append(story)

        # Add stories by type
        if story_types["key_actor"]:
            html += "<h2>Key Actor Stories</h2>"
            for story in story_types["key_actor"]:
                html += self._story_to_html(story, "key-actor")

        if story_types["significant_event"]:
            html += "<h2>Significant Event Stories</h2>"
            for story in story_types["significant_event"]:
                html += self._story_to_html(story, "significant-event")

        if story_types["email_thread"]:
            html += "<h2>Email Thread Stories</h2>"
            for story in story_types["email_thread"]:
                html += self._story_to_html(story, "email-thread")

        if story_types["topic_evolution"]:
            html += "<h2>Topic Evolution Stories</h2>"
            for story in story_types["topic_evolution"]:
                html += self._story_to_html(story, "topic-evolution")

        html += """
        </body>
        </html>
        """

        return html

    def _story_to_html(self, story, css_class):
        """
        Convert a story to HTML.

        Args:
            story (dict): Story dictionary
            css_class (str): CSS class for styling

        Returns:
            str: HTML representation of the story
        """
        html = f"<div class='story {css_class}'>"
        html += f"<div class='story-title'>{story['title']}</div>"
        html += f"<div class='story-summary'>{story['summary']}</div>"
        html += "<div class='story-details'>"

        # Add type-specific details
        if story["type"] == "key_actor":
            html += f"<p>Emails sent: {story['metrics']['sent']}</p>"
            html += f"<p>Emails received: {story['metrics']['received']}</p>"
            html += (
                "<p>Common topics: "
                + ", ".join([f"{word} ({count})" for word, count in story["common_topics"][:10]])
                + "</p>"
            )
            if "sample_subjects" in story and story["sample_subjects"]:
                html += "<p>Sample email subjects:</p><ul>"
                for subject in story["sample_subjects"]:
                    html += f"<li>{subject}</li>"
                html += "</ul>"

        elif story["type"] == "significant_event":
            html += f"<p>Date: {story['date']}</p>"
            html += f"<p>Email count: {story['email_count']}</p>"
            html += (
                "<p>Common words: "
                + ", ".join([f"{word} ({count})" for word, count in story["common_words"][:10]])
                + "</p>"
            )
            if "sample_subjects" in story and story["sample_subjects"]:
                html += "<p>Sample email subjects:</p><ul>"
                for subject in story["sample_subjects"]:
                    html += f"<li>{subject}</li>"
                html += "</ul>"

        elif story["type"] == "email_thread":
            html += f"<p>Subject: {story['subject']}</p>"
            html += f"<p>Number of emails: {story['num_emails']}</p>"
            html += (
                f"<p>Participants: {', '.join(story['participants'][:5])}"
                + (
                    f" and {len(story['participants']) - 5} more"
                    if len(story["participants"]) > 5
                    else ""
                )
                + "</p>"
            )
            if story["start_date"] and story["end_date"]:
                html += f"<p>Time span: {story['start_date']} to {story['end_date']}</p>"

        elif story["type"] == "topic_evolution":
            html += f"<p>Topic: {story['topic_id']}</p>"
            html += f"<p>Keywords: {', '.join(story['keywords'][:10])}</p>"

        html += "</div></div>"
        return html

    def develop_stories(self, df=None, analysis_results=None):
        """
        Develop stories from email data.

        Args:
            df (pandas.DataFrame, optional): DataFrame containing email data.
                If not provided, data will be loaded from the most recent file.
            analysis_results (dict, optional): Dictionary containing analysis results.
                If not provided, results will be loaded from the most recent file.

        Returns:
            dict: Dictionary containing story development results
        """
        # Load data if not provided
        if df is None or df.empty:
            df = self.load_data()
            if df.empty:
                logger.error("No data available for story development")
                return {}

        # Load analysis results if not provided
        if analysis_results is None:
            analysis_results = self.load_analysis_results()

        logger.info(f"Developing stories from {len(df)} emails")

        # 1. Identify key actors
        key_actors = self.identify_key_actors(df)

        # 2. Track topics over time
        topic_evolution = self.track_topics_over_time(df, analysis_results)

        # 3. Detect significant events
        significant_events = self.detect_significant_events(df)

        # 4. Construct email threads
        email_threads = self.construct_email_threads(df)

        # 5. Generate story summaries
        stories = self.generate_story_summaries(
            df,
            analysis_results,
            key_actors,
            topic_evolution,
            significant_events,
            email_threads,
        )

        # 6. Save stories
        stories_path = self.save_stories(stories)

        # Return results
        results = {
            "key_actors": key_actors,
            "topic_evolution": topic_evolution,
            "significant_events": significant_events,
            "email_threads": email_threads,
            "stories": stories,
            "stories_path": stories_path,
        }

        return results


if __name__ == "__main__":
    # Create StoryDevelopment instance
    story_dev = StoryDevelopment()

    # Load data
    df = story_dev.load_data()

    # Load analysis results
    analysis_results = story_dev.load_analysis_results()

    # Develop stories
    results = story_dev.develop_stories(df, analysis_results)

    print("Done!")
