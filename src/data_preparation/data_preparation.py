import os
import re
import time
import dateparser
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from cleantext.clean import clean
import string
from itertools import groupby
import spacy
from spacy.cli.download import download as spacy_download
import html
from bs4 import BeautifulSoup
from typing import Optional
from loguru import logger
from database import DatabaseManager
from utils import load_processed_df


class DataPreparation:
	"""
	Author - Parinaz Teimouri
	Matriculation Number - 319905
	Class responsible for data preparation and storage tasks:
	- Loading email data
	- Cleaning and preprocessing text
	- Extracting metadata (sender, recipient, date, etc.)
	- Storing processed data
	"""

	def __init__(self, input_dir="./data/", output_dir="./processed_data/"):
		"""
		Initialize the DataPreparation class.

		Args:
			email_dir (str): Directory containing the email files
			output_dir (str): Directory to store processed data
		"""
		self.input_dir = input_dir
		self.output_dir = output_dir

		try:
			self.nlp = spacy.load("en_core_web_sm")
		except Exception as e:
			logger.error(f"Error loading spaCy model: {e}")
			logger.warning("Downloading the 'en_core_web_sm' model...")
			spacy_download("en_core_web_sm")
			self.nlp = spacy.load("en_core_web_sm")

		# Create output directory if it doesn't exist
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	def process_all_emails(self, skip=False, limit: Optional[int] = None):
		"""Reads and processes all email files in the given folder and shows progress with stats."""
		data = {
			"message_id": [],
			"main_id": [],
			"filename": [],
			"type": [],
			"date": [],
			"from": [],
			"to": [],
			"X-From": [],
			"X-To": [],
			"subject": [],
			"cc": [],
			"X-cc": [],
			"body": [],
		}
		main_count = 0
		original_count = 0
		forwarded_count = 0
		total_emails = 0
		start_time = time.time()
		total_files = sum(len(files) for _, _, files in os.walk(self.input_dir))
		logger.info(f"📂 Looking for files in: {os.path.join(self.input_dir)}")

		i = 0
		with tqdm(total=total_files, desc="📬 Processing emails") as pbar:
			for root, dirs, files in os.walk(self.input_dir):
				for filename in files:
					file_path = os.path.join(root, filename)
					if os.path.isfile(file_path):
						with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
							if limit is not None and i >= limit:
								break
							i += 1
							raw_text = f.read()
							email_data = self.extract_all_emails(raw_text)
							for email in email_data:
								if (email.get("from") or email.get("to") or email.get("subject")):
									email["filename"] = filename
									for key in data.keys():
										if key in email:
											value = email.get(key)
											if isinstance(value, set):
												data[key].append(
													next(iter(value)) if value else None)
											else:
												data[key].append(value)
										else:
											data[key].append(None)
									total_emails += 1
									if email.get("type") == "original":
										original_count += 1
									elif email.get("type") == "forwarded":
										forwarded_count += 1
									elif email.get("type") == "main":
										main_count += 1
					# Update tqdm bar with counts
					pbar.set_postfix({
						"Total": total_emails,
						"Original": original_count,
						"Forwarded": forwarded_count,
						"main": main_count,
					})
					pbar.update(1)
		end_time = time.time()
		elapsed = end_time - start_time
		logger.success(f"\n✅ Done! Total emails: {total_emails}")
		logger.debug(
			f"  Original: {original_count}, Forwarded: {forwarded_count}, main: {main_count}")
		logger.debug(f"⏱️ Time elapsed: {elapsed:.2f} seconds")

		df = pd.DataFrame(data)

		self.save_to_csv(df)

		df = df.drop_duplicates(
			subset=[
			"date",
			"from",
			"X-From",
			"X-To",
			"to",
			"subject",
			"cc",
			"X-cc",
			"body",
			],
			keep="first",
		)
		df = df.sort_values(by="date", ascending=True)

		self.save_to_csv(df)

		return df

	def split_all_messages(self, email_text: str):
		"""
		Split main message and all forwarded and original messages (preserve markers).
		"""
		forward_pattern = re.compile(r"-{2,}\s*Forwarded by\s+.*?-{2,}",
			re.DOTALL | re.MULTILINE | re.IGNORECASE)

		original_pattern = re.compile(r".*[-]+\s*Original Message\s*-+.*",
			re.MULTILINE | re.IGNORECASE)

		forward_matches = list(forward_pattern.finditer(email_text))
		original_matches = list(original_pattern.finditer(email_text))

		forward_parts = []
		original_parts = []
		main_text = email_text

		# --- forwarded ---
		if forward_matches:
			main_text = email_text[:forward_matches[0].start()]
			for i in range(len(forward_matches)):
				start = forward_matches[i].start()
				next_forward_start = (forward_matches[i + 1].start() if i +
					1 < len(forward_matches) else None)
				next_original_start = None
				for orig_match in original_matches:
					if orig_match.start() > start:
						next_original_start = orig_match.start()
						break
				candidates = [
					pos for pos in [next_forward_start, next_original_start] if pos is not None
				]
				end = min(candidates) if candidates else len(email_text)
				forward_parts.append(email_text[start:end])

			original_parts = []

		# --- original ---
		if original_matches:
			main_text = main_text[:original_matches[0].start()]
			for i in range(len(original_matches)):
				start = original_matches[i].start()
				end = (original_matches[i + 1].start() if i +
					1 < len(original_matches) else len(email_text))
				original_parts.append(email_text[start:end])

		return main_text.strip(), forward_parts, original_parts

	def extract_all_emails(self, email_text):
		"""Extracts main and forwarded emails as individual dictionaries."""
		extracted_emails = []

		main_text, forwards_raw, original_parts = self.split_all_messages(email_text)

		# Main email
		header_part, body = self.split_headers_body(main_text)
		main_data = self.parse_headers_main(header_part)
		main_data["body"] = self.clean_body(body)
		main_data["type"] = "main"
		extracted_emails.append(main_data)

		# original emails
		for org_text in original_parts:
			org_data = {
				"message_id": f"{main_data['message_id']}_original",
				"main_id": main_data["message_id"],
				"filename": None,
				"type": "original",
				"date": None,
				"from": None,
				"to": None,
				"subject": None,
				"cc": None,
				"body": None,
			}
			org_text = re.sub(r"\.{2,}", ".", org_text)
			parsed_org = self.parse_original_block(org_text)
			org_data["date"] = self.normalize_dates(next(iter(parsed_org.get("date", [])), None))
			org_data["from"] = parsed_org.get("original_sender")
			org_data["to"] = parsed_org.get("to")
			org_data["cc"] = parsed_org.get("cc")
			org_data["subject"] = parsed_org.get("subject")
			org_data["body"] = self.clean_body(parsed_org.get("body"))
			extracted_emails.append(org_data)

		# Forwarded emails
		for fwd_text in forwards_raw:
			fwd_data1 = {
				"message_id": f"{main_data['message_id']}_forwarded",
				"main_id": main_data["message_id"],
				"filename": None,
				"type": "forwarded",
				"date": None,
				"from": None,
				"original_sender": None,
				"original_Date": None,
				"to": None,
				"subject": None,
				"cc": None,
				"body": None,
			}
			parsed_fwd = self.parse_forwarded_block(fwd_text)
			fwd_data1["date"] = self.normalize_dates(next(iter(parsed_fwd.get("date", [])), None))
			fwd_data1["from"] = parsed_fwd.get("from")
			fwd_data1["original_sender"] = parsed_fwd.get("original_sender")
			fwd_data1["original_Date"] = parsed_fwd.get("original_Date")
			fwd_data1["to"] = parsed_fwd.get("to")
			fwd_data1["cc"] = parsed_fwd.get("cc")
			fwd_data1["subject"] = parsed_fwd.get("subject")
			fwd_data1["body"] = self.clean_body(parsed_fwd.get("body"))
			extracted_emails.append(fwd_data1)

		return extracted_emails

	def split_headers_body(self, email_text):
		"""Split headers and body. The body starts after a real blank line or a line containing only '>' and spaces."""
		lines = email_text.splitlines()
		headers = []
		body_lines = []
		found_split = False

		for line in lines:
			if not found_split:
				if re.match(r"^\s*$", line) or re.match(r"^\s*>+\s*$", line):
					found_split = True
					continue
				headers.append(line)
			else:
				body_lines.append(line)

		return "\n".join(headers), "\n".join(body_lines)

	def parse_headers_main(self, header_text):
		"""Parses the headers into a dictionary."""
		fields = {
			"message_id": set(),
			"filename": "",
			"date": set(),
			"from": set(),
			"to": set(),
			"subject": set(),
			"X-From": set(),
			"X-To": set(),
			"X-cc": set(),
		}

		for line in header_text.split("\n"):
			if line.startswith("Message-ID:"):
				fields["message_id"] = line.split(":", 1)[1].strip()
			elif line.startswith("Date:"):
				date_main = line.split(":", 1)[1].strip()
				fields["date"] = self.normalize_dates(date_main)
			elif line.startswith("From:"):
				fields["from"] = line.split(":", 1)[1].strip()
			elif line.startswith("To:"):
				fields["to"] = line.split(":", 1)[1].strip()
			elif line.startswith("Subject:"):
				fields["subject"] = line.split(":", 1)[1].strip()
			elif line.startswith("X-From:"):
				fields["X-From"] = line.split(":", 1)[1].strip()
			elif line.startswith("X-To:"):
				fields["X-To"] = line.split(":", 1)[1].strip()
			elif line.startswith("X-cc:"):
				fields["X-cc"] = line.split(":", 1)[1].strip()

		return fields

	def parse_original_block(self, original_block):
		org_data = {
			"date": set(),
			"from": set(),
			"original_sender": set(),
			"original_Date": set(),
			"to": set(),
			"cc": set(),
			"subject": set(),
			"body": set(),
		}

		# header and body
		header_body_split = re.split(r"\n\s*\n", original_block, maxsplit=1)
		header_text = header_body_split[0] if len(header_body_split) > 0 else ""

		# header
		headers = dict(
			re.findall(
			r"^>?\s*(From|To|Cc|Subject|Date|Sent):\s*(.+)",
			header_text,
			re.MULTILINE | re.IGNORECASE,
			))

		if "From" in headers:
			org_data["original_sender"].add(headers["From"].strip())

		if "To" in headers:
			org_data["to"].add(headers["To"].strip())

		if "Cc" in headers:
			org_data["cc"].add(headers["Cc"].strip())

		if "Subject" in headers:
			org_data["subject"].add(headers["Subject"].strip())

		if "Date" in headers:
			org_data["date"].add(headers["Date"].strip())

		if "Sent" in headers:
			org_data["date"].add(headers["Sent"].strip())

		# extract information
		found_emails = re.findall(r"[\w\.-]+@[\w\.-]+", original_block)
		for email in found_emails:
			if "to" in email.lower():
				org_data["to"].add(email)
			elif "cc" in email.lower():
				org_data["cc"].add(email)
			else:
				org_data["from"].add(email)

		# body after subject

		message_match = re.search(
			r"(?s)(?:Subject:.*?)\n\n(.*)",
			original_block,
		)

		if message_match:
			body_part = message_match.group(1).strip()
			org_data["body"].add(body_part)

			body_index = original_block.find(body_part)
		else:
			body_part = ""
			body_index = -1

		if body_index != -1:
			headers_only = original_block[:body_index].strip()
		else:
			headers_only = original_block

		# NLP
		doc = self.nlp(headers_only)
		date_entities = []
		time_entities = []

		for ent in doc.ents:
			if ent.label_ == "PERSON":
				org_data["from"].add(ent.text.strip())
			elif ent.label_ == "DATE":
				date_entities.append(ent.text.strip())
			elif ent.label_ == "TIME":
				time_entities.append(ent.text.strip())
			elif ent.label_ == "ORG":
				org_data["from"].add(ent.text.strip())

		# Combine date and time
		for i in range(max(len(date_entities), len(time_entities))):
			date = date_entities[i] if i < len(date_entities) else ""
			time = time_entities[i] if i < len(time_entities) else ""
			combined = f"{date} {time}".strip()
			org_data["date"].add(combined)

		return org_data

	def parse_forwarded_block(self, forward_block):
		"""Parses a single forwarded message block and extracts headers from the body."""
		fwd_data = {
			"date": set(),
			"from": set(),
			"original_sender": set(),
			"original_Date": set(),
			"to": set(),
			"cc": set(),
			"subject": set(),
			"body": set(),
		}

		# Extract information about the forwarder
		forward_info = re.search(
			r"-+ Forwarded by ([^/]+)(?:/[^ ]+)* on [\s\n]*((?:\d{2}|\d{4})[\s\n]*[-/][\s\n]*(?:\d{2}|\d{4})[\s\n]*[-/][\s\n]*(?:\d{2}|\d{4})[\s\n]*\d{2}:[\s\n]*\d{2}(?:[\s\n]*[AP]M)?)",
			forward_block,
		)
		if forward_info:
			fwd_data["date"].add(forward_info.group(2).replace("\n", " ").strip())
			from_strip = forward_info.group(1).strip()
			from_split = re.split(r"[;,]", from_strip)
			fwd_data["from"].update(from_split)

		# Extract original sender info (original sender and date)
		sender_info = re.search(
			r"(.+(?:<.+?>)?)\s*\n\s*(\d{2}[-/]\d{2}[-/](?:\d{2}|\d{4}) \d{1,2}:\d{2}(?: [AP]M)?)",
			forward_block,
		)
		if sender_info:
			fwd_data["original_sender"].add(sender_info.group(1).strip())
			fwd_data["date"].add(sender_info.group(2).strip())
		else:
			fwd_data["original_sender"].add("can not find")

		# Extract date
		date_match = re.search(
			r"Date:\s*(\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}[AP]M)",
			forward_block,
			re.IGNORECASE,
		)
		if date_match:
			fwd_data["date"].add(date_match.group(1).strip())

		# Extract recipient (To)
		to_info = re.search(r"To:\s*(.+)", forward_block)
		fwd_data["to"].add(to_info.group(1).strip() if to_info else "")

		# Extract To and Cc blocks and email addresses
		to_match = re.search(r"To:(.*?)(\n\s*\n|Subject:)", forward_block,
			re.DOTALL | re.IGNORECASE)
		cc_match = re.search(r"Cc:(.*?)(\n\s*\n|Subject:)", forward_block,
			re.DOTALL | re.IGNORECASE)

		# Extract email addresses
		to_block = to_match.group(1).strip() if to_match else ""
		cc_block = cc_match.group(1).strip() if cc_match else ""

		emails_to = re.findall(r"[\w\.-]+@[\w\.-]+", to_block)
		emails_cc = re.findall(r"[\w\.-]+@[\w\.-]+", cc_block)

		fwd_data["to"].update(emails_to)
		fwd_data["cc"].update(emails_cc)

		# Extract subject
		subject_info = re.search(r"Subject:\s*(.+)", forward_block)
		fwd_data["subject"].add(subject_info.group(1).strip()) if subject_info else None

		# Extract message body after the subject
		message_match = re.search(r"Subject:.*?\n\n(.*)", forward_block, re.DOTALL)

		if message_match:
			body_part = message_match.group(1).strip()
			fwd_data["body"].add(body_part)

			body_index = forward_block.find(body_part)
		else:
			body_part = ""
			body_index = -1

		if body_index != -1:
			headers_only = forward_block[:body_index].strip()
		else:
			headers_only = forward_block

		# NLP
		doc = self.nlp(headers_only)
		date_entities = []
		time_entities = []

		for ent in doc.ents:
			if ent.label_ == "PERSON":
				fwd_data["from"].add(ent.text.strip())
			elif ent.label_ == "DATE":
				date_entities.append(ent.text.strip())
			elif ent.label_ == "TIME":
				time_entities.append(ent.text.strip())
			elif ent.label_ == "ORG":
				fwd_data["from"].add(ent.text.strip())

		# Combine date and time
		for i in range(max(len(date_entities), len(time_entities))):
			date = date_entities[i] if i < len(date_entities) else ""
			time = time_entities[i] if i < len(time_entities) else ""
			combined = f"{date} {time}".strip()
			fwd_data["date"].add(combined)

		return fwd_data

	def clean_body(self, body):
		"""Clean body text (remove reply chains, extra spaces, etc)."""

		result = []
		for char, group in groupby(body):
			if char in string.punctuation:
				result.append(char)
			else:
				result.extend(group)
		body = "".join(result)

		if not re.search(r"[a-zA-Z]", body):
			body = ""

		body = clean(
			body,
			fix_unicode=True,  # fix various unicode errors
			to_ascii=True,  # transliterate to closest ASCII representation
			no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
			lang="en",
		)

		body = re.sub(r"\*{5,}.*?\*{5,}", "", body, flags=re.DOTALL)

		# Remove lines starting with "to:", "cc:", "subject:", etc.
		body = re.sub(
			r"^(to:|cc:|subject:|from:|copy:|re:).*\n?",
			"",
			body,
			flags=re.MULTILINE | re.IGNORECASE,
		)

		# Remove any remaining \n characters.
		body = re.sub(r"\s*\n\s*", " ", body)

		# Decode HTML entities like &nbsp;, &gt;, etc.
		body = html.unescape(body)

		# Remove HTML tags (e.g., <div>, <font>, <br>, etc.)
		body = BeautifulSoup(body, "html.parser").get_text(separator=" ")

		# Remove repeating punctuation like ".....", "???", "!!!", "---", etc.
		body = re.sub(r"([!?.,;:\-])\1{1,}", r"\1", body)

		# 4. Remove leftover special characters like "�" or weird unicode
		body = re.sub(r"[^\x00-\x7F]+", " ", body)

		# Normalize whitespace (multiple spaces → one space)
		body = re.sub(r"\s+", " ", body).strip()

		return body

	def normalize_dates(self, text):
		european_date = None
		if text is None:
			return ""
		cleaned_text = re.sub(r"\s*\([A-Z]{2,}\)$", "", text.strip())
		parsed_date = dateparser.parse(cleaned_text)
		if parsed_date:
			european_date = parsed_date.strftime("%d/%m/%Y %H:%M:%S")
			# text = text.replace(text, european_date)
		return european_date

	def save_to_json(self, df):
		"""Save a Pandas DataFrame to a JSON file."""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.json")
		df.to_json(
			output_path,
			orient="records",
			indent=2,
			force_ascii=False,
		)
		logger.success(f"\n✅ Saved {len(df)} rows to {output_path}")

	def save_to_pickle(self, df: pd.DataFrame):
		"""Save a list of cleaned email dataframe to a pickle file."""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.pkl")
		df.to_pickle(output_path)
		logger.success(f"\n✅ Saved {len(df)} cleaned emails to {output_path}")

	def save_to_csv(self, df: pd.DataFrame):
		"""Save a Pandas DataFrame to a CSV file."""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.csv")
		df.to_csv(output_path, )
		logger.success(f"\n✅ Saved {len(df)} rows to {output_path}")

	def load_data(self):
		df = load_processed_df(self.output_dir, "processed_data_")
		return df

	def save_to_database(self, df: pd.DataFrame, db: DatabaseManager):
		"""Save a Pandas DataFrame to a database table."""
		if df is None or df.empty:
			logger.warning("DataFrame is empty. Skipping database save.")
			return

		if db is None:
			logger.error("Database connection is not provided. Skipping database save.")
			return

		try:
			table_name = "processed_emails"
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
			]
			db.insert_from_dataframe(df, table_name, columns=columns)
			logger.success(f"✅ Successfully saved {len(df)} rows to database table '{table_name}'")
		except Exception as e:
			logger.error(f"Error saving DataFrame to database: {e}")


if __name__ == "__main__":
	# Create a DataPreparation instance
	data_prep = DataPreparation()

	data = data_prep.process_all_emails()
	data_prep.save_to_json(data)
	data_prep.save_to_pickle(data)

	logger.success("Done!")
