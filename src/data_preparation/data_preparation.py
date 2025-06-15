# import concurrent.futures
# import os
# import re
# import time
# from datetime import datetime
# from email import message_from_string
# from email.policy import default
# from email.utils import parsedate_to_datetime
# from typing import Dict, List, Optional, Any
#
# import dateparser
# import pandas as pd
# from loguru import logger
# from tqdm import tqdm
# from utils import save_to_postgresql, load_processed_df
#
#
# class DataPreparation:
#     """
#     Class responsible for data preparation and storage tasks:
#     - Loading email data
#     - Cleaning and preprocessing text
#     - Extracting metadata (sender, recipient, date, etc.)
#     - Storing processed data
#     """
#
#     def __init__(self, input_dir="./data/", output_dir="./processed_data/", save_to_db=False, max_workers=None,
#                  skip=False):
#         """
#         Initialize the DataPreparation class.
#
#         Args:
#             input_dir (str): Directory containing the email files
#             output_dir (str): Directory to store processed data
#             save_to_db (bool): Whether to save processed data to PostgreSQL database
#             max_workers (int): Maximum number of worker processes for parallel processing
#             skip (bool): Whether to skip data preparation and load existing data from any possible resource
#         """
#         self.input_dir = input_dir
#         self.output_dir = output_dir
#         self.save_to_db = save_to_db
#         self.max_workers = max_workers or os.cpu_count() or 1
#         self.skip = skip
#
#         self.df = pd.DataFrame(columns=[
#             "message_id",
#             "original_message_id",
#             "main_id",
#             "filename",
#             "type",
#             "date",
#             "from",
#             "X-From",
#             "X-To",
#             "original_sender",
#             "original_Date",
#             "to",
#             "subject",
#             "cc",
#             "X-cc",
#             "body",
#         ])
#
#         # Create an output directory if it doesn't exist
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#     @staticmethod
#     def normalize_dates(text):
#         if text is None:
#             return None
#         cleaned_text = re.sub(r"\s*\([A-Z]{2,}\)$", "", text.strip())
#         parsed_date = dateparser.parse(cleaned_text)
#         if parsed_date:
#             european_date = parsed_date.strftime("%d/%m/%Y %H:%M:%S")
#             text = text.replace(text, european_date)
#         return text
#
#     @staticmethod
#     def split_headers_body(email_text):
#         """Splits raw email into headers and body."""  # docstring
#         parts = email_text.split("\n\n", 1)
#         headers = parts[0]
#         body = parts[1] if len(parts) > 1 else ""
#         return headers, body
#
#     def parse_headers_org(self, header_text, main_id):
#         """Parses the original headers into a dictionary."""
#         fields = {
#             "original_message_id": main_id + "original",  # we create
#             "main_id": main_id,
#             "filename": "",
#             "date": set(),
#             "from": set(),
#             "to": set(),
#             "subject": set(),
#             "cc": set(),
#         }
#
#         # First try to parse the "Original Message" format
#         original_msg_pattern = re.compile(
#             r"Original Message From: (.*?)(?:\n|$)"
#             r"(?:Sent: (.*?)(?:\n|$))?"
#             r"(?:To: (.*?)(?:\n|$))?"
#             r"(?:Cc: (.*?)(?:\n|$))?"
#             r"(?:Subject: (.*?)(?:\n|$))?",
#             re.IGNORECASE | re.MULTILINE
#         )
#
#         match = original_msg_pattern.search(header_text)
#         if match:
#             # Extract fields from the match
#             from_text = match.group(1).strip() if match.group(1) else ""
#             sent_text = match.group(2).strip() if match.group(2) else ""
#             to_text = match.group(3).strip() if match.group(3) else ""
#             cc_text = match.group(4).strip() if match.group(4) else ""
#             subject_text = match.group(5).strip() if match.group(5) else ""
#
#             # Add to fields
#             if from_text:
#                 fields["from"].add(from_text)
#             if sent_text:
#                 fields["date"].add(self.normalize_dates(sent_text))
#             if to_text:
#                 fields["to"].add(to_text)
#             if cc_text:
#                 fields["cc"].add(cc_text)
#             if subject_text:
#                 fields["subject"].add(subject_text)
#
#         # If no match found or fields are empty, try the standard header parsing
#         if not any(fields.values()):
#             for line in header_text.split("\n"):
#                 line = line.strip()  # Remove leading/trailing whitespace
#                 if not line:  # Skip empty lines
#                     continue
#                 if line.startswith("From:"):
#                     fields["from"].add(line.split(":", 1)[1].strip())
#                 elif line.startswith("Sent:"):
#                     date_text = line.split(":", 1)[1].strip()
#                     normalized_date = self.normalize_dates(date_text)
#                     fields["date"].add(normalized_date)
#                 elif line.startswith("To:") and not fields["to"]:  # First "To:"
#                     fields["to"].add(line.split(":", 1)[1].strip())
#                 elif line.startswith("Cc:"):
#                     fields["cc"].add(line.split(":", 1)[1].strip())
#                 elif line.startswith("Subject:"):
#                     fields["subject"].add(line.split(":", 1)[1].strip())
#
#         return fields
#
#     def parse_headers_main(self, header_text):
#         """Parses the headers into a dictionary."""
#         fields = {
#             "message_id": set(),
#             "filename": "",
#             "date": set(),
#             "from": set(),
#             "to": set(),
#             "subject": set(),
#             "X-From": set(),
#             "X-To": set(),
#             "X-cc": set(),
#         }
#
#         for line in header_text.split("\n"):
#             if line.startswith("Message-ID:"):
#                 fields["message_id"].add(line.split(":", 1)[1].strip())
#             elif line.startswith("Date:"):
#                 date_main = line.split(":", 1)[1].strip()
#                 fields["date"].add(self.normalize_dates(date_main))
#             elif line.startswith("From:"):
#                 fields["from"].add(line.split(":", 1)[1].strip())
#             elif line.startswith("To:"):
#                 fields["to"].add(line.split(":", 1)[1].strip())
#             elif line.startswith("Subject:"):
#                 fields["subject"].add(line.split(":", 1)[1].strip())
#             elif line.startswith("X-From:"):
#                 fields["X-From"].add(line.split(":", 1)[1].strip())
#             elif line.startswith("X-To:"):
#                 fields["X-To"].add(line.split(":", 1)[1].strip())
#             elif line.startswith("X-cc:"):
#                 fields["X-cc"].add(line.split(":", 1)[1].strip())
#
#         return fields
#
#     @staticmethod
#     def parse_forwarded_block(forward_block):
#         """Parses a single forwarded message block and extracts headers from the body."""
#         fwd_data = {
#             "date": set(),
#             "from": set(),
#             "original_sender": set(),
#             "original_Date": set(),
#             "to": set(),
#             "cc": set(),
#             "subject": set(),
#             "body": set(),
#         }
#
#         # Extract information about the forwarder
#         forward_info = re.search(
#             r"-+ Forwarded by ([^/]+)(?:/[^ ]+)* on [\s\n]*((?:\d{2}|\d{4})[\s\n]*[-/][\s\n]*(?:\d{2}|\d{4})[\s\n]*[-/][\s\n]*(?:\d{2}|\d{4})[\s\n]*\d{2}:[\s\n]*\d{2}(?:[\s\n]*[AP]M)?)",
#             forward_block,
#         )
#         if forward_info:
#             fwd_data["date"].add(forward_info.group(2).replace("\n", " ").strip())
#             fwd_data["from"].add(forward_info.group(1).strip())
#         # else:
#         # fwd_data["from"] = "can not find"
#
#         # Extract original sender info (original sender and date)
#         sender_info = re.search(
#             r"(.+(?:<.+?>)?)\s*\n\s*(\d{2}[-/]\d{2}[-/](?:\d{2}|\d{4}) \d{1,2}:\d{2}(?: [AP]M)?)",
#             forward_block,
#         )
#         if sender_info:
#             fwd_data["original_sender"].add(sender_info.group(1).strip())
#             fwd_data["original_Date"].add(sender_info.group(2).strip())
#         else:
#             fwd_data["original_sender"].add("can not find")
#
#         # Extract recipient (To)
#         to_info = re.search(r"To:\s*(.+)", forward_block)
#         fwd_data["to"].add(to_info.group(1).strip() if to_info else "")
#
#         # Extract To and Cc blocks and email addresses
#         to_match = re.search(
#             r"To:(.*?)(\n\s*\n|Cc:|Subject:)", forward_block, re.DOTALL | re.IGNORECASE
#         )
#         cc_match = re.search(
#             r"Cc:(.*?)(\n\s*\n|Subject:)", forward_block, re.DOTALL | re.IGNORECASE
#         )
#
#         # Extract email addresses
#         to_block = to_match.group(1).strip() if to_match else ""
#         cc_block = cc_match.group(1).strip() if cc_match else ""
#
#         emails_to = re.findall(r"[\w\.-]+@[\w\.-]+", to_block)
#         emails_cc = re.findall(r"[\w\.-]+@[\w\.-]+", cc_block)
#
#         fwd_data["to"].update(emails_to)
#         fwd_data["cc"].update(emails_cc)
#
#         # Extract subject
#         subject_info = re.search(r"Subject:\s*(.+)", forward_block)
#         fwd_data["subject"].add(subject_info.group(1).strip()) if subject_info else None
#
#         # Extract the message body after the subject
#         message_match = re.search(r"Subject:.*?\n\n(.*)", forward_block, re.DOTALL)
#         fwd_data["body"].add(message_match.group(1).strip()) if message_match else None
#
#         return fwd_data
#
#     @staticmethod
#     def clean_body(body):
#         """Clean body text (remove reply chains, extra spaces, etc.)."""
#
#         # Remove lines starting with "to:", "cc:", "subject:", etc.
#         body = re.sub(
#             r"^(to:|cc:|subject:|from:|copy:|re:).*\n?",
#             "",
#             body,
#             flags=re.MULTILINE | re.IGNORECASE,
#         )
#
#         # Remove strange repeating symbols like ?????, ...., etc.
#         body = re.sub(r"(\s*[?\.!;:-])\1+", " ", body)
#         # body = re.sub(r'(\s*[?\.!;:-]]\s*){2,}', ' ', body)
#
#         # Normalize whitespace (replace multiple spaces and newlines with a single space)
#         body = re.sub(r"\s+", " ", body)
#
#         # Remove any remaining \n characters.
#         # body = re.sub(r'(\s*[\n]\s*){2,}', ' ', body)
#         body = re.sub(r"\s*\n\s*", " ", body)
#
#         # Remove extra spaces at the start and end
#         body = body.strip()
#
#         return body
#
#     @staticmethod
#     def split_all_messages(email_text: str) -> tuple:
#         """Split the main message and all forwarded and original messages."""
#         # Updated patterns to handle more email formats
#         forward_pattern = re.compile(r"^[-]+\s*Forwarded by.*|^[-]+\s*Forwarded message.*",
#                                      re.MULTILINE | re.IGNORECASE)
#         original_pattern = re.compile(r"^[-]+\s*Original Message.*|^[-]+\s*Original message.*",
#                                       re.MULTILINE | re.IGNORECASE)
#
#         forward_matches = list(forward_pattern.finditer(email_text))
#         original_matches = list(original_pattern.finditer(email_text))
#
#         forward_parts = []
#         original_parts = []
#         main_text = email_text
#
#         def filter_list(matches, context_list):
#             if not matches:
#                 return main_text
#             _main_text = email_text[: matches[0].start()]
#             for i in range(len(matches)):
#                 start = matches[i].start()
#                 end = matches[i + 1].start() if i + 1 < len(matches) else len(email_text)
#                 context_list.append(email_text[start:end])
#             return _main_text
#
#         if forward_matches:
#             main_text = filter_list(forward_matches, forward_parts)
#         if original_matches:
#             main_text = filter_list(original_matches, original_parts)
#
#         return main_text.strip(), forward_parts, original_parts
#
#     def extract_all_emails(self, email_text):
#         """Extracts main and forwarded emails as individual dictionaries."""
#         extracted_emails = []
#
#         main_text, forwards_raw, original_parts = self.split_all_messages(email_text)
#
#         # Main email
#         header_part, body = self.split_headers_body(main_text)
#         main_data = self.parse_headers_main(header_part)
#         main_data["body"] = self.clean_body(body)
#         main_data["type"] = "main"
#         extracted_emails.append(main_data)
#
#         # original
#         for org_text in original_parts:
#             header_part_original, body_original = self.split_headers_body(org_text)
#             main_id = next(iter(main_data["message_id"]))
#             original_data = self.parse_headers_org(header_part_original, main_id)
#             original_data["body"] = self.clean_body(body_original)
#             original_data["type"] = "original"
#             extracted_emails.append(original_data)
#
#         # Forwarded emails
#         for fwd_text in forwards_raw:
#             fwd_data1 = {
#                 "message_id": next(iter(main_data["message_id"])) + "forwarded",
#                 "original_message_id": main_data["message_id"],
#                 "filename": None,
#                 "type": "forwarded",
#                 "date": None,
#                 "from": None,
#                 "original_sender": None,
#                 "original_Date": None,
#                 "to": None,
#                 "subject": None,
#                 "cc": None,
#                 "body": None,
#             }
#             parsed_fwd = self.parse_forwarded_block(fwd_text)
#             fwd_data1["date"] = self.normalize_dates(next(iter(parsed_fwd.get("date", [])), None))
#             fwd_data1["from"] = parsed_fwd.get("from")
#             fwd_data1["original_sender"] = parsed_fwd.get("original_sender")
#             fwd_data1["original_Date"] = parsed_fwd.get("original_Date")
#             fwd_data1["to"] = parsed_fwd.get("to")
#             fwd_data1["cc"] = parsed_fwd.get("cc")
#             fwd_data1["subject"] = parsed_fwd.get("subject")
#             fwd_data1["body"] = parsed_fwd.get("body")
#             extracted_emails.append(fwd_data1)
#
#         return extracted_emails
#
#     def _process_single_file(self, file_path: str) -> List[Dict[str, Any]]:
#         """Process a single email file and return extracted data."""
#         try:
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 raw_text = f.read()
#
#             # Parse the main email
#             main_data = EmailParser.parse_email_content(raw_text)
#             if not main_data:
#                 return []
#
#             main_data['filename'] = os.path.basename(file_path)
#             main_data['type'] = 'main'
#             main_data['date'] = EmailParser.normalize_date(main_data['date'])
#             main_data['body'] = EmailParser.clean_body(main_data['body'])
#
#             # Ensure main_id is set for the main message
#             main_data['main_id'] = main_data['message_id']
#
#             # Extract forwarded and original messages
#             extracted_emails = [main_data]
#
#             # Split into main, forwarded, and original parts
#             main_text, forwards_raw, original_parts = self.split_all_messages(raw_text)
#
#             # Process original messages
#             for org_text in original_parts:
#                 original_data = EmailParser.parse_email_content(org_text)
#                 if original_data:
#                     original_data['filename'] = os.path.basename(file_path)
#                     original_data['type'] = 'original'
#                     original_data['message_id'] = f"{main_data['message_id']}_original"
#                     original_data['main_id'] = main_data['message_id']  # Set main_id to main message's ID
#                     original_data['date'] = EmailParser.normalize_date(original_data['date'])
#                     original_data['body'] = EmailParser.clean_body(original_data['body'])
#                     extracted_emails.append(original_data)
#
#             # Process forwarded messages
#             for fwd_text in forwards_raw:
#                 fwd_data = EmailParser.parse_email_content(fwd_text)
#                 if fwd_data:
#                     fwd_data['filename'] = os.path.basename(file_path)
#                     fwd_data['type'] = 'forwarded'
#                     fwd_data['message_id'] = f"{main_data['message_id']}_forwarded"
#                     fwd_data['main_id'] = main_data['message_id']  # Set main_id to main message's ID
#                     fwd_data['date'] = EmailParser.normalize_date(fwd_data['date'])
#                     fwd_data['body'] = EmailParser.clean_body(fwd_data['body'])
#                     extracted_emails.append(fwd_data)
#
#             return extracted_emails
#
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {e}")
#             return []
#
#     def process_all_emails(self) -> pd.DataFrame:
#         """
#         Process all email files in parallel and return a DataFrame with the results.
#         """
#         logger.info(f"📂 Looking for files in: {os.path.abspath(self.input_dir)}")
#
#         # Get all file paths
#         file_paths = []
#         for root, _, files in os.walk(self.input_dir):
#             for filename in files:
#                 file_paths.append(os.path.join(root, filename))
#
#         total_files = 1 or len(file_paths)
#         logger.info(f"Found {total_files} files to process")
#
#         # Process files in parallel
#         all_emails = []
#         start_time = time.time()
#
#         with tqdm(total=total_files, desc="📬 Processing emails") as pbar:
#             with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
#                 # Process files in chunks to avoid memory issues
#                 chunk_size = 1000
#                 for i in range(0, total_files, chunk_size):
#                     chunk = file_paths[i:i + chunk_size]
#                     futures = [executor.submit(self._process_single_file, file_path) for file_path in chunk]
#
#                     for future in concurrent.futures.as_completed(futures):
#                         emails = future.result()
#                         all_emails.extend(emails)
#                     pbar.update(1)
#
#                     # Update progress bar with counts
#                     main_count = sum(1 for e in all_emails if e.get('type') == 'main')
#                     original_count = sum(1 for e in all_emails if e.get('type') == 'original')
#                     forwarded_count = sum(1 for e in all_emails if e.get('type') == 'forwarded')
#                     pbar.set_postfix({
#                         'Total': len(all_emails),
#                         'Original': original_count,
#                         'Forwarded': forwarded_count,
#                         'Main': main_count
#                     })
#
#         end_time = time.time()
#         elapsed = end_time - start_time
#
#         # Convert to DataFrame
#         df = pd.DataFrame(all_emails)
#
#         # Log statistics
#         logger.success(f"\n✅ Done! Total emails: {len(df)}")
#         logger.info(f"Original: {len(df[df['type'] == 'original'])}, "
#                     f"Forwarded: {len(df[df['type'] == 'forwarded'])}, "
#                     f"Main: {len(df[df['type'] == 'main'])}")
#         logger.info(f"⏱️ Time elapsed: {elapsed:.2f} seconds")
#
#         # Save to PostgreSQL if requested
#         if self.save_to_db and not df.empty:
#             try:
#                 logger.info("Saving processed emails to PostgreSQL database...")
#                 table_name = "processed_emails"
#                 success_message = f"\n✅ Saved {len(df)} rows to PostgreSQL table: {table_name}"
#                 save_to_postgresql(df, table_name, if_exists="replace", success_message=success_message)
#             except Exception as e:
#                 logger.error(f"Failed to save to PostgreSQL: {e}")
#                 logger.info("Continuing without saving to database.")
#
#         return df
#
#     def save_to_json(self, df: pd.DataFrame) -> None:
#         """Save a Pandas DataFrame to a JSON file."""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}_v1.json")
#         df.to_json(output_path, orient="records", indent=2, force_ascii=False)
#         logger.success(f"\n✅ Saved {len(df)} rows to {output_path}")
#
#     def save_to_pickle(self, df: pd.DataFrame) -> None:
#         """Save a DataFrame to a pickle file."""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}_v1.pkl")
#         df.to_pickle(output_path)
#         logger.success(f"\n✅ Saved {len(df)} rows to {output_path}")
#
#     def load_data(self) -> pd.DataFrame:
#         """
#         Load the processed data from a pickle file.
#         """
#
#         df = load_processed_df(self.output_dir,"processed_data_")
#         return df
#
#
# class EmailParser:
#     """Utility class for parsing email content using the email library."""
#
#     @staticmethod
#     def _parse_x400_address(address: str) -> Dict[str, str]:
#         """Parse X.400 format address into its components."""
#         result = {
#             'full_name': '',
#             'organization': '',
#             'org_unit': '',
#             'common_name': '',
#             'email': '',
#             'display_name': ''
#         }
#
#         # Extract organization
#         org_match = re.search(r'O=([^\/]+)', address)
#         if org_match:
#             result['organization'] = org_match.group(1)
#
#         # Extract organizational unit
#         ou_match = re.search(r'OU=([^\/]+)', address)
#         if ou_match:
#             result['org_unit'] = ou_match.group(1)
#
#         # Extract common name
#         cn_match = re.search(r'CN=([^\/]+)', address)
#         if cn_match:
#             result['common_name'] = cn_match.group(1)
#
#         # Try to extract email if present
#         email_match = re.search(r'[\w\.-]+@[\w\.-]+', address)
#         if email_match:
#             result['email'] = email_match.group(0)
#
#         # Try to extract full name from the address
#         name_match = re.search(r'([^<]+)(?=<)', address)
#         if name_match:
#             result['full_name'] = name_match.group(1).strip()
#
#         # Create display name
#         display_parts = []
#         if result['full_name']:
#             display_parts.append(result['full_name'])
#         elif result['common_name'] and result['common_name'].upper() != 'RECIPIENTS':
#             display_parts.append(result['common_name'])
#
#         if result['organization'] and result['organization'].upper() != 'ENRON':
#             display_parts.append(f"({result['organization']})")
#
#         if result['email']:
#             display_parts.append(f"<{result['email']}>")
#
#         result['display_name'] = ' '.join(display_parts) if display_parts else ''
#
#         return result
#
#     @staticmethod
#     def _clean_email_address(address: str) -> str:
#         """Clean and normalize email addresses."""
#         if not address:
#             return ""
#
#         # Check if it's an X.400 format address
#         if '/O=' in address or '/OU=' in address or '/CN=' in address:
#             x400_data = EmailParser._parse_x400_address(address)
#             return x400_data['display_name'] or x400_data['full_name'] or x400_data['common_name']
#
#         # Handle regular email addresses
#         # Remove angle brackets and their contents if they don't contain an email
#         if not re.search(r'@', address):
#             address = re.sub(r'<.*?>', '', address)
#
#         # Extract email addresses
#         emails = re.findall(r'[\w\.-]+@[\w\.-]+', address)
#         if emails:
#             # If we found emails, return them
#             return ', '.join(emails)
#
#         # If no email found, clean the name
#         name = re.sub(r'[\\\/]', '', address)  # Remove backslashes and forward slashes
#         name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
#         return name.strip()
#
#     @staticmethod
#     def parse_email_content(email_text: str) -> Dict[str, Any]:
#         """Parse email content using the email library."""
#         try:
#             msg = message_from_string(email_text, policy=default)
#
#             # Parse addresses into structured format
#             from_data = EmailParser._parse_x400_address(msg.get('From', ''))
#             to_data = EmailParser._parse_x400_address(msg.get('To', ''))
#             cc_data = EmailParser._parse_x400_address(msg.get('Cc', ''))
#             x_from_data = EmailParser._parse_x400_address(msg.get('X-From', ''))
#             x_to_data = EmailParser._parse_x400_address(msg.get('X-To', ''))
#             x_cc_data = EmailParser._parse_x400_address(msg.get('X-cc', ''))
#
#             return {
#                 'message_id': msg.get('Message-ID', ''),
#                 'date': msg.get('Date', ''),
#                 'from': from_data['display_name'] or from_data['full_name'] or from_data['common_name'],
#                 'to': to_data['display_name'] or to_data['full_name'] or to_data['common_name'],
#                 'cc': cc_data['display_name'] or cc_data['full_name'] or cc_data['common_name'],
#                 'subject': EmailParser._clean_subject(msg.get('Subject', '')),
#                 'body': EmailParser._get_email_body(msg),
#                 'x_from': x_from_data['display_name'] or x_from_data['full_name'] or x_from_data['common_name'],
#                 'x_to': x_to_data['display_name'] or x_to_data['full_name'] or x_to_data['common_name'],
#                 'x_cc': x_cc_data['display_name'] or x_cc_data['full_name'] or x_cc_data['common_name'],
#             }
#         except Exception as e:
#             logger.error(f"Error parsing email: {e}")
#             return {}
#
#     @staticmethod
#     def _clean_subject(subject: str) -> str:
#         """Clean email subject line."""
#         if not subject:
#             return ""
#
#         # Remove common prefixes
#         subject = re.sub(r'^(re|fwd|fw):\s*', '', subject, flags=re.IGNORECASE)
#
#         # Remove any remaining escape characters
#         subject = re.sub(r'\\[a-zA-Z]', '', subject)
#
#         # Remove multiple spaces and normalize whitespace
#         subject = re.sub(r'\s+', ' ', subject)
#
#         return subject.strip()
#
#     @staticmethod
#     def _get_email_body(msg) -> str:
#         """Extract and clean email body from message object."""
#         body = []
#         if msg.is_multipart():
#             for part in msg.walk():
#                 if part.get_content_type() == "text/plain":
#                     try:
#                         content = part.get_content()
#                         if content:
#                             body.append(content)
#                     except:
#                         continue
#         else:
#             try:
#                 content = msg.get_content()
#                 if content:
#                     body.append(content)
#             except:
#                 pass
#
#         return EmailParser.clean_body("\n".join(body))
#
#     @staticmethod
#     def clean_body(body: str) -> str:
#         """Clean email body text."""
#         if not body:
#             return ""
#
#         # Remove common email markers and quoted text
#         body = re.sub(r'^>.*$', '', body, flags=re.MULTILINE)
#         body = re.sub(r'^(to:|cc:|subject:|from:|copy:|re:).*\n?', '', body, flags=re.MULTILINE | re.IGNORECASE)
#
#         # Remove separator lines
#         body = re.sub(r'={3,}|-{3,}|\*{3,}', '', body)
#
#         # Remove escape characters and their sequences
#         body = re.sub(r'\\[a-zA-Z]', '', body)
#         body = re.sub(r'\\/', '', body)
#
#         # Remove repeating symbols
#         body = re.sub(r'(\s*[?\.!;:-])\1+', ' ', body)
#
#         # Remove multiple newlines and normalize whitespace
#         body = re.sub(r'\n{3,}', '\n\n', body)
#         body = re.sub(r'\s+', ' ', body)
#
#         # Remove any remaining HTML-like tags
#         body = re.sub(r'<[^>]+>', '', body)
#
#         # Remove any remaining special characters
#         body = re.sub(r'[^\w\s.,!?-]', '', body)
#
#         return body.strip()
#
#     @staticmethod
#     def normalize_date(date_str: str) -> Optional[str]:
#         """Normalize date string to a standard format."""
#         if not date_str:
#             return None
#         try:
#             dt = parsedate_to_datetime(date_str)
#             return dt.strftime("%d/%m/%Y %H:%M:%S")
#         except:
#             try:
#                 dt = dateparser.parse(date_str)
#                 return dt.strftime("%d/%m/%Y %H:%M:%S") if dt else None
#             except:
#                 return None
#
#
# if __name__ == "__main__":
#     # Create a DataPreparation instance
#     data_prep = DataPreparation()
#
#     data_df = data_prep.process_all_emails()
#     data_prep.save_to_json(data_df)
#     data_prep.save_to_pickle(data_df)
#
#     logger.info("Data Preparation Finished")

# from loguru import logger
import os
import re
import time
from datetime import datetime

import dateparser
import pandas as pd
from tqdm import tqdm

from utils import load_processed_df


class DataPreparation:
    """
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

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def normalize_dates(self, text):
        if text is None:
            return None
        cleaned_text = re.sub(r"\s*\([A-Z]{2,}\)$", "", text.strip())
        parsed_date = dateparser.parse(cleaned_text)
        if parsed_date:
            european_date = parsed_date.strftime("%d/%m/%Y %H:%M:%S")
            text = text.replace(text, european_date)
        return text

    def split_headers_body(self, email_text):
        """Splits raw email into headers and body."""  # docstring
        parts = email_text.split("\n\n", 1)
        headers = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        return headers, body

    def parse_headers_org(self, header_text, main_id):
        """Parses the originals headers into a dictionary."""
        fields = {
            "original_message_id": main_id + "original",  # we create
            "main_id": main_id,
            "filename": "",
            "date": set(),
            "from": set(),
            "to": set(),
            "subject": set(),
            "cc": set(),
        }

        for line in header_text.split("\n"):
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                continue
            if line.startswith("From:"):
                fields["from"].add(line.split(":", 1)[1].strip())
            elif line.startswith("Sent:"):
                date_text = line.split(":", 1)[1].strip()
                normalized_date = self.normalize_dates(date_text)
                fields["date"].add(normalized_date)

            elif line.startswith("To:") and not fields["to"]:  # First "To:"
                fields["to"].add(line.split(":", 1)[1].strip())
            elif line.startswith("Cc:"):
                fields["cc"].add(line.split(":", 1)[1].strip())
            elif line.startswith("Subject:"):
                fields["subject"].add(line.split(":", 1)[1].strip())

        return fields

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
                fields["message_id"].add(line.split(":", 1)[1].strip())
            elif line.startswith("Date:"):
                date_main = line.split(":", 1)[1].strip()
                fields["date"].add(self.normalize_dates(date_main))
            elif line.startswith("From:"):
                fields["from"].add(line.split(":", 1)[1].strip())
            elif line.startswith("To:"):
                fields["to"].add(line.split(":", 1)[1].strip())
            elif line.startswith("Subject:"):
                fields["subject"].add(line.split(":", 1)[1].strip())
            elif line.startswith("X-From:"):
                fields["X-From"].add(line.split(":", 1)[1].strip())
            elif line.startswith("X-To:"):
                fields["X-To"].add(line.split(":", 1)[1].strip())
            elif line.startswith("X-cc:"):
                fields["X-cc"].add(line.split(":", 1)[1].strip())

        return fields

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
        body_lines = []
        header_found = False

        # Extract information about the forwarder
        forward_info = re.search(
            r"-+ Forwarded by ([^/]+)(?:/[^ ]+)* on [\s\n]*((?:\d{2}|\d{4})[\s\n]*[-/][\s\n]*(?:\d{2}|\d{4})[\s\n]*[-/][\s\n]*(?:\d{2}|\d{4})[\s\n]*\d{2}:[\s\n]*\d{2}(?:[\s\n]*[AP]M)?)",
            forward_block,
        )
        if forward_info:
            fwd_data["date"].add(forward_info.group(2).replace("\n", " ").strip())
            fwd_data["from"].add(forward_info.group(1).strip())
        # else:
        # fwd_data["from"] = "can not find"

        # Extract original sender info (original sender and date)
        sender_info = re.search(
            r"(.+(?:<.+?>)?)\s*\n\s*(\d{2}[-/]\d{2}[-/](?:\d{2}|\d{4}) \d{1,2}:\d{2}(?: [AP]M)?)",
            forward_block,
        )
        if sender_info:
            fwd_data["original_sender"].add(sender_info.group(1).strip())
            fwd_data["original_Date"].add(sender_info.group(2).strip())
        else:
            fwd_data["original_sender"].add("can not find")

        # Extract recipient (To)
        to_info = re.search(r"To:\s*(.+)", forward_block)
        fwd_data["to"].add(to_info.group(1).strip() if to_info else "")

        # Extract To and Cc blocks and email addresses
        to_match = re.search(
            r"To:(.*?)(\n\s*\n|Cc:|Subject:)", forward_block, re.DOTALL | re.IGNORECASE
        )
        cc_match = re.search(
            r"Cc:(.*?)(\n\s*\n|Subject:)", forward_block, re.DOTALL | re.IGNORECASE
        )

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
        fwd_data["body"].add(message_match.group(1).strip()) if message_match else None

        return fwd_data

    def clean_body(self, body):
        """Clean body text (remove reply chains, extra spaces, etc)."""

        # Remove lines starting with "to:", "cc:", "subject:", etc.
        body = re.sub(
            r"^(to:|cc:|subject:|from:|copy:|re:).*\n?",
            "",
            body,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Remove strange repeating symbols like ?????, ...., etc.
        body = re.sub(r"(\s*[?\.!;:-])\1+", " ", body)
        # body = re.sub(r'(\s*[?\.!;:-]]\s*){2,}', ' ', body)

        # Normalize whitespace (replace multiple spaces and newlines with a single space)
        body = re.sub(r"\s+", " ", body)

        # Remove any remaining \n characters.
        # body = re.sub(r'(\s*[\n]\s*){2,}', ' ', body)
        body = re.sub(r"\s*\n\s*", " ", body)

        # Remove extra spaces at the start and end
        body = body.strip()

        return body

    def split_all_messages(self, email_text):
        """Split main message and all forwarded and original messages (preserve markers)."""
        # regex for forwarded and originals
        forward_pattern = re.compile(r"^[-]+\s*Forwarded by.*", re.MULTILINE | re.IGNORECASE)
        original_pattern = re.compile(
            r"^\s*[-]+\s*Original Message.*", re.MULTILINE | re.IGNORECASE
        )

        forward_matches = list(
            forward_pattern.finditer(email_text)
        )  # find all non-overlapping occurrences of a pattern within the string
        original_matches = list(original_pattern.finditer(email_text))  # -----Original Message-----

        forward_parts = []
        original_parts = []
        main_text = email_text

        if forward_matches:
            # msin text before first forwards
            main_text = email_text[: forward_matches[0].start()]
            for i in range(len(forward_matches)):
                start = forward_matches[i].start()
                end = (
                    forward_matches[i + 1].start()
                    if i + 1 < len(forward_matches)
                    else len(email_text)
                )
                forward_parts.append(email_text[start:end])

        if original_matches:
            # main text before first original
            main_text = email_text[: original_matches[0].start()]
            for i in range(len(original_matches)):
                start = original_matches[i].start()
                end = (
                    original_matches[i + 1].start()
                    if i + 1 < len(original_matches)
                    else len(email_text)
                )
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

        # original
        for org_text in original_parts:
            header_part_original, body_original = self.split_headers_body(org_text)
            main_id = next(iter(main_data["message_id"]))
            original_data = self.parse_headers_org(header_part_original, main_id)
            original_data["body"] = self.clean_body(body_original)
            original_data["type"] = "original"
            extracted_emails.append(original_data)

        # Forwarded emails
        for fwd_text in forwards_raw:
            fwd_data1 = {
                "message_id": next(iter(main_data["message_id"])) + "forwarded",
                "original_message_id": main_data["message_id"],
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
            fwd_data1["body"] = parsed_fwd.get("body")
            extracted_emails.append(fwd_data1)

        return extracted_emails

    def process_all_emails(self, skip=False):
        """Reads and processes all email files in the given folder and shows progress with stats."""
        data = {
            "message_id": [],
            "original_message_id": [],
            "main_id": [],
            "filename": [],
            "type": [],
            "date": [],
            "from": [],
            "X-From": [],
            "X-To": [],
            "original_sender": [],
            "original_Date": [],
            "to": [],
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

        i = 0

        print(f"📂 Looking for files in: {os.path.abspath(self.input_dir)}")
        with tqdm(total=total_files, desc="📬 Processing emails") as pbar:
            for root, dirs, files in os.walk(self.input_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            if i == 100:
                                break
                            i += 1
                            raw_text = f.read()
                            email_data = self.extract_all_emails(raw_text)
                            for email in email_data:
                                # if (
                                #     email.get("from")
                                #     or email.get("to")
                                #     or email.get("subject")
                                # ):
                                email["filename"] = filename
                                for key in data.keys():
                                    if key in email:
                                        value = email.get(key)
                                        if isinstance(value, set):
                                            data[key].append(next(iter(value)) if value else None)
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
                            # else:
                            #     print(
                            #         f"⚠️ Skipping entry due to missing essential header fields: {filename}"
                            #     )
                    # Update tqdm bar with counts
                    pbar.set_postfix(
                        {
                            "Total": total_emails,
                            "Original": original_count,
                            "Forwarded": forwarded_count,
                            "main": main_count,
                        }
                    )
                    pbar.update(1)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n✅ Done! Total emails: {total_emails}")
        print(f"  Original: {original_count}, Forwarded: {forwarded_count}, main: {main_count}")
        print(f"⏱️ Time elapsed: {elapsed:.2f} seconds")

        df = pd.DataFrame(data)
        return df

    def save_to_json(self, df):
        """Save a Pandas DataFrame to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.json")
        output_path = os.path.join(self.output_dir, "clean_emails_1.json")
        df.to_json(
            output_path,
            orient="records",
            indent=2,
            force_ascii=False,
        )
        print(f"\n✅ Saved {len(df)} rows to {output_path}")

    def save_to_pickle(self, df: pd.DataFrame):
        """Save a list of cleaned email dataframe to a pickle file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.pkl")
        df.to_pickle(output_path)
        print(f"\n✅ Saved {len(df)} cleaned emails to {output_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load the processed data from a pickle file.
        """
        df = load_processed_df(self.output_dir, "processed_data_")
        return df


if __name__ == "__main__":
    # Create a DataPreparation instance
    data_prep = DataPreparation()

    data = data_prep.process_all_emails()
    data_prep.save_to_json(data)
    data_prep.save_to_pickle(data)

    print("Done!")
