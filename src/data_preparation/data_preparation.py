import os
import re
import time
from datetime import datetime

import dateparser
import pandas as pd
from loguru import logger
from tqdm import tqdm


class DataPreparation:
    """
    Class responsible for data preparation and storage tasks:
    - Loading email data
    - Cleaning and preprocessing text
    - Extracting metadata (sender, recipient, date, etc.)
    - Storing processed data
    """

    def __init__(self, input_dir="./data/", output_dir="./processed_data/", save_to_db=False):
        """
        Initialize the DataPreparation class.

        Args:
            input_dir (str): Directory containing the email files
            output_dir (str): Directory to store processed data
            save_to_db (bool): Whether to save processed data to PostgreSQL database
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_to_db = save_to_db

        # Create an output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def normalize_dates(text):
        if text is None:
            return None
        cleaned_text = re.sub(r"\s*\([A-Z]{2,}\)$", "", text.strip())
        parsed_date = dateparser.parse(cleaned_text)
        if parsed_date:
            european_date = parsed_date.strftime("%d/%m/%Y %H:%M:%S")
            text = text.replace(text, european_date)
        return text

    @staticmethod
    def split_headers_body(email_text):
        """Splits raw email into headers and body."""  # docstring
        parts = email_text.split("\n\n", 1)
        headers = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        return headers, body

    def parse_headers_org(self, header_text, main_id):
        """Parses the original headers into a dictionary."""
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

    @staticmethod
    def parse_forwarded_block(forward_block):
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

        # Extract the message body after the subject
        message_match = re.search(r"Subject:.*?\n\n(.*)", forward_block, re.DOTALL)
        fwd_data["body"].add(message_match.group(1).strip()) if message_match else None

        return fwd_data

    @staticmethod
    def clean_body(body):
        """Clean body text (remove reply chains, extra spaces, etc.)."""

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

    @staticmethod
    def split_all_messages(email_text):
        """Split the main message and all forwarded and original messages (preserve markers)."""
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

        def filter_list(matches, context_list):
            _main_text = email_text[: matches[0].start()]
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(email_text)
                context_list.append(email_text[start:end])
            return _main_text

        if forward_matches:
            # main text before first forwards
            main_text = filter_list(forward_matches, forward_parts)

        if original_matches:
            # main text before first original
            main_text = filter_list(original_matches, original_parts)

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

    def process_all_emails(self):
        """
        Reads and processes all email files in the given folder and shows progress with stats.
        """
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
        logger.info(f"📂 Looking for files in: {os.path.abspath(self.input_dir)}")

        with tqdm(total=total_files, desc="📬 Processing emails") as pbar:
            for root, dirs, files in os.walk(self.input_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
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
        logger.success(f"\n✅ Done! Total emails: {total_emails}")
        logger.info(f"Original: {original_count}, Forwarded: {forwarded_count}, main: {main_count}")
        logger.info(f"⏱️ Time elapsed: {elapsed:.2f} seconds")

        df = pd.DataFrame(data)

        # Save to PostgreSQL if requested
        if self.save_to_db:
            try:
                logger.info("Saving processed emails to PostgreSQL database...")
                table_name = "processed_emails"
                success_message = f"\n✅ Saved {len(df)} rows to PostgreSQL table: {table_name}"
                save_to_postgresql(
                    df, table_name, if_exists="replace", success_message=success_message
                )
            except Exception as e:
                logger.error(f"Failed to save to PostgreSQL: {e}")
                logger.info("Continuing without saving to database.")

        return df

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

    def load_data(self, file_path: str | None = None):
        """Load the cleaned email dataframe from a pickle file."""
        if file_path is None:
            # Find the most recent processed data file
            pkl_files = [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("processed_data_") and f.endswith(".pkl")
            ]
            logger.info(f"Found {len(pkl_files)} processed data files in {self.output_dir}")
            if not pkl_files:
                logger.error(f"No processed data files found in {self.input_dir}")
                return pd.DataFrame()

            # Sort by timestamp in the filename
            pkl_files.sort(reverse=True)
            file_path = os.path.join(self.output_dir, pkl_files[0])

        try:
            df = pd.read_pickle(file_path)
            logger.info(f"Loaded data from {file_path}: {len(df)} emails")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Create a DataPreparation instance
    data_prep = DataPreparation()

    data_df = data_prep.process_all_emails()
    data_prep.save_to_json(data_df)
    data_prep.save_to_pickle(data_df)

    logger.info("Data Preparation Finished")
