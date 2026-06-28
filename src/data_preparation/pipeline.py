import os
import time
from datetime import datetime
from typing import Optional
from tqdm import tqdm
import pandas as pd
from loguru import logger

from . import extractor
from utils import load_processed_df


class DataPreparation:
    def __init__(self, input_dir="./data/", output_dir="./processed_data/"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_all_emails(self, limit: Optional[int] = None):
        data = {
            "message_id": [],
            "parent_message_id": [],
            "main_id": [],
            "filename": [],
            "type": [],
            "date": [],
            "from": [],
            "to": [],
            "cc": [],
            "X-From": [],
            "X-To": [],
            "X-cc": [],
            "subject": [],
            "body": [],
            "has_body": [],
            "is_html": [],
        }
        main_count = 0
        original_count = 0
        forwarded_count = 0
        total_emails = 0
        start_time = time.time()
        total_files = sum(len(files) for _, _, files in os.walk(self.input_dir))
        logger.info(f"Looking for files in: {os.path.join(self.input_dir)}")

        file_count = 0
        limit_reached = False
        with tqdm(total=total_files, desc="Processing emails") as pbar:
            for root, dirs, files in os.walk(self.input_dir):
                if limit_reached:
                    break
                for filename in files:
                    if limit is not None and file_count >= limit:
                        limit_reached = True
                        break
                    file_count += 1
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            raw_text = f.read()
                            email_data = extractor.extract_all_emails(raw_text)
                            for email_item in email_data:
                                if (
                                    email_item.get("from")
                                    or email_item.get("to")
                                    or email_item.get("subject")
                                ):
                                    email_item["filename"] = filename
                                    for key in data.keys():
                                        if key in email_item:
                                            value = email_item.get(key)
                                            data[key].append(value if value not in (None, "") else None)
                                        else:
                                            data[key].append(None)
                                    total_emails += 1
                                    t = email_item.get("type")
                                    if t == "original":
                                        original_count += 1
                                    elif t == "forwarded":
                                        forwarded_count += 1
                                    elif t == "main":
                                        main_count += 1
                    pbar.set_postfix({"Total": total_emails, "Orig": original_count, "Fwd": forwarded_count, "Main": main_count})
                    pbar.update(1)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"\nDone! Total emails: {total_emails}")
        logger.info(f"  Original: {original_count}, Forwarded: {forwarded_count}, Main: {main_count}")
        logger.info(f"Time elapsed: {elapsed:.2f} seconds")

        df = pd.DataFrame(data)
        df = df.drop_duplicates(
            subset=["date", "from", "X-From", "X-To", "to", "subject", "cc", "X-cc", "body"],
            keep="first",
        )
        df = df.sort_values(by="date", ascending=True)
        return df

    def save_to_json(self, df):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.json")
        df.to_json(output_path, orient="records", indent=2, force_ascii=False)
        logger.info(f"\nSaved {len(df)} rows to {output_path}")

    def save_to_pickle(self, df: pd.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.pkl")
        df.to_pickle(output_path)
        logger.info(f"\nSaved {len(df)} cleaned emails to {output_path}")

    def save_to_csv(self, df: pd.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"processed_data_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"\nSaved {len(df)} rows to {output_path}")

    def save_to_database(self, df: pd.DataFrame, db):
        if df is None or df.empty:
            logger.warning("DataFrame is empty. Skipping database save.")
            return
        if db is None:
            logger.error("Database connection is not provided. Skipping database save.")
            return
        try:
            table_name = "processed_emails"
            columns = [
                "message_id", "main_id", "filename", "type",
                "date", "from", "to", "subject", "body",
            ]
            db.insert_from_dataframe(df, table_name, columns=columns)
            logger.info(f"Successfully saved {len(df)} rows to database table '{table_name}'")
        except Exception as e:
            logger.error(f"Error saving DataFrame to database: {e}")

    def load_data(self):
        return load_processed_df(self.output_dir, "processed_data_")
