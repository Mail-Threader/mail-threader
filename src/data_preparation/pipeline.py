import os
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from tqdm import tqdm
import pandas as pd
from loguru import logger

from . import extractor
from . import threading
from . import classifier
from . import dedup
from utils import load_processed_df


def _worker(files_batch):
    results = []
    for file_path in files_batch:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception:
            continue
        emails = extractor.extract_all_emails(raw_text)
        for email_item in emails:
            if email_item.get("from") or email_item.get("to") or email_item.get("subject"):
                email_item["filename"] = filename
                results.append(email_item)
    return results


class DataPreparation:
    def __init__(self, input_dir="./data/", output_dir="./processed_data/", workers=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.workers = workers or os.cpu_count()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_all_emails(self, limit: Optional[int] = None):
        all_files = []
        for root, _dirs, files in os.walk(self.input_dir):
            for fname in files:
                all_files.append(os.path.join(root, fname))

        if limit:
            all_files = all_files[:limit]

        total_files = len(all_files)
        logger.info(f"Looking for files in: {os.path.join(self.input_dir)}")
        logger.info(f"Found {total_files} files, processing with {self.workers} workers")

        n_workers = min(self.workers, total_files) if total_files else 1
        chunk_size = math.ceil(total_files / n_workers) if n_workers else total_files
        chunks = [all_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

        all_results = []
        start_time = time.time()
        main_count = 0
        original_count = 0
        forwarded_count = 0
        total_emails = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, chunk): len(chunk) for chunk in chunks}
            with tqdm(total=total_files, desc="Processing emails") as pbar:
                for future in as_completed(futures):
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    for email_item in chunk_results:
                        total_emails += 1
                        t = email_item.get("type")
                        if t == "original":
                            original_count += 1
                        elif t == "forwarded":
                            forwarded_count += 1
                        elif t == "main":
                            main_count += 1
                    pbar.set_postfix({"Total": total_emails, "Orig": original_count, "Fwd": forwarded_count, "Main": main_count})
                    pbar.update(futures[future])

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"\nDone! Total emails: {total_emails}")
        logger.info(f"  Original: {original_count}, Forwarded: {forwarded_count}, Main: {main_count}")
        logger.info(f"Time elapsed: {elapsed:.2f} seconds")

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
        for email_item in all_results:
            for key in data:
                if key in email_item:
                    value = email_item.get(key)
                    data[key].append(value if value not in (None, "") else None)
                else:
                    data[key].append(None)

        df = pd.DataFrame(data)
        df = df.drop_duplicates(
            subset=["date", "from", "X-From", "X-To", "to", "subject", "cc", "X-cc", "body"],
            keep="first",
        )
        df = df.sort_values(by="date", ascending=True)
        df = dedup.fuzzy_dedup(df, threshold=0.90)
        df = threading.build_threads(df)
        df = classifier.classify_all(df)
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
            columns = ["message_id", "main_id", "filename", "type", "date", "from", "to", "subject", "body"]
            db.insert_from_dataframe(df, table_name, columns=columns)
            logger.info(f"Successfully saved {len(df)} rows to database table '{table_name}'")
        except Exception as e:
            logger.error(f"Error saving DataFrame to database: {e}")

    def load_data(self):
        return load_processed_df(self.output_dir, "processed_data_")
