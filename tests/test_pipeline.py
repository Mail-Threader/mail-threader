import os
import pytest
import pandas as pd
from data_preparation.pipeline import DataPreparation


def test_limit_returns_small_df():
    dp = DataPreparation(output_dir="output/processed_data")
    df = dp.process_all_emails(limit=5)
    assert len(df) > 0
    assert all(c in df.columns for c in ["message_id", "type", "subject", "body", "thread_id", "content_type"])


def test_exact_dedup_works():
    dp = DataPreparation(output_dir="output/processed_data")
    df = dp.process_all_emails(limit=3)
    mids = df["message_id"].tolist()
    assert len(mids) == len(set(mids))
