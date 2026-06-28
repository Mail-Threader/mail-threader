import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from loguru import logger


def fuzzy_dedup(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    min_len = 200
    texts = []
    valid_indices = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        body = (row.get("body") or "").strip()
        subject = (row.get("subject") or "").strip()
        text = f"{subject} {body}".strip()
        if len(text) >= min_len:
            texts.append(text)
            valid_indices.append(idx)

    if len(texts) < 2:
        df["duplicate_of"] = None
        return df

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=min(20, len(texts)), metric="cosine", algorithm="brute")
    nn.fit(tfidf)
    distances, indices = nn.kneighbors(tfidf)

    body_lengths = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        body = row.get("body") or ""
        body_lengths[row["message_id"]] = len(body)

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    mid_to_idx = {}
    for i, idx in enumerate(valid_indices):
        mid_to_idx[df.iloc[idx]["message_id"]] = i

    duplicate_of = {}

    for i in order:
        orig_idx = valid_indices[i]
        original_mid = df.iloc[orig_idx]["message_id"]
        if original_mid in duplicate_of.values() or original_mid in duplicate_of:
            continue
        for j in range(1, len(indices[i])):
            sim = 1 - distances[i][j]
            if sim < threshold:
                break
            peer_idx = valid_indices[indices[i][j]]
            peer_mid = df.iloc[peer_idx]["message_id"]
            if peer_mid in duplicate_of or peer_mid in duplicate_of.values():
                continue
            if peer_mid == original_mid:
                continue
            if body_lengths[original_mid] >= body_lengths[peer_mid]:
                duplicate_of[peer_mid] = original_mid
            else:
                duplicate_of[original_mid] = peer_mid

    df["duplicate_of"] = None
    for dup_mid, kept_mid in duplicate_of.items():
        df.loc[df["message_id"] == dup_mid, "duplicate_of"] = kept_mid

    logger.info(
        f"Fuzzy dedup: {len(duplicate_of)} near-duplicates found "
        f"({len(texts)} texts compared, threshold={threshold})"
    )

    return df
