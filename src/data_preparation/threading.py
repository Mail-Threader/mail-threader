import pandas as pd
from loguru import logger


def build_threads(df: pd.DataFrame) -> pd.DataFrame:
    mid_set = set(df["message_id"].dropna().unique())

    parent_map = {}
    for _, row in df.iterrows():
        mid = row["message_id"]
        pid = row.get("parent_message_id")
        if pid and pid in mid_set:
            parent_map[mid] = pid

    roots = [mid for mid in df["message_id"] if mid not in parent_map]
    root_set = set(roots)

    children = {}
    for mid, pid in parent_map.items():
        children.setdefault(pid, []).append(mid)

    thread_id = {}
    thread_depth = {}

    def assign(root_mid):
        queue = [(root_mid, 0)]
        while queue:
            mid, depth = queue.pop(0)
            thread_id[mid] = root_mid
            thread_depth[mid] = depth
            for child in children.get(mid, []):
                queue.append((child, depth + 1))

    for root in roots:
        assign(root)

    df["thread_id"] = df["message_id"].map(thread_id)
    df["thread_depth"] = df["message_id"].map(thread_depth).fillna(0).astype(int)

    orphans = df[df["thread_id"].isna()]
    if len(orphans):
        logger.warning(f"{len(orphans)} emails could not be assigned to any thread")
        for idx in orphans.index:
            mid = df.at[idx, "message_id"]
            df.at[idx, "thread_id"] = mid
            df.at[idx, "thread_depth"] = 0

    logger.info(
        f"Threads: {len(roots)} roots, "
        f"{df['thread_id'].nunique()} total threads, "
        f"{len(orphans)} orphans"
    )

    return df
