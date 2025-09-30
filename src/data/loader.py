import logging
import os
import time
from io import BytesIO

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.config import Config

def _make_s3_client() -> boto3.session.Session.client:
    """Create an S3 client with configurable retries/timeouts from env vars."""
    read_timeout = float(os.environ.get("S3_READ_TIMEOUT", "120"))
    connect_timeout = float(os.environ.get("S3_CONNECT_TIMEOUT", "30"))
    max_attempts = int(os.environ.get("S3_MAX_ATTEMPTS", "5"))
    max_pool = int(os.environ.get("S3_MAX_POOL", "50"))
    region = os.environ.get("S3_REGION")
    cfg = Config(
        retries={"max_attempts": max_attempts, "mode": "standard"},
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        max_pool_connections=max_pool,
        tcp_keepalive=True,
    )
    kwargs = {"config": cfg}
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def load_data(bucket_name, comments_prefix, posts_prefix, sample=False, nrows=100):
    s3 = _make_s3_client()

    def list_all_keys(prefix: str):
        continuation = None
        while True:
            kwargs = {"Bucket": bucket_name, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            resp = s3.list_objects_v2(**kwargs)
            for item in resp.get("Contents", []):
                yield item["Key"]
            if resp.get("IsTruncated"):
                continuation = resp.get("NextContinuationToken")
            else:
                break

    def load_from_prefix(prefix, label):
        print(f"Listing files for {label} (prefix={prefix})...")
        dataframes = []
        found_any = False
        for key in list_all_keys(prefix):
            if not key.endswith('.parquet'):
                continue
            found_any = True
            print(f"Loading {label} file: {key}")
            try:
                response = s3.get_object(Bucket=bucket_name, Key=key)
                body = BytesIO(response['Body'].read())
                table = pq.read_table(body)
                df = table.to_pandas() if not sample else table.to_pandas().head(nrows)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {key}: {e}")
            if sample:
                # when sampling, only take the first matching file
                break
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        if not found_any:
            print(f"No parquet files found for {label}.")
        return pd.DataFrame()

    comments_df = load_from_prefix(comments_prefix, "comments")
    posts_df = load_from_prefix(posts_prefix, "posts")

    return comments_df, posts_df


from typing import Optional, Set, List, Tuple
from src.utils.processed_store import ProcessedKeyStore


def list_keys_with_sizes(bucket_name: str, prefix: str):
    """Yield (key, size_bytes) for all objects under a prefix."""
    s3 = _make_s3_client()
    continuation = None
    while True:
        kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            key = item["Key"]
            if key.endswith('.parquet'):
                yield key, int(item.get("Size", 0))
        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break


def iter_load_data(
    bucket_name: str,
    comments_prefix: str,
    posts_prefix: str,
    files_per_chunk: int = 10,
    processed_keys: Optional[Set[str]] = None,
    sample_rows_per_file: Optional[int] = None,
    rows_per_chunk: Optional[int] = None,
    processed_store_path: Optional[str] = None,
):
    """
    Yield concatenated DataFrame chunks from S3 parquet files across both prefixes.

    - files_per_chunk: number of parquet files to include per yielded chunk
    - processed_keys: set of keys already processed; they will be skipped
    - sample_rows_per_file: if set, only take head(N) rows from each file
    - rows_per_chunk: ULTRA-FAST mode - limit each chunk to N rows
    """
    logger = logging.getLogger("classifier.loader")
    logger.info(f"ULTRA-FAST MODE: rows_per_chunk={rows_per_chunk}, files_per_chunk={files_per_chunk}")
    s3 = boto3.client('s3')
    logger = logging.getLogger("classifier.loader")
    processed = processed_keys or set()
    store = ProcessedKeyStore(processed_store_path) if processed_store_path else None

    def list_all_keys(prefix: str):
        continuation = None
        while True:
            kwargs = {"Bucket": bucket_name, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            resp = s3.list_objects_v2(**kwargs)
            for item in resp.get("Contents", []):
                key = item["Key"]
                if key.endswith('.parquet'):
                    yield key
            if resp.get("IsTruncated"):
                continuation = resp.get("NextContinuationToken")
            else:
                break

    current_keys: List[str] = []
    current_frames: List[pd.DataFrame] = []
    current_rows: int = 0

    def flush_chunk():
        nonlocal current_keys, current_frames
        nonlocal current_rows
        if current_frames:
            yield_df = pd.concat(current_frames, ignore_index=True) if len(current_frames) > 1 else current_frames[0]
            yield_keys = list(current_keys)
            # reset
            current_keys = []
            current_frames = []
            current_rows = 0
            return yield_df, yield_keys
        return None

    for prefix, label in ((comments_prefix, "comments"), (posts_prefix, "posts")):
        logger.info(f"Scanning prefix for chunks: {prefix}")
        for key in list_all_keys(prefix):
            if key in processed or (store and store.exists(key)):
                continue
            # Retries with exponential backoff for transient errors/timeouts
            base_sleep = float(os.environ.get("S3_RETRY_BACKOFF_BASE", "1.0"))
            max_sleep = float(os.environ.get("S3_RETRY_BACKOFF_MAX", "20.0"))
            attempts = int(os.environ.get("S3_PER_FILE_ATTEMPTS", "3"))
            attempt = 0
            while True:
                try:
                    resp = s3.get_object(Bucket=bucket_name, Key=key)
                    body = BytesIO(resp['Body'].read())
                    if rows_per_chunk is None:
                        # Load entire file
                        table = pq.read_table(body)
                        df = table.to_pandas()
                        if sample_rows_per_file is not None:
                            df = df.head(sample_rows_per_file)
                        df = df.copy()
                        df["__source_label__"] = label
                        current_frames.append(df)
                        current_keys.append(key)
                        current_rows += len(df)
                    else:
                        # ULTRA-FAST: Stream by row group to limit memory and yield by rows_per_chunk
                        pf = pq.ParquetFile(body)
                        for rg_idx in range(pf.num_row_groups):
                            table_rg = pf.read_row_group(rg_idx)
                            df_rg = table_rg.to_pandas()
                            if sample_rows_per_file is not None:
                                df_rg = df_rg.head(sample_rows_per_file)
                            if df_rg.empty:
                                continue
                            
                            # ULTRA-FAST: Limit to rows_per_chunk
                            remaining_rows = rows_per_chunk - current_rows
                            if remaining_rows <= 0:
                                break
                            if len(df_rg) > remaining_rows:
                                df_rg = df_rg.head(remaining_rows)
                            
                            df_rg = df_rg.copy()
                            df_rg["__source_label__"] = label
                            current_frames.append(df_rg)
                            current_keys.append(key)
                            current_rows += len(df_rg)
                            
                            # ULTRA-FAST: Yield chunk as soon as we hit rows_per_chunk
                            if current_rows >= rows_per_chunk:
                                chunk = flush_chunk()
                                if chunk:
                                    yield chunk
                                    break  # Stop processing this file
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= attempts:
                        logger.error(f"Error loading {key}: {e} (giving up after {attempts} attempts)")
                        # skip this key, move on
                        break
                    sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                    logger.warning(f"Error loading {key}: {e} (attempt {attempt}/{attempts}), retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)

            if rows_per_chunk is None and len(current_keys) >= files_per_chunk:
                chunk = flush_chunk()
                if chunk:
                    yield chunk

    # final tail
    tail = flush_chunk()
    if tail:
        yield tail
