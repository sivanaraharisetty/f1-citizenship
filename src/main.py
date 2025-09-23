import os
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.loader import load_data, iter_load_data, list_keys_with_sizes
from src.data.preprocess import preprocess_data
from src.model.train import BertClassifierTrainer
from src.utils.alerts import send_email_alert, format_exception_alert
from src.utils.helpers import set_global_seed, setup_logging
import yaml

def main():
    # --- Config ---
    config_path = os.environ.get("CLASSIFIER_CONFIG", os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml"))
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}

    # logging setup
    logger = setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_file=cfg.get("logging", {}).get("log_file", ""),
    )

    bucket_name = cfg.get("data", {}).get("s3", {}).get("bucket", "coop-published-zone-298305347319")
    comments_prefix = cfg.get("data", {}).get("s3", {}).get("comments_path", "arcticshift_reddit/comments/")
    posts_prefix = cfg.get("data", {}).get("s3", {}).get("posts_path", "arcticshift_reddit/posts/")
    default_seed = int(os.environ.get("SEED", "42"))
    set_global_seed(default_seed)
    # results/YYYYMMDD
    run_date = datetime.utcnow().strftime("%Y%m%d")
    results_root = "results"
    results_dir = os.path.join(results_root, run_date)
    os.makedirs(results_dir, exist_ok=True)

    # --- Load/initialize state ---
    state_path = os.path.join(results_dir, "state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
    else:
        state = {"processed_keys": [], "label_mapping": None, "last_checkpoint": None, "chunk_index": 0}

    processed_keys = set(state.get("processed_keys", []))
    chunk_index = int(state.get("chunk_index", 0))

    # --- Initialize trainer ---
    lr = float(cfg.get("model", {}).get("parameters", {}).get("learning_rate", 2e-5))
    if state.get("last_checkpoint"):
        trainer = BertClassifierTrainer(num_labels=len(state.get("label_mapping", []) or [0,1]), resume_checkpoint=state["last_checkpoint"], learning_rate=lr)
    else:
        # temporary 2 labels; will be updated after first chunk builds mapping
        trainer = BertClassifierTrainer(num_labels=2, learning_rate=lr)

    # --- Alert config (set your verified SES emails here) ---
    alert_sender = os.environ.get("ALERT_SENDER_EMAIL", "")
    alert_recipient = os.environ.get("ALERT_RECIPIENT_EMAIL", "lahari.naraharisetty@austin.utexas.edu")
    alert_region = os.environ.get("ALERT_AWS_REGION", "us-east-1")

    # --- Estimate total bytes remaining (for ETA) ---
    try:
        total_bytes = 0
        for p in (comments_prefix, posts_prefix):
            for key, size in list_keys_with_sizes(bucket_name, p):
                if key not in processed_keys:
                    total_bytes += size
        logger.info(f"Estimated unprocessed bytes: {total_bytes}")
    except Exception as e:
        logger.warning(f"Failed to estimate total bytes: {e}")
        total_bytes = 0

    processed_bytes = 0
    # --- Iterate chunks ---
    for df_raw, keys in iter_load_data(
        bucket_name,
        comments_prefix,
        posts_prefix,
        files_per_chunk=int(cfg.get("data", {}).get("chunking", {}).get("files_per_chunk", 10)),
        processed_keys=processed_keys,
        sample_rows_per_file=None,
        rows_per_chunk=int(cfg.get("data", {}).get("chunking", {}).get("rows_per_chunk", 0)) or None,
        processed_store_path=os.path.join(results_dir, "processed_keys.sqlite"),
    ):
        logger.info(f"Processing chunk {chunk_index} with {len(keys)} files and {len(df_raw)} rows")
        chunk_start = datetime.utcnow()
        try:
            # Build text fields per source
            if "body" in df_raw.columns:
                mask = df_raw["__source_label__"] == "comments"
                df_raw.loc[mask, "text"] = df_raw.loc[mask, "body"].fillna("")
            if "title" in df_raw.columns or "selftext" in df_raw.columns:
                mask = df_raw["__source_label__"] == "posts"
                title = df_raw.loc[mask, "title"].fillna("") if "title" in df_raw.columns else ""
                selftext = df_raw.loc[mask, "selftext"].fillna("") if "selftext" in df_raw.columns else ""
                df_raw.loc[mask, "text"] = title + " " + selftext

            # Preprocess + label
            processed = preprocess_data(df_raw)
            processed = processed[["text", "label"]].dropna()
            if processed.empty:
                logger.warning("Chunk produced no training rows; skipping.")
                processed_keys.update(keys)
                state.update({"processed_keys": list(processed_keys), "chunk_index": chunk_index + 1})
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
                chunk_index += 1
                continue

            # Label mapping (global, stable across chunks)
            if not state.get("label_mapping"):
                cats = pd.Categorical(processed["label"]).categories
                label_mapping = list(cats)
                state["label_mapping"] = label_mapping
                with open(os.path.join(results_dir, "label_mapping.json"), "w") as f:
                    json.dump({i: v for i, v in enumerate(label_mapping)}, f, indent=4)
                # reinit trainer with correct label size if needed
                if len(label_mapping) != trainer.model.num_labels:
                    trainer = BertClassifierTrainer(num_labels=len(label_mapping), resume_checkpoint=state.get("last_checkpoint"))
            else:
                # ensure on-disk mapping exists and sync from disk if needed
                mapping_path = os.path.join(results_dir, "label_mapping.json")
                if os.path.exists(mapping_path):
                    try:
                        with open(mapping_path, "r") as f:
                            existing = json.load(f)
                        # preserve order by numeric keys if stored as dict
                        if isinstance(existing, dict):
                            ordered = [existing[str(i)] if str(i) in existing else existing[i] for i in range(len(existing))]
                            state["label_mapping"] = ordered
                    except Exception as e:
                        logger.warning(f"Failed to read label mapping from disk: {e}")

            # Encode labels to stable integer codes using global mapping
            mapping = {v: i for i, v in enumerate(state["label_mapping"])}
            before = len(processed)
            processed = processed[processed["label"].isin(mapping.keys())].copy()
            dropped = before - len(processed)
            if dropped:
                logger.warning(f"Dropped {dropped} rows with unseen labels not in mapping")
            processed["label"] = processed["label"].map(mapping).astype(int)

            # Train on this chunk
            # Save under results/YYYYMMDD/chunk_{i}
            ckpt_dir = os.path.join(results_dir, f"chunk_{chunk_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            epochs = int(cfg.get("model", {}).get("parameters", {}).get("epochs", 1))
            metrics = trainer.train_on_dataframe(processed, epochs=epochs, save_dir=ckpt_dir, resume_from=None)
            # Save checkpoint path (HF saves to output_dir)
            state["last_checkpoint"] = ckpt_dir

            # Update processed keys and state
            processed_keys.update(keys)
            state["processed_keys"] = list(processed_keys)
            state["chunk_index"] = chunk_index + 1
            # Persist metrics snapshot (global file and per-chunk file)
            chunk_record = {
                "date": run_date,
                "chunk_index": chunk_index,
                "metrics": metrics,
                "files": keys,
                "num_files": len(keys),
                "num_rows": int(len(processed)),
            }
            with open(os.path.join(results_dir, "evaluation_results.json"), "a") as f:
                f.write(json.dumps(chunk_record) + "\n")
            with open(os.path.join(ckpt_dir, f"metrics_{run_date}_chunk_{chunk_index}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(ckpt_dir, f"files_{run_date}_chunk_{chunk_index}.json"), "w") as f:
                json.dump({
                    "date": run_date,
                    "chunk_index": chunk_index,
                    "files": keys,
                    "num_files": len(keys),
                    "num_rows": int(len(processed)),
                }, f, indent=2)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            # bytes processed in this chunk
            try:
                chunk_bytes = 0
                s3_client = None
                # Not re-listing keys; estimate from HEAD requests could be added. Here we approximate via listing cache at start.
                # For accuracy, re-list and sum sizes of keys in this chunk.
                for p in (comments_prefix, posts_prefix):
                    for key, size in list_keys_with_sizes(bucket_name, p):
                        if key in keys:
                            chunk_bytes += size
                processed_bytes += chunk_bytes
            except Exception:
                chunk_bytes = 0
            elapsed = (datetime.utcnow() - chunk_start).total_seconds()
            speed = (chunk_bytes / elapsed) if elapsed > 0 and chunk_bytes else 0
            remaining = (total_bytes - processed_bytes) if total_bytes else 0
            eta_sec = (remaining / speed) if speed > 0 else 0
            if speed > 0 and remaining > 0:
                logger.info(f"Finished chunk {chunk_index}. Metrics: {metrics}. Speed ~ {speed/1e6:.2f} MB/s. ETA ~ {int(eta_sec/3600)}h {int((eta_sec%3600)/60)}m")
            else:
                logger.info(f"Finished chunk {chunk_index}. Metrics: {metrics}")
            chunk_index += 1
        except Exception as err:
            # Send alert email if configured
            if alert_sender and alert_recipient:
                subject = f"S3 Classifier Failure on chunk {chunk_index}"
                body = format_exception_alert(f"Run date {run_date}, chunk {chunk_index}", err)
                send_email_alert(subject, body, sender=alert_sender, recipient=alert_recipient, region=alert_region)
            # Re-raise after alert so the job fails visibly
            raise

if __name__ == "__main__":
    main()
