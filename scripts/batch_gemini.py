"""
Gemini Batch API worker for Minority-Shield.

Processes pending opinions in bulk via the Gemini Batch API — 50% cheaper
than real-time calls and runs entirely in the background (no caffeinate needed).

How it works:
  1. Fetch pending rows from raw_opinions
  2. Write them as a JSONL request file → upload to GCS
  3. Submit a Gemini batch job pointing at that file
  4. Poll until the job completes
  5. Download results from GCS, parse, write to cases + law_firms tables

Requirements:
  - A GCS bucket (set GCS_BUCKET env var)
  - Google Cloud credentials (run: gcloud auth application-default login)
  - google-cloud-storage package: pip install google-cloud-storage

Usage:
    python scripts/batch_gemini.py --submit          # create and submit a batch job
    python scripts/batch_gemini.py --poll JOB_NAME   # poll an existing job and ingest results

Environment variables:
    DATABASE_URL    PostgreSQL connection URL
    GEMINI_API_KEY  Gemini API key
    GCS_BUCKET      GCS bucket name (e.g. my-minority-shield-bucket)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.gemini_parser import (
    EXTRACTION_PROMPT, CaseMetadata, _strip_markdown_fences, CASE_TYPES
)
from pydantic import ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
POLL_INTERVAL = 60  # seconds between status checks


# ---------------------------------------------------------------------------
# Step 1: Build JSONL request file
# ---------------------------------------------------------------------------

def build_request_jsonl(rows: list[dict], output_path: str) -> int:
    """
    Write one Gemini request per opinion to a JSONL file.
    Each line is a self-contained request with a custom_id for tracking.
    Returns number of requests written.
    """
    count = 0
    with open(output_path, "w") as f:
        for row in rows:
            prompt = EXTRACTION_PROMPT.format(opinion_text=row["raw_text"][:50_000])
            request = {
                "custom_id": str(row["id"]),   # opinion id — used to match results back
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": {
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                },
            }
            f.write(json.dumps(request) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Step 2: Upload to GCS
# ---------------------------------------------------------------------------

def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """Upload a file to GCS and return the gs:// URI."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{blob_name}"
    log.info("Uploaded to %s", uri)
    return uri


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str) -> list[str]:
    """Download all blobs under a GCS prefix to a local directory."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    paths = []
    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name.replace("/", "_"))
        blob.download_to_filename(local_path)
        paths.append(local_path)
    log.info("Downloaded %d file(s) from gs://%s/%s", len(blobs), bucket_name, prefix)
    return paths


# ---------------------------------------------------------------------------
# Step 3: Submit batch job
# ---------------------------------------------------------------------------

def submit_batch_job(local_jsonl_path: str, api_key: str) -> str:
    """Upload JSONL via Gemini Files API, then submit a batch job. Returns the job name."""
    client = genai.Client(api_key=api_key)

    # Upload the JSONL file via Gemini Files API (required for non-Vertex batch)
    log.info("Uploading JSONL to Gemini Files API ...")
    uploaded = client.files.upload(
        file=local_jsonl_path,
        config=types.UploadFileConfig(mime_type="application/jsonl"),
    )
    log.info("File uploaded: %s", uploaded.name)

    job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
    )
    log.info("Batch job created: %s", job.name)
    log.info("State: %s", job.state)
    return job.name


# ---------------------------------------------------------------------------
# Step 4: Poll for completion
# ---------------------------------------------------------------------------

def poll_until_done(job_name: str, api_key: str) -> object:
    """Poll the batch job until it succeeds or fails. Returns the completed job object."""
    client = genai.Client(api_key=api_key)
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}

    while True:
        job = client.batches.get(name=job_name)
        state = str(job.state)
        log.info("Job %s — state: %s", job_name, state)

        if state in terminal_states:
            if state == "JOB_STATE_SUCCEEDED":
                log.info("Batch job completed successfully.")
                return job
            else:
                log.error("Batch job ended with state: %s", state)
                return None

        log.info("Waiting %ds before next poll ...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Step 5: Ingest results back into the database
# ---------------------------------------------------------------------------

def _get_or_create_firm(cur, name: str) -> int:
    cur.execute(
        """
        INSERT INTO law_firms (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (name,),
    )
    return cur.fetchone()[0]


def ingest_results(result_files: list[str], db_url: str) -> None:
    """Parse batch output JSONL files and write results to the DB."""
    conn = psycopg2.connect(db_url)
    ok = fail = 0

    for fpath in result_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    fail += 1
                    continue

                opinion_id = int(result.get("custom_id", 0))
                if not opinion_id:
                    fail += 1
                    continue

                # Extract the model's text response
                try:
                    raw = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    raw = _strip_markdown_fences(raw.strip())
                    data = json.loads(raw)
                    metadata = CaseMetadata(**data)
                except (KeyError, IndexError, json.JSONDecodeError, ValidationError) as exc:
                    _mark_failed(conn, opinion_id, str(exc))
                    fail += 1
                    continue

                try:
                    with conn:
                        with conn.cursor() as cur:
                            p_id = _get_or_create_firm(cur, metadata.plaintiff_firm)
                            d_id = _get_or_create_firm(cur, metadata.defendant_firm)

                            if p_id == d_id:
                                raise ValueError(
                                    f"Same firm for both sides: '{metadata.plaintiff_firm}'"
                                )

                            cur.execute(
                                """
                                INSERT INTO cases
                                    (plaintiff_firm, defendant_firm, case_type, outcome, minority_focus)
                                VALUES (%s, %s, %s, %s, %s)
                                """,
                                (p_id, d_id, metadata.case_type,
                                 metadata.outcome, metadata.minority_focus),
                            )
                            cur.execute(
                                "UPDATE raw_opinions SET status='completed', processed_at=NOW() WHERE id=%s",
                                (opinion_id,),
                            )
                    ok += 1
                except Exception as exc:
                    _mark_failed(conn, opinion_id, str(exc))
                    fail += 1

    conn.close()
    log.info("Ingestion complete — inserted: %d, failed: %d", ok, fail)


def _mark_failed(conn, opinion_id: int, error: str) -> None:
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE raw_opinions SET status='failed', error_msg=%s, processed_at=NOW() WHERE id=%s",
                    (error[:1000], opinion_id),
                )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Fetch pending rows from DB
# ---------------------------------------------------------------------------

def fetch_pending(db_url: str, limit: int) -> list[dict]:
    conn = psycopg2.connect(db_url)
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, raw_text FROM raw_opinions WHERE status='pending' LIMIT %s",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Batch API worker for Minority-Shield.")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET"),
                        help="GCS bucket name.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit", action="store_true",
                      help="Fetch pending opinions, build JSONL, and submit a batch job.")
    mode.add_argument("--poll", metavar="JOB_NAME",
                      help="Poll an existing job and ingest results when done.")

    parser.add_argument("--limit", type=int, default=50_000,
                        help="Max opinions to include in one batch (default: 50000).")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: DATABASE_URL not set.")
        sys.exit(1)
    if not args.api_key:
        print("Error: GEMINI_API_KEY not set.")
        sys.exit(1)
    if not args.bucket:
        print("Error: GCS_BUCKET not set.")
        sys.exit(1)

    if args.submit:
        log.info("Fetching pending opinions ...")
        rows = fetch_pending(args.db_url, args.limit)
        log.info("Found %d pending opinions.", len(rows))

        if not rows:
            print("No pending opinions to process.")
            sys.exit(0)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        count = build_request_jsonl(rows, tmp_path)
        log.info("Built %d requests → %s", count, tmp_path)

        job_name = submit_batch_job(tmp_path, args.api_key)
        print(f"\nBatch job submitted: {job_name}")
        print(f"To poll for results, run:")
        print(f"  python scripts/batch_gemini.py --poll '{job_name}'")

    elif args.poll:
        client = genai.Client(api_key=args.api_key)
        job = poll_until_done(args.poll, args.api_key)
        if not job:
            sys.exit(1)

        # Results are available by iterating the completed job
        log.info("Fetching results ...")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            tmp_path = tmp.name
            for result in client.batches.list_results(name=args.poll):
                tmp.write(json.dumps(result) + "\n")

        ingest_results([tmp_path], args.db_url)
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
