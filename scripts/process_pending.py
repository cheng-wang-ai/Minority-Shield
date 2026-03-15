"""
Gemini extraction worker for Minority-Shield (async, concurrent).

Reads 'pending' opinions from the raw_opinions staging table, calls the Gemini
API concurrently to extract structured metadata, and writes results to the cases table.

Restart-safe: rows are marked 'processing' before the API call, and 'completed'
or 'failed' after. Crashed runs leave rows in 'failed' or 'processing' state —
re-run with --retry-processing to recover stuck rows.

Usage:
    python scripts/process_pending.py [--concurrency 20] [--batch-size 100] [--retry-processing]

Environment variables:
    DATABASE_URL    PostgreSQL connection URL
    GEMINI_API_KEY  Gemini API key
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from google.genai.errors import ClientError

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.gemini_parser import GeminiParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

_BATCH_SIZE            = 30   # rows fetched from DB per loop iteration
_CONCURRENCY           = 5    # max simultaneous Gemini requests
_MAX_RETRIES_PER_MODEL = 2    # consecutive 429s before switching to next model
_MAX_RETRIES           = 6    # total retries across all models before giving up
_RETRY_BASE            = 45.0 # seconds — base wait on 429


# ---------------------------------------------------------------------------
# Database helpers (synchronous — run via asyncio.to_thread)
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


def _fetch_pending_batch(db_url: str, batch_size: int, include_processing: bool) -> list[dict]:
    statuses = ("pending", "processing") if include_processing else ("pending",)
    placeholders = ",".join(["%s"] * len(statuses))
    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, raw_text
                    FROM raw_opinions
                    WHERE status IN ({placeholders})
                    ORDER BY fetched_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (*statuses, batch_size),
                )
                rows = cur.fetchall()
                # Mark all fetched rows as processing in the same transaction
                if rows:
                    ids = [r["id"] for r in rows]
                    cur.execute(
                        f"""
                        UPDATE raw_opinions
                        SET status = 'processing', processed_at = NOW()
                        WHERE id = ANY(%s)
                        """,
                        (ids,),
                    )
                return [dict(r) for r in rows]
    finally:
        conn.close()


def _write_result(db_url: str, opinion_id: int, raw_text: str, metadata) -> None:
    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                p_id = _get_or_create_firm(cur, metadata.plaintiff_firm)
                d_id = _get_or_create_firm(cur, metadata.defendant_firm)

                if p_id == d_id:
                    raise ValueError(
                        f"Plaintiff and defendant resolved to the same firm: '{metadata.plaintiff_firm}'"
                    )

                cur.execute(
                    """
                    INSERT INTO cases
                        (plaintiff_firm, defendant_firm, case_type, outcome,
                         minority_focus, source_text, decision_date, court)
                    VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL)
                    """,
                    (p_id, d_id, metadata.case_type, metadata.outcome,
                     metadata.minority_focus, raw_text[:10_000]),
                )
                cur.execute(
                    "UPDATE raw_opinions SET status = 'completed', processed_at = NOW() WHERE id = %s",
                    (opinion_id,),
                )
    finally:
        conn.close()


def _mark_failed(db_url: str, opinion_id: int, error: str) -> None:
    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE raw_opinions SET status = 'failed', error_msg = %s, processed_at = NOW() WHERE id = %s",
                    (error[:1000], opinion_id),
                )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def _process_one(
    opinion_id: int,
    raw_text: str,
    parser: GeminiParser,
    semaphore: asyncio.Semaphore,
    switch_lock: asyncio.Lock,
    db_url: str,
    counters: dict,
) -> None:
    async with semaphore:
        consecutive_429s = 0  # track 429s on the current model

        for attempt in range(_MAX_RETRIES):
            try:
                metadata = await parser.parse_async(raw_text)
                break
            except ClientError as exc:
                if exc.code == 429:
                    consecutive_429s += 1

                    # After N consecutive 429s, switch to the next model.
                    # Use a lock so only one coroutine triggers the switch at a time.
                    if consecutive_429s >= _MAX_RETRIES_PER_MODEL:
                        async with switch_lock:
                            if parser.switch_to_next_model():
                                log.warning(
                                    "429 quota exhausted — switched to model: %s",
                                    parser.model_name,
                                )
                            else:
                                log.warning("429 quota exhausted — no more fallback models.")
                        consecutive_429s = 0

                    wait = _RETRY_BASE * (2 ** min(attempt, 4))
                    log.warning(
                        "429 rate limit on id=%d (model=%s) — retrying in %.0fs (attempt %d/%d)",
                        opinion_id, parser.model_name, wait, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                elif exc.code == 404:
                    # Model no longer exists — switch immediately without counting
                    # as a quality failure (it's an infrastructure issue, not output quality).
                    async with switch_lock:
                        if parser.switch_to_next_model():
                            log.warning(
                                "404 model unavailable — switched to model: %s",
                                parser.model_name,
                            )
                        else:
                            log.error("id=%d FAILED: 404 and no more fallback models.", opinion_id)
                            await asyncio.to_thread(_mark_failed, db_url, opinion_id, str(exc))
                            counters["fail"] += 1
                            return
                else:
                    # Other API error — record as quality failure
                    parser.record_result(success=False)
                    await asyncio.to_thread(_mark_failed, db_url, opinion_id, str(exc))
                    log.error("id=%d FAILED: %s", opinion_id, exc)
                    counters["fail"] += 1
                    return
            except Exception as exc:
                # Quality failure (bad JSON, validation error, Unknown firms, etc.)
                parser.record_result(success=False)
                await asyncio.to_thread(_mark_failed, db_url, opinion_id, str(exc))
                log.error("id=%d FAILED: %s", opinion_id, exc)
                counters["fail"] += 1

                # If the current model's quality has dropped below the threshold,
                # proactively switch to the next model without waiting for 429s.
                if not parser.is_current_model_reliable():
                    async with switch_lock:
                        if parser.switch_to_next_model():
                            log.warning(
                                "High failure rate detected — switched to model: %s",
                                parser.model_name,
                            )
                return
        else:
            msg = "Exceeded max retries across all fallback models"
            await asyncio.to_thread(_mark_failed, db_url, opinion_id, msg)
            log.error("id=%d FAILED: %s", opinion_id, msg)
            counters["fail"] += 1
            return

    try:
        await asyncio.to_thread(_write_result, db_url, opinion_id, raw_text, metadata)
        parser.record_result(success=True)
        log.info("id=%d OK (model=%s)", opinion_id, parser.model_name)
        counters["ok"] += 1
    except Exception as exc:
        # DB write failed or same-firm rejection — count as quality failure
        parser.record_result(success=False)
        await asyncio.to_thread(_mark_failed, db_url, opinion_id, str(exc))
        log.error("id=%d FAILED: %s", opinion_id, exc)
        counters["fail"] += 1


async def run_worker(
    db_url: str,
    api_key: str | None,
    batch_size: int,
    concurrency: int,
    retry_processing: bool,
) -> None:
    parser = GeminiParser(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    switch_lock = asyncio.Lock()  # prevents concurrent model switches
    counters = {"ok": 0, "fail": 0}

    log.info("Worker started (model=%s, concurrency=%d, batch_size=%d, retry_processing=%s)",
             parser.model_name, concurrency, batch_size, retry_processing)

    while True:
        rows = await asyncio.to_thread(
            _fetch_pending_batch, db_url, batch_size, retry_processing
        )

        if not rows:
            log.info("No pending opinions remaining. Done — processed: %d, failed: %d",
                     counters["ok"], counters["fail"])
            break

        log.info("Fetched %d rows — dispatching concurrently (model=%s) ...",
                 len(rows), parser.model_name)

        tasks = [
            _process_one(row["id"], row["raw_text"], parser, semaphore, switch_lock, db_url, counters)
            for row in rows
        ]
        await asyncio.gather(*tasks)

        log.info("Batch done — running total: processed=%d, failed=%d",
                 counters["ok"], counters["fail"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process pending raw opinions through the Gemini extraction API."
    )
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE,
                        help=f"Rows fetched per DB query (default: {_BATCH_SIZE}).")
    parser.add_argument("--concurrency", type=int, default=_CONCURRENCY,
                        help=f"Max simultaneous Gemini requests (default: {_CONCURRENCY}).")
    parser.add_argument("--retry-processing", action="store_true",
                        help="Also reprocess rows stuck in 'processing' state (from a prior crash).")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url is required (or set DATABASE_URL env var).")
        sys.exit(1)

    asyncio.run(run_worker(
        args.db_url,
        args.gemini_api_key,
        args.batch_size,
        args.concurrency,
        args.retry_processing,
    ))


if __name__ == "__main__":
    main()
