"""
Async CourtListener fetcher for Minority-Shield.

Fetches Federal District Court civil opinions from the CourtListener REST API
and saves raw text to the `raw_opinions` staging table with status='pending'.
Fully decoupled from the Gemini worker — run independently.

Usage:
    python scripts/fetch_courtlistener.py [--limit 500] [--filed-after 2010-01-01]

Environment variables:
    DATABASE_URL              PostgreSQL connection URL
    COURTLISTENER_API_TOKEN   CourtListener API token (get one at courtlistener.com/profile/tokens/)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COURTLISTENER_BASE = "https://www.courtlistener.com"
SEARCH_PATH = "/api/rest/v3/search/"

# Federal district court slugs (covers the highest-volume courts).
# Full list: https://www.courtlistener.com/api/rest/v3/courts/?type=fd
FEDERAL_DISTRICT_COURTS = [
    "dcd", "nysd", "nyed", "cacd", "caed", "cand", "casd",
    "ilnd", "ilcd", "ilsd", "txsd", "txed", "txnd", "txwd",
    "flsd", "flmd", "flnd", "gamd", "gand", "gasd",
    "njd", "pad", "paed", "pamd", "pawd",
    "ohnd", "ohsd", "mied", "miwd", "mnd",
    "waed", "wawd", "ord", "cod", "azd",
]

# CourtListener free tokens: ~5k requests/day, burst limit enforced via 403.
# Stay conservative to avoid triggering rate limits.
_REQUESTS_PER_SECOND = 2
_CONCURRENCY = 2  # parallel in-flight requests


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    semaphore: asyncio.Semaphore,
    token: Optional[str],
    _retries: int = 4,
) -> dict:
    headers = {"Authorization": f"Token {token}"} if token else {}
    backoff = 5.0
    for attempt in range(_retries):
        async with semaphore:
            resp = await client.get(url, params=params, headers=headers, timeout=30.0)
            await asyncio.sleep(1.0 / _REQUESTS_PER_SECOND)
        if resp.status_code in (403, 429):
            wait = backoff * (2 ** attempt)
            log.warning("Rate limited (%s) — waiting %.0fs before retry %d/%d",
                        resp.status_code, wait, attempt + 1, _retries)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # raise after all retries exhausted


async def _fetch_opinion_text(
    client: httpx.AsyncClient,
    opinion_api_url: str,
    semaphore: asyncio.Semaphore,
    token: Optional[str],
) -> Optional[str]:
    """
    Fetch the full opinion text by hitting the opinion's REST API URL directly.
    The search results don't embed full text, so we follow the opinion URL to get it.
    """
    headers = {"Authorization": f"Token {token}"} if token else {}
    try:
        async with semaphore:
            resp = await client.get(opinion_api_url, headers=headers, timeout=30.0)
            await asyncio.sleep(1.0 / _REQUESTS_PER_SECOND)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception as exc:
        log.debug("Could not fetch opinion detail %s: %s", opinion_api_url, exc)
        return None

    # Prefer plain text; fall back to HTML variants
    for field in ("plain_text", "html_with_citations", "html_lawbox", "html"):
        text = (data.get(field) or "").strip()
        if len(text) > 200:
            return text[:200_000]

    # Last resort: download_url
    download_url = data.get("download_url")
    if download_url:
        try:
            async with semaphore:
                resp = await client.get(download_url, headers=headers, timeout=45.0, follow_redirects=True)
                await asyncio.sleep(1.0 / _REQUESTS_PER_SECOND)
            if resp.status_code == 200:
                return resp.text[:200_000]
        except Exception as exc:
            log.debug("Could not fetch download_url %s: %s", download_url, exc)

    return None


# ---------------------------------------------------------------------------
# Paginated opinion stream
# ---------------------------------------------------------------------------

async def search_stream(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    token: Optional[str],
    filed_after: str,
    limit: int,
) -> AsyncIterator[dict]:
    """
    Async generator that pages through the CourtListener search API,
    yielding one result dict per opinion up to `limit`.

    Each result contains: id, cluster_id, court_id, caseName, dateFiled,
    absolute_url, and the opinion API path we use to fetch full text.
    """
    params = {
        "type": "o",                                    # opinions
        "stat_Precedential": "on",
        "filed_after": filed_after,
        "court": " ".join(FEDERAL_DISTRICT_COURTS),     # space-separated for search API
        "order_by": "dateFiled desc",
        "page_size": 20,
    }
    url = COURTLISTENER_BASE + SEARCH_PATH
    yielded = 0

    while url and yielded < limit:
        try:
            data = await _get_json(client, url, params, semaphore, token)
        except httpx.HTTPStatusError as exc:
            log.error("HTTP %s fetching search page: %s", exc.response.status_code, url)
            break

        for result in data.get("results", []):
            if yielded >= limit:
                return
            yield result
            yielded += 1

        url = data.get("next")
        params = {}  # next URL already has params embedded


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _upsert_opinion(cur, cl_id: int, case_url: str, raw_text: str) -> bool:
    """Insert opinion; skip if courtlistener_id already exists. Returns True on insert."""
    cur.execute(
        """
        INSERT INTO raw_opinions (courtlistener_id, case_url, raw_text, status)
        VALUES (%s, %s, %s, 'pending')
        ON CONFLICT (courtlistener_id) DO NOTHING
        """,
        (cl_id, case_url, raw_text),
    )
    return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

async def run(db_url: str, token: Optional[str], limit: int, filed_after: str) -> None:
    conn = psycopg2.connect(db_url)
    inserted = skipped = failed = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        semaphore = asyncio.Semaphore(_CONCURRENCY)

        async for result in search_stream(client, semaphore, token, filed_after, limit):
            # Search results use 'id' as the opinion pk and expose the API path
            cl_id = result.get("id")
            case_url = COURTLISTENER_BASE + (result.get("absolute_url") or "")
            # Build the REST API URL to fetch full opinion detail
            opinion_api_url = f"{COURTLISTENER_BASE}/api/rest/v3/opinions/{cl_id}/"

            try:
                text = await _fetch_opinion_text(client, opinion_api_url, semaphore, token)
                if not text:
                    log.debug("No usable text for opinion %s — skipping.", cl_id)
                    skipped += 1
                    continue

                with conn:
                    with conn.cursor() as cur:
                        if _upsert_opinion(cur, cl_id, case_url, text):
                            inserted += 1
                            log.info("Saved opinion %s (%d chars)", cl_id, len(text))
                        else:
                            skipped += 1
                            log.debug("Opinion %s already in DB — skipped.", cl_id)

            except Exception as exc:
                log.error("Error on opinion %s: %s", cl_id, exc)
                failed += 1

    conn.close()
    log.info("Fetch complete — inserted: %d, skipped: %d, failed: %d", inserted, skipped, failed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CourtListener opinions into the raw_opinions staging table."
    )
    parser.add_argument("--limit", type=int, default=500,
                        help="Maximum number of opinions to fetch (default: 500).")
    parser.add_argument("--filed-after", default="2005-01-01",
                        help="Only fetch opinions filed after this date (YYYY-MM-DD).")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"),
                        help="PostgreSQL connection URL.")
    parser.add_argument("--cl-token", default=os.environ.get("COURTLISTENER_API_TOKEN"),
                        help="CourtListener API token.")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url is required (or set DATABASE_URL env var).")
        sys.exit(1)

    if not args.cl_token:
        log.warning(
            "No CourtListener API token set. Unauthenticated requests are heavily rate-limited. "
            "Get a free token at https://www.courtlistener.com/profile/tokens/"
        )

    asyncio.run(run(args.db_url, args.cl_token, args.limit, args.filed_after))


if __name__ == "__main__":
    main()
