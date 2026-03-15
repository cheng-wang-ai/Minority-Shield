"""
Ingest Harvard Caselaw Access Project (case.law) bulk zip archives into raw_opinions.

Each zip contains a data.jsonl file where every line is one case JSON object.
Archives are downloaded one at a time, streamed into the DB, then deleted —
so only one zip occupies disk at a time.

Usage:
    # Ingest volumes 1-50 of F.Supp.2d
    python scripts/ingest_caselaw_json.py \
        --base-url https://static.case.law/f-supp-2d \
        --volumes 1-50

    # Ingest specific volumes
    python scripts/ingest_caselaw_json.py \
        --base-url https://static.case.law/f-supp-3d \
        --volumes 1,2,3

    # Ingest from already-downloaded zip files
    python scripts/ingest_caselaw_json.py --zip-dir ~/Downloads/zips

Environment variables:
    DATABASE_URL   PostgreSQL connection URL
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

MIN_TEXT_LEN = 200  # characters — skip near-empty opinions


# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------

def _build_raw_text(data: dict) -> str:
    """Combine attorney listing + opinion text for Gemini extraction."""
    parts = []

    case_name = data.get("name", "")
    if case_name:
        parts.append(f"Case: {case_name}")

    attorneys = data.get("casebody", {}).get("attorneys", [])
    if attorneys:
        parts.append("Attorneys:")
        for a in attorneys:
            parts.append(f"  {a}")

    for opinion in data.get("casebody", {}).get("opinions", []):
        text = (opinion.get("text") or "").strip()
        if text:
            parts.append("\nOpinion:")
            parts.append(text)
            break  # majority opinion only

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _upsert_opinion(cur, cap_id: int, case_url: str, raw_text: str,
                    decision_date: str, court: str) -> bool:
    cur.execute(
        """
        INSERT INTO raw_opinions (courtlistener_id, case_url, raw_text, status, decision_date, court)
        VALUES (%s, %s, %s, 'pending', %s, %s)
        ON CONFLICT (courtlistener_id) DO NOTHING
        """,
        (cap_id, case_url, raw_text, decision_date or None, court or None),
    )
    return cur.rowcount == 1


# ---------------------------------------------------------------------------
# JSONL processing
# ---------------------------------------------------------------------------

def _process_jsonl(fileobj, conn, min_date: str) -> tuple[int, int]:
    """Stream through a data.jsonl file, inserting qualifying cases. Returns (inserted, skipped)."""
    inserted = skipped = 0

    for line in fileobj:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        # Date filter
        decision_date = data.get("decision_date", "")
        if decision_date and decision_date < min_date:
            skipped += 1
            continue

        cap_id = data.get("id")
        if not cap_id:
            skipped += 1
            continue

        raw_text = _build_raw_text(data)
        if len(raw_text) < MIN_TEXT_LEN:
            skipped += 1
            continue

        case_url = f"https://case.law/caselaw/?id={cap_id}"
        court = data.get("court", {}).get("name", "")

        try:
            with conn:
                with conn.cursor() as cur:
                    if _upsert_opinion(cur, cap_id, case_url, raw_text[:200_000],
                                       decision_date, court):
                        inserted += 1
                    else:
                        skipped += 1
        except Exception as exc:
            log.error("DB error on case %s: %s", cap_id, exc)
            skipped += 1

    return inserted, skipped


def _process_zip(zip_path: Path, conn, min_date: str) -> tuple[int, int]:
    """Open a zip archive and process all JSON files inside it."""
    inserted = skipped = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_names:
            log.warning("No JSON files found in %s", zip_path.name)
            return 0, 0

        for name in json_names:
            with zf.open(name) as f:
                try:
                    parsed = json.loads(f.read().decode("utf-8"))
                except Exception:
                    skipped += 1
                    continue

                items = parsed if isinstance(parsed, list) else [parsed]
                for data in items:
                    if not isinstance(data, dict):
                        skipped += 1
                        continue

                    decision_date = data.get("decision_date", "")
                    if decision_date and decision_date < min_date:
                        skipped += 1
                        continue

                    cap_id = data.get("id")
                    if not cap_id:
                        skipped += 1
                        continue

                    raw_text = _build_raw_text(data)
                    if len(raw_text) < MIN_TEXT_LEN:
                        skipped += 1
                        continue

                    case_url = f"https://case.law/caselaw/?id={cap_id}"
                    court = data.get("court", {}).get("name", "")
                    try:
                        with conn:
                            with conn.cursor() as cur:
                                if _upsert_opinion(cur, cap_id, case_url, raw_text[:200_000],
                                                   decision_date, court):
                                    inserted += 1
                                else:
                                    skipped += 1
                    except Exception as exc:
                        log.error("DB error on case %s: %s", cap_id, exc)
                        skipped += 1

    return inserted, skipped


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_zip(url: str, dest: Path) -> bool:
    """Download a zip to dest. Returns False on HTTP error."""
    log.info("Downloading %s ...", url)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            log.warning("Volume not found (404): %s", url)
            return False
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
        size_mb = dest.stat().st_size / 1e6
        log.info("Downloaded %.1f MB → %s", size_mb, dest.name)
        return True
    except Exception as exc:
        log.error("Failed to download %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Volume range parsing
# ---------------------------------------------------------------------------

def _parse_volumes(spec: str) -> list[int]:
    """Parse '1-50' or '1,2,3' into a list of ints."""
    volumes = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            volumes.extend(range(int(lo), int(hi) + 1))
        else:
            volumes.append(int(part))
    return volumes


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def ingest_from_urls(base_url: str, volumes: list[int], db_url: str, min_date: str) -> None:
    conn = psycopg2.connect(db_url)
    total_inserted = total_skipped = 0

    base_url = base_url.rstrip("/")

    for vol in volumes:
        url = f"{base_url}/{vol}.zip"

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if not _download_zip(url, tmp_path):
                continue

            log.info("Processing volume %d ...", vol)
            ins, skp = _process_zip(tmp_path, conn, min_date)
            total_inserted += ins
            total_skipped += skp
            log.info("Volume %d: inserted %d, skipped %d", vol, ins, skp)
        finally:
            tmp_path.unlink(missing_ok=True)

    conn.close()
    log.info("All done — total inserted: %d, total skipped: %d", total_inserted, total_skipped)


def ingest_from_dir(zip_dir: str, db_url: str, min_date: str) -> None:
    conn = psycopg2.connect(db_url)
    zips = sorted(Path(zip_dir).glob("*.zip"))

    if not zips:
        log.error("No zip files found in %s", zip_dir)
        conn.close()
        return

    log.info("Found %d zip file(s) in %s", len(zips), zip_dir)
    total_inserted = total_skipped = 0

    for zp in zips:
        log.info("Processing %s ...", zp.name)
        ins, skp = _process_zip(zp, conn, min_date)
        total_inserted += ins
        total_skipped += skp
        log.info("%s: inserted %d, skipped %d", zp.name, ins, skp)

    conn.close()
    log.info("All done — total inserted: %d, total skipped: %d", total_inserted, total_skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest case.law bulk zip archives into raw_opinions."
    )
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--filed-after", default="2000-01-01",
        help="Skip opinions decided before this date (default: 2000-01-01).",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--base-url",
        help="Base URL for zip archives, e.g. https://static.case.law/f-supp-2d",
    )
    source.add_argument(
        "--zip-dir",
        help="Directory of already-downloaded zip files.",
    )

    parser.add_argument(
        "--volumes",
        help="Volume range to download, e.g. '1-50' or '1,2,5'. Required with --base-url.",
    )

    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url is required (or set DATABASE_URL env var).")
        sys.exit(1)

    if args.base_url and not args.volumes:
        print("Error: --volumes is required when using --base-url.")
        sys.exit(1)

    if args.base_url:
        ingest_from_urls(args.base_url, _parse_volumes(args.volumes), args.db_url, args.filed_after)
    else:
        ingest_from_dir(args.zip_dir, args.db_url, args.filed_after)


if __name__ == "__main__":
    main()
