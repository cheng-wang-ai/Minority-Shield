"""
Data ingestion pipeline for Minority-Shield.

Reads raw court opinion text files from a directory, calls the Gemini parser
to extract structured metadata, and upserts results into PostgreSQL.

Usage:
    python scripts/ingest.py --opinions-dir ./opinions --db-url postgresql://...
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.gemini_parser import GeminiParser, CaseMetadata


def get_or_create_firm(cur, name: str) -> int:
    cur.execute(
        "INSERT INTO law_firms (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
        (name,),
    )
    return cur.fetchone()[0]


def insert_case(cur, metadata: CaseMetadata, plaintiff_id: int, defendant_id: int, source_text: str) -> int:
    cur.execute(
        """
        INSERT INTO cases (plaintiff_firm, defendant_firm, case_type, outcome, minority_focus, source_text)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (plaintiff_id, defendant_id, metadata.case_type, metadata.outcome, metadata.minority_focus, source_text),
    )
    return cur.fetchone()[0]


def ingest_directory(opinions_dir: str, db_url: str, api_key: str | None = None) -> None:
    parser = GeminiParser(api_key=api_key)
    conn = psycopg2.connect(db_url)

    opinion_files = list(Path(opinions_dir).glob("*.txt")) + list(Path(opinions_dir).glob("*.pdf"))
    if not opinion_files:
        print(f"No .txt or .pdf files found in {opinions_dir}")
        return

    print(f"Found {len(opinion_files)} opinion file(s).")
    success, failure = 0, 0

    with conn:
        with conn.cursor() as cur:
            for fpath in opinion_files:
                print(f"  Processing: {fpath.name} ...", end=" ", flush=True)
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    metadata = parser.parse(text)

                    p_id = get_or_create_firm(cur, metadata.plaintiff_firm)
                    d_id = get_or_create_firm(cur, metadata.defendant_firm)
                    case_id = insert_case(cur, metadata, p_id, d_id, text[:10_000])

                    print(f"OK (case_id={case_id}, type={metadata.case_type}, outcome={metadata.outcome})")
                    success += 1
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    failure += 1

    conn.close()
    print(f"\nIngestion complete: {success} succeeded, {failure} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest court opinions into Minority-Shield database.")
    parser.add_argument("--opinions-dir", required=True, help="Directory containing .txt or .pdf opinion files.")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"), help="PostgreSQL connection URL.")
    parser.add_argument("--gemini-api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var).")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url is required (or set DATABASE_URL env var).")
        sys.exit(1)

    ingest_directory(args.opinions_dir, args.db_url, api_key=args.gemini_api_key)


if __name__ == "__main__":
    main()
