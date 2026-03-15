"""
Classify law firms as private firms vs. government agencies / public offices / NGOs.

Sets is_private_firm=FALSE for any firm matching known non-private patterns.
Run after each training cycle or whenever new firms are added.

Usage:
    python scripts/classify_firms.py [--dry-run]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Keyword patterns that identify non-private entities
# ---------------------------------------------------------------------------

# Regex patterns matched case-insensitively against firm name.
# A firm matching ANY pattern is classified as non-private.
NON_PRIVATE_PATTERNS = [
    # Federal government agencies and commissions
    r"\bfederal\b",
    r"\bu\.?s\.?\s+(attorney|government|department|district|securities|trade|equal\s+employment)",
    r"\bunited\s+states\b",
    r"\binternal\s+revenue\b",
    r"\bdepartment\s+of\b",
    r"\bcommission\b",
    r"\bnational\s+labor\s+relations\b",
    r"\bsecurities\s+(and|&)\s+exchange\b",
    r"\bfederal\s+trade\b",
    r"\benvironmental\s+protection\s+agency\b",
    r"\bequal\s+employment\s+opportunity\b",
    r"\bnational\s+labor\b",
    r"\boffice\s+of\b",
    r"\bbureau\s+of\b",
    r"\badministration\b",

    # State and local government
    r"\battorney\s+general\b",
    r"\bdistrict\s+attorney\b",
    r"\bstate\s+of\b",
    r"\bcity\s+of\b",
    r"\bcounty\s+of\b",
    r"\bmunicipality\b",
    r"\bdepartment\s+of\s+(justice|labor|education|health|corrections)\b",
    r"\bstate\s+attorney\b",
    r"\bpublic\s+safety\b",

    # Public defenders and legal aid
    r"\bpublic\s+defender\b",
    r"\blegal\s+aid\b",
    r"\blegal\s+services\b",
    r"\bpro\s+bono\b",
    r"\bpublic\s+counsel\b",
    r"\bpublic\s+interest\s+law\b",

    # NGOs, advocacy groups, non-profits
    r"\bamerican\s+civil\s+liberties\s+union\b",
    r"\baclu\b",
    r"\bearthjustice\b",
    r"\bcitizens\s+for\b",
    r"\bproject\s+on\b",
    r"\bnational\s+association\s+for\b",
    r"\bnaacp\b",
    r"\badvocacy\b",
    r"\bfoundation\b",
    r"\binstitute\s+for\b",
    r"\bcenter\s+for\b",
    r"\bcenter\s+on\b",
    r"\bcouncil\s+on\b",
    r"\bcoalition\b",
    r"\balliance\b",
    r"\bunion\s+of\b",
    r"\bworkers\s+united\b",

    # Misc non-law-firm identifiers
    r"\bpro\s+se\b",
    r"\bunknown\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in NON_PRIVATE_PATTERNS]


def is_private_firm(name: str) -> bool:
    """Return False if the firm name matches any non-private pattern."""
    for pattern in _COMPILED:
        if pattern.search(name):
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_url: str, dry_run: bool) -> None:
    conn = psycopg2.connect(db_url)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, name FROM law_firms")
        firms = cur.fetchall()

    private_ids   = []
    nonprivate_ids = []

    for firm in firms:
        if is_private_firm(firm["name"]):
            private_ids.append(firm["id"])
        else:
            nonprivate_ids.append(firm["id"])
            print(f"  [excluded] {firm['name']}")

    print(f"\nTotal firms:   {len(firms)}")
    print(f"Private firms: {len(private_ids)}")
    print(f"Excluded:      {len(nonprivate_ids)}")

    if dry_run:
        print("\n[dry-run] No changes written.")
        conn.close()
        return

    with conn:
        with conn.cursor() as cur:
            if private_ids:
                cur.execute(
                    "UPDATE law_firms SET is_private_firm = TRUE WHERE id = ANY(%s)",
                    (private_ids,),
                )
            if nonprivate_ids:
                cur.execute(
                    "UPDATE law_firms SET is_private_firm = FALSE WHERE id = ANY(%s)",
                    (nonprivate_ids,),
                )

    print("\nClassification saved to database.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify law firms as private or non-private.")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print classifications without writing to the database.")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: DATABASE_URL not set.")
        sys.exit(1)

    run(args.db_url, args.dry_run)


if __name__ == "__main__":
    main()
