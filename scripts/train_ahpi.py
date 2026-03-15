"""
Train the AHPI model from interactions stored in PostgreSQL and save results.

Usage:
    python scripts/train_ahpi.py --db-url postgresql://... [--q-factor 15] [--decay-lambda 0.1]
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.ahpi_engine import Interaction, fit, rank_firms


def load_interactions(cur) -> list[Interaction]:
    # Only include cases where BOTH sides are private law firms
    cur.execute("""
        SELECT
            p.name AS plaintiff_firm,
            d.name AS defendant_firm,
            c.case_type,
            c.outcome,
            c.decision_date
        FROM cases c
        JOIN law_firms p ON p.id = c.plaintiff_firm AND p.is_private_firm = TRUE
        JOIN law_firms d ON d.id = c.defendant_firm AND d.is_private_firm = TRUE
    """)
    return [
        Interaction(
            plaintiff_firm=row[0],
            defendant_firm=row[1],
            case_type=row[2],
            outcome=row[3],
            decision_date=row[4].isoformat() if row[4] else None,
        )
        for row in cur.fetchall()
    ]


def save_model(cur, model, interactions, q_factor, converged) -> int:
    cur.execute(
        """
        INSERT INTO model_runs (q_factor, converged, n_firms, n_interactions)
        VALUES (%s, %s, %s, %s) RETURNING id
        """,
        (q_factor, converged, len(model.scores), len(interactions)),
    )
    run_id = cur.fetchone()[0]

    # Insert one score row per (firm, case_type)
    for firm_name, ct_scores in model.scores.items():
        cur.execute("SELECT id FROM law_firms WHERE name = %s", (firm_name,))
        row = cur.fetchone()
        if row:
            firm_id = row[0]
            for case_type, score in ct_scores.items():
                cur.execute(
                    """
                    INSERT INTO firm_scores (model_run_id, firm_id, score, case_type)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (run_id, firm_id, score, case_type),
                )

    # Save case type params
    for ct, eps in model.epsilon.items():
        q_val = model.q.get(ct, 0.5)
        cur.execute(
            "INSERT INTO case_type_params (model_run_id, case_type, epsilon, q) VALUES (%s, %s, %s, %s)",
            (run_id, ct, eps, q_val),
        )

    return run_id


def _check_win_rates(interactions: list[Interaction]) -> None:
    """
    Log per-case-type plaintiff win rates. Warn if any case type is severely
    imbalanced (< 10% or > 90% plaintiff wins), which causes the optimizer to
    absorb unexplained losses into epsilon rather than firm skill scores.
    """
    from collections import defaultdict
    totals: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)
    for itx in interactions:
        totals[itx.case_type] += 1
        wins[itx.case_type] += itx.outcome

    print("Win rates by case type:")
    for ct in sorted(totals):
        n = totals[ct]
        rate = wins[ct] / n if n else 0.0
        print(f"  {ct:<15} {rate*100:5.1f}%  (N={n})")
        if rate < 0.10 or rate > 0.90:
            print(
                f"  WARNING: {ct} has a plaintiff win rate of {rate*100:.1f}% (N={n}). "
                f"Epsilon estimate for this case type may be unreliable. Consider "
                f"excluding it from the ranked output or increasing data collection "
                f"for {ct} cases."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AHPI model and persist results.")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--q-factor", type=int, default=30)
    parser.add_argument("--decay-lambda", type=float, default=0.1,
                        help="Time decay rate (years^-1). 0 disables decay.")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url required (or DATABASE_URL env var).")
        sys.exit(1)

    conn = psycopg2.connect(args.db_url)
    with conn:
        with conn.cursor() as cur:
            print("Loading interactions from database...")
            interactions = load_interactions(cur)
            print(f"  {len(interactions)} interactions loaded.")

            if not interactions:
                print("No data to train on.")
                sys.exit(0)

            _check_win_rates(interactions)
            print(f"Fitting AHPI model (decay_lambda={args.decay_lambda})...")
            model = fit(interactions, apply_filter=True, decay_lambda=args.decay_lambda)

            print("Saving model results...")
            run_id = save_model(cur, model, interactions, args.q_factor, converged=True)
            print(f"  Saved as model_run_id={run_id}")

    conn.close()

    print("\nTop 20 firms by AHPI score (Civil Rights):")
    for rank, (firm, score) in enumerate(rank_firms(model, case_type="Civil Rights", top_n=20), 1):
        print(f"  {rank:>3}. {firm:<50} {score:+.4f}")


if __name__ == "__main__":
    main()
