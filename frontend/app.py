"""
Minority-Shield Frontend — Streamlit web app.

Provides:
  1. Firm search and ranking by AHPI score (filterable by case type / minority focus / court).
  2. Head-to-head win probability predictor.

Run:
    streamlit run frontend/app.py
"""

import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analytics.ahpi_engine import AHPIModel, predict_plaintiff_win, CASE_TYPES

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL environment variable is not set.")
        st.stop()
    return psycopg2.connect(db_url)


@st.cache_data(ttl=300)
def load_latest_scores(case_type: str = None, courts: tuple = ()) -> list[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        query = """
            SELECT lfs.firm_name, lfs.score, lfs.case_type, lfs.run_at
            FROM latest_firm_scores lfs
            JOIN law_firms lf ON lf.name = lfs.firm_name
            WHERE lf.is_private_firm = TRUE
        """
        params: list = []
        if case_type:
            query += " AND lfs.case_type = %s"
            params.append(case_type)
        if courts:
            query += """
                AND lfs.firm_name IN (
                    SELECT DISTINCT lf2.name FROM cases c2
                    JOIN law_firms lf2 ON (lf2.id = c2.plaintiff_firm OR lf2.id = c2.defendant_firm)
                    WHERE c2.court = ANY(%s)
                )
            """
            params.append(list(courts))
        query += " ORDER BY lfs.score DESC"
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


@st.cache_data(ttl=300)
def load_case_count() -> int:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM cases")
        return cur.fetchone()[0]


@st.cache_data(ttl=300)
def load_courts() -> list[str]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT court FROM cases WHERE court IS NOT NULL AND court <> '' ORDER BY court"
        )
        return [r[0] for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_model() -> AHPIModel:
    """Load the latest converged model parameters into an AHPIModel."""
    conn = get_conn()
    with conn.cursor() as cur:
        # Latest run id
        cur.execute("SELECT MAX(id) FROM model_runs WHERE converged = TRUE")
        row = cur.fetchone()
        if not row or row[0] is None:
            return AHPIModel()
        run_id = row[0]

        # Scores — one row per (firm, case_type); reconstruct nested dict
        cur.execute("""
            SELECT lf.name, fs.score, fs.case_type
            FROM firm_scores fs JOIN law_firms lf ON lf.id = fs.firm_id
            WHERE fs.model_run_id = %s
        """, (run_id,))
        scores: dict[str, dict[str, float]] = {}
        for firm_name, score, ct in cur.fetchall():
            if firm_name not in scores:
                scores[firm_name] = {}
            scores[firm_name][ct] = score

        # Params
        cur.execute("""
            SELECT case_type, epsilon, q FROM case_type_params WHERE model_run_id = %s
        """, (run_id,))
        epsilon = {}
        q = {}
        for r in cur.fetchall():
            epsilon[r[0]] = r[1]
            q[r[0]] = r[2]

    return AHPIModel(scores=scores, epsilon=epsilon, q=q)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Minority Shield",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Minority Shield")
st.caption(
    "Objective law firm rankings based on litigation performance — "
    "not prestige. Powered by the AHPI algorithm."
)

tab_rankings, tab_predict = st.tabs(["Firm Rankings", "Win Probability Predictor"])

# ---------------------------------------------------------------------------
# Tab 1: Firm Rankings
# ---------------------------------------------------------------------------

with tab_rankings:
    st.subheader("Law Firm Rankings by AHPI Score")
    st.markdown(
        "Scores reflect relative litigation skill estimated from actual federal case outcomes. "
        "Higher scores indicate better plaintiff-side performance after controlling for defendant bias."
    )

    n_cases = load_case_count()
    st.info(
        f"**Data scope:** Rankings are based on **{n_cases:,}** litigated-to-opinion cases. "
        "Approximately 90% of federal civil cases settle before producing a written opinion "
        "and are not reflected in these scores.",
        icon="ℹ️",
    )
    st.warning(
        "**Private firms only:** These rankings cover private law firms. "
        "Legal aid organizations and public interest firms are excluded because "
        "they typically accept cases regardless of difficulty, which causes the "
        "model to systematically underrate them — making direct score comparisons "
        "misleading. If you are seeking legal aid, these rankings do not apply.",
        icon="⚠️",
    )

    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        selected_case_type = st.selectbox(
            "Case Type", ["All"] + CASE_TYPES, key="rank_case_type"
        )
    with col_filter2:
        courts = load_courts()
        selected_courts = st.multiselect(
            "Filter by Court", courts, key="rank_courts"
        )

    ct_param = None if selected_case_type == "All" else selected_case_type
    scores = load_latest_scores(
        case_type=ct_param,
        courts=tuple(selected_courts),
    )

    if not scores:
        st.warning("No model results found. Run `scripts/train_ahpi.py` first.")
    else:
        import pandas as pd

        df = pd.DataFrame(scores)
        df["Rank"] = range(1, len(df) + 1)
        df = df.rename(columns={"firm_name": "Law Firm", "score": "AHPI Score", "case_type": "Case Type"})
        df = df[["Rank", "Law Firm", "Case Type", "AHPI Score"]]
        df["AHPI Score"] = df["AHPI Score"].map(lambda x: f"{x:+.4f}")

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} qualified firms (Q ≥ 30 interactions).")

# ---------------------------------------------------------------------------
# Tab 2: Head-to-Head Predictor
# ---------------------------------------------------------------------------

with tab_predict:
    st.subheader("Head-to-Head Win Probability")
    st.markdown(
        "Select a plaintiff firm, defendant firm, and case type to estimate the "
        "predicted win probability based on AHPI skill differentials."
    )

    model = load_model()

    if not model.scores:
        st.warning("No model loaded. Run `scripts/train_ahpi.py` first.")
    else:
        all_zero = all(
            score == 0.0
            for ct_scores in model.scores.values()
            for score in ct_scores.values()
        )
        if all_zero:
            st.warning(
                "Model scores are all zero — the AHPI model has not been trained "
                "on real data yet. Run `scripts/train_ahpi.py` to generate scores. "
                "Win probability estimates are not meaningful until training is complete.",
                icon="⚠️",
            )

        # Case type must be selected first — firm list is filtered to only firms
        # with an observed, non-zero score for the chosen case type.
        case_type = st.selectbox("Case Type", CASE_TYPES, key="pred_case_type")

        firms_for_ct = sorted([
            firm for firm, ct_scores in model.scores.items()
            if ct_scores.get(case_type, 0.0) != 0.0
        ])

        if len(firms_for_ct) < 2:
            st.warning(
                f"Not enough ranked firms for {case_type} — try a different case type "
                "or ingest more cases of this type."
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                plaintiff = st.selectbox("Plaintiff Firm", firms_for_ct, key="pred_plaintiff")
            with col2:
                defendant = st.selectbox(
                    "Defendant Firm",
                    [f for f in firms_for_ct if f != plaintiff],
                    key="pred_defendant",
                )

        if len(firms_for_ct) >= 2 and st.button("Calculate", type="primary"):
            try:
                result = predict_plaintiff_win(model, plaintiff, defendant, case_type)
                win_pct = result["win_probability"] * 100
                prop_pct = result["propensity"] * 100

                st.metric("Predicted Win Probability", f"{win_pct:.1f}%")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Plaintiff AHPI Score", f"{result['plaintiff_score']:+.4f}")
                col_b.metric("Defendant AHPI Score", f"{result['defendant_score']:+.4f}")
                col_c.metric("Defendant Bias (ε)", f"{result['defendant_bias']:.3f}")

                st.progress(int(win_pct))

                if win_pct >= 60:
                    st.success(f"Favorable matchup for plaintiff ({win_pct:.1f}% win probability).")
                elif win_pct >= 45:
                    st.info(f"Competitive matchup ({win_pct:.1f}% win probability).")
                else:
                    st.warning(
                        f"Challenging matchup for plaintiff ({win_pct:.1f}% win probability). "
                        "Consider firms with higher AHPI scores."
                    )
            except ValueError as e:
                st.error(str(e))
