# Minority Shield

**Minority Shield** is a data-driven legal advocacy platform that ranks law firms for minority litigants in the United States. Rankings are derived from actual federal court outcomes — not firm prestige or peer surveys — using the **AHPI (Asymmetric Heterogeneous Pairwise Interactions)** algorithm, which models litigation as a competitive pairwise game and explicitly corrects for the structural defendant-side advantage in the US legal system.

---

## The problem

Existing law firm rankings (US News, Chambers, Martindale) are based on peer reputation surveys. Reputation correlates poorly with actual litigation performance, and virtually no objective performance data exists specifically for Civil Rights and Labor cases — the fields where representation quality most directly affects minority communities.

---

## How it works

### Data pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Harvard Caselaw Access Project (case.law)                  │
│  ~100k+ federal court opinions in bulk zip archives         │
└────────────────────────┬────────────────────────────────────┘
                         │  ingest_caselaw_json.py
                         │  streams zips one at a time (~50–200 MB peak)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL — raw_opinions table                            │
│  stores raw opinion text, status = pending                  │
└────────────────────────┬────────────────────────────────────┘
                         │  process_pending.py
                         │  async Gemini API calls (concurrent)
                         │  extracts: plaintiff firm, defendant firm,
                         │            case type, outcome, minority focus
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL — cases table                                   │
│  structured interaction records with decision_date + court  │
└────────────────────────┬────────────────────────────────────┘
                         │  train_ahpi.py
                         │  EM optimization (L-BFGS-B)
                         │  per-case-type scores S_{k,m}
                         │  time-decay weighting w_t = e^(−λΔt)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL — firm_scores + model_runs tables               │
│  latent skill score per firm per case type                  │
└────────────────────────┬────────────────────────────────────┘
                         │  streamlit run frontend/app.py
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit frontend                                         │
│  • Firm rankings filtered by case type + jurisdiction       │
│  • Head-to-head win probability predictor                   │
└─────────────────────────────────────────────────────────────┘
```

### AHPI algorithm

The AHPI algorithm was introduced in:

> Mojon, A., Mahari, R., & Lera, S. C. (2025). Data-driven law firm rankings to reduce information asymmetry in legal disputes. *Nature Computational Science*. https://doi.org/10.1038/s43588-025-00899-2

Every law firm $k$ receives a latent skill score $S_k$. Each case type $m$ has a defendant bias parameter $\epsilon_m$ fitted to the US legal system. The probability that plaintiff firm $A$ beats defendant firm $B$ in case type $m$ is:

$$\rho_n(A) = \frac{1}{1 + \exp(-(S_A - (S_B + \epsilon_m)))}$$

$$p_n(A) = q_m \cdot \rho_n(A) + (1 - q_m)(1 - \rho_n(A))$$

where $q_m > 0.85$ indicates that skill — not randomness — dominates outcomes. Parameters are estimated jointly via L-BFGS-B optimization over the full interaction network. Firms with fewer than 30 observed interactions (Q-filter) are excluded for statistical reliability. Recent cases are weighted more heavily via exponential time decay.

**Initialized defendant bias values** (fitted to the US legal system):

| Case Type | $\epsilon_m$ |
|---|---|
| Civil Rights | 2.03 |
| Labor | 1.99 |
| Other | 1.90 |
| Contracts | 1.66 |
| Torts | 0.32 |

---

## Data sources

Cases are sourced from the [Harvard Caselaw Access Project](https://case.law) bulk download API. Recommended reporter series:

| Series | Coverage | Notes |
|---|---|---|
| F. Supp. 3d | 2014–present | Primary source, district court |
| F. Supp. 2d | 1998–2014 | District court, start from vol ~150 for post-2000 |
| F.3d | 1993–present | Published circuit court opinions |
| F. App'x | 2001–present | Unpublished circuit court opinions, high volume |

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- [Gemini API key](https://aistudio.google.com) — paid tier recommended for large corpus runs

### 1. Install dependencies

```bash
git clone <repo-url>
cd Minority-Shield
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Create `.env` in the project root:

```
DATABASE_URL=postgresql://admin:password@localhost:5432/minority_shield
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start PostgreSQL

```bash
docker run -d \
  --name minority-shield-db \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=minority_shield \
  -p 5432:5432 \
  postgres:16
```

### 4. Apply schema and migrations

```bash
docker exec -i minority-shield-db psql -U admin -d minority_shield < database/schema.sql
docker exec -i minority-shield-db psql -U admin -d minority_shield < database/migration_001_raw_opinions.sql
docker exec -i minority-shield-db psql -U admin -d minority_shield < database/migration_002_firm_type.sql
docker exec -i minority-shield-db psql -U admin -d minority_shield < database/migration_003_date_court.sql
docker exec -i minority-shield-db psql -U admin -d minority_shield < database/migration_004_scores_by_casetype.sql
```

---

## Running the pipeline

### Step 1 — Ingest

```bash
# F. Supp. 3d — recommended starting point
python scripts/ingest_caselaw_json.py \
    --base-url https://static.case.law/f-supp-3d \
    --volumes 1-999

# F. Supp. 2d — skip early pre-2000 volumes
python scripts/ingest_caselaw_json.py \
    --base-url https://static.case.law/f-supp-2d \
    --volumes 150-999

# F.3d — published appellate opinions
python scripts/ingest_caselaw_json.py \
    --base-url https://static.case.law/f3d \
    --volumes 1-999

# From local zip files
python scripts/ingest_caselaw_json.py --zip-dir ~/Downloads/cap-zips
```

Monitor:
```bash
docker exec -i minority-shield-db psql -U admin -d minority_shield \
    -c "SELECT status, COUNT(*) FROM raw_opinions GROUP BY status;"
```

### Step 2 — Extract metadata (Gemini)

```bash
python scripts/process_pending.py --concurrency 5
```

| Flag | Default | Description |
|---|---|---|
| `--concurrency` | 5 | Simultaneous Gemini requests |
| `--batch-size` | 30 | Rows fetched per DB query |
| `--retry-processing` | off | Recover rows stuck from a prior crash |

The worker automatically falls back across models on quota exhaustion or quality degradation:
```
gemini-2.0-flash  →  gemini-2.5-flash
```

Recover rows that failed due to a transient error (e.g. model unavailability):
```bash
docker exec -i minority-shield-db psql -U admin -d minority_shield \
    -c "UPDATE raw_opinions SET status='pending', error_msg=NULL WHERE status='failed' AND error_msg LIKE '%404%';"
python scripts/process_pending.py --retry-processing --concurrency 5
```

### Step 3 — Train

```bash
python scripts/train_ahpi.py
```

| Flag | Default | Description |
|---|---|---|
| `--q-factor` | 30 | Minimum interactions per firm (Q-filter) |
| `--decay-lambda` | 0.1 | Time decay rate λ in years⁻¹ |

The script logs per-case-type plaintiff win rates before fitting and warns if any case type is severely imbalanced (< 10% or > 90%), which can cause epsilon to diverge.

Verify results:
```bash
docker exec -i minority-shield-db psql -U admin -d minority_shield \
    -c "SELECT firm_name, case_type, ROUND(score::numeric, 4)
        FROM latest_firm_scores
        ORDER BY case_type, score DESC LIMIT 30;"
```

### Step 4 — Launch

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`.

---

## Project structure

```
analytics/
  ahpi_engine.py             AHPI model: EM fitting, per-case-type scoring, prediction

api/
  gemini_parser.py           Gemini API client, model fallback chain, output validation

database/
  schema.sql                 Canonical schema
  migration_001_raw_opinions.sql
  migration_002_firm_type.sql
  migration_003_date_court.sql
  migration_004_scores_by_casetype.sql

frontend/
  app.py                     Streamlit UI: rankings + win probability predictor

scripts/
  ingest_caselaw_json.py     Download and stage CAP bulk zip archives
  process_pending.py         Async Gemini extraction worker
  train_ahpi.py              Fit AHPI model and persist results
```

---

## Data limitations

**Selection bias.** Rankings are derived from litigated-to-opinion cases only. Approximately 90% of federal civil cases settle before producing a written opinion and are not reflected in scores. The case count is displayed quantitatively in the frontend on every rankings page.

**Intake-selectivity bias.** A firm's score reflects both litigation skill and which cases it chose to accept. The pairwise structure partially controls for opponent strength — a win against a higher-ranked firm is weighted more heavily — but does not correct for intake selectivity. Legal aid organizations and public interest firms are excluded from rankings because the model systematically underrates them: they accept cases regardless of difficulty, which suppresses their apparent win rate.

**Binary outcome.** Outcomes are coded as plaintiff win (1) or defendant win (0). Damages amounts, partial wins, and settlements on remand are not captured.

---

## Citation

If you use this project or the underlying methodology, please cite the original paper:

```bibtex
@article{mojon2025datadriven,
  title   = {Data-driven law firm rankings to reduce information asymmetry in legal disputes},
  author  = {Mojon, Alexandre and Mahari, Robert and Lera, Sandro Claudio},
  journal = {Nature Computational Science},
  year    = {2025},
  doi     = {10.1038/s43588-025-00899-2},
  url     = {https://doi.org/10.1038/s43588-025-00899-2}
}
```
