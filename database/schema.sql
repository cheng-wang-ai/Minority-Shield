-- Minority-Shield PostgreSQL Schema
-- Stores law firms, case interactions, and AHPI model results.

-- ---------------------------------------------------------------------------
-- Case type enum
-- ---------------------------------------------------------------------------
CREATE TYPE case_type_enum AS ENUM (
    'Civil Rights',
    'Contracts',
    'Labor',
    'Torts',
    'Other'
);

-- ---------------------------------------------------------------------------
-- Law firms
-- ---------------------------------------------------------------------------
CREATE TABLE law_firms (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_law_firms_name ON law_firms (name);

-- ---------------------------------------------------------------------------
-- Case interactions (one row per resolved federal case)
-- ---------------------------------------------------------------------------
CREATE TABLE cases (
    id               SERIAL PRIMARY KEY,
    plaintiff_firm   INTEGER NOT NULL REFERENCES law_firms(id),
    defendant_firm   INTEGER NOT NULL REFERENCES law_firms(id),
    case_type        case_type_enum NOT NULL,
    outcome          SMALLINT NOT NULL CHECK (outcome IN (0, 1)),  -- 1 = plaintiff wins
    minority_focus   BOOLEAN NOT NULL DEFAULT FALSE,
    source_text      TEXT,      -- raw opinion text or excerpt
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decision_date    DATE,
    court            TEXT,
    CONSTRAINT chk_different_firms CHECK (plaintiff_firm <> defendant_firm)
);

CREATE INDEX idx_cases_plaintiff ON cases (plaintiff_firm);
CREATE INDEX idx_cases_defendant ON cases (defendant_firm);
CREATE INDEX idx_cases_type      ON cases (case_type);
CREATE INDEX idx_cases_minority  ON cases (minority_focus);

-- ---------------------------------------------------------------------------
-- AHPI model runs (one row per training run)
-- ---------------------------------------------------------------------------
CREATE TABLE model_runs (
    id             SERIAL PRIMARY KEY,
    run_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    q_factor       INTEGER NOT NULL DEFAULT 30,
    converged      BOOLEAN NOT NULL DEFAULT FALSE,
    n_firms        INTEGER,
    n_interactions INTEGER,
    notes          TEXT
);

-- ---------------------------------------------------------------------------
-- Firm skill scores (output of AHPI EM algorithm)
-- ---------------------------------------------------------------------------
CREATE TABLE firm_scores (
    id           SERIAL PRIMARY KEY,
    model_run_id INTEGER NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    firm_id      INTEGER NOT NULL REFERENCES law_firms(id),
    score        DOUBLE PRECISION NOT NULL,
    case_type    TEXT NOT NULL DEFAULT 'overall',
    UNIQUE (model_run_id, firm_id, case_type)
);

CREATE INDEX idx_firm_scores_run  ON firm_scores (model_run_id);
CREATE INDEX idx_firm_scores_firm ON firm_scores (firm_id);

-- ---------------------------------------------------------------------------
-- Case-type parameters (epsilon and q per case type per model run)
-- ---------------------------------------------------------------------------
CREATE TABLE case_type_params (
    id           SERIAL PRIMARY KEY,
    model_run_id INTEGER NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    case_type    case_type_enum NOT NULL,
    epsilon      DOUBLE PRECISION NOT NULL,  -- defendant bias
    q            DOUBLE PRECISION NOT NULL,  -- valence probability
    UNIQUE (model_run_id, case_type)
);

-- ---------------------------------------------------------------------------
-- Convenience view: latest scores per firm
-- ---------------------------------------------------------------------------
CREATE VIEW latest_firm_scores AS
SELECT
    lf.name     AS firm_name,
    fs.score,
    fs.case_type,
    mr.run_at,
    mr.id       AS model_run_id
FROM firm_scores fs
JOIN law_firms  lf ON lf.id = fs.firm_id
JOIN model_runs mr ON mr.id = fs.model_run_id
WHERE mr.id = (SELECT MAX(id) FROM model_runs WHERE converged = TRUE);

-- ---------------------------------------------------------------------------
-- Convenience view: interaction counts per firm (for Q-factor filtering)
-- ---------------------------------------------------------------------------
CREATE VIEW firm_interaction_counts AS
SELECT
    lf.name  AS firm_name,
    lf.id    AS firm_id,
    COUNT(*) AS total_interactions
FROM (
    SELECT plaintiff_firm AS firm_id FROM cases
    UNION ALL
    SELECT defendant_firm AS firm_id FROM cases
) sub
JOIN law_firms lf ON lf.id = sub.firm_id
GROUP BY lf.id, lf.name;
