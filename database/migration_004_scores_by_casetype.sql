-- Migration 004: Add case_type column to firm_scores so the AHPI model can store
-- per-case-type skill scores (S_{k,m}) instead of a single aggregate score.

ALTER TABLE firm_scores ADD COLUMN IF NOT EXISTS case_type TEXT DEFAULT 'overall';

-- Replace the old (run, firm) unique constraint with (run, firm, case_type).
ALTER TABLE firm_scores DROP CONSTRAINT IF EXISTS firm_scores_model_run_id_firm_id_key;
ALTER TABLE firm_scores ADD CONSTRAINT firm_scores_run_firm_ct_key
    UNIQUE (model_run_id, firm_id, case_type);

-- Rebuild the convenience view to expose case_type.
DROP VIEW IF EXISTS latest_firm_scores;
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
