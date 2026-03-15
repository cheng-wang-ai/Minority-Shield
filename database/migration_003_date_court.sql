-- Migration 003: Add decision_date and court to cases and raw_opinions.
-- Enables time-decay weighting in AHPI and jurisdiction filtering in the frontend.

ALTER TABLE cases ADD COLUMN IF NOT EXISTS decision_date DATE;
ALTER TABLE cases ADD COLUMN IF NOT EXISTS court TEXT;
CREATE INDEX IF NOT EXISTS idx_cases_date  ON cases (decision_date);
CREATE INDEX IF NOT EXISTS idx_cases_court ON cases (court);

-- Also add to raw_opinions so the ingest pipeline can preserve this metadata.
ALTER TABLE raw_opinions ADD COLUMN IF NOT EXISTS decision_date DATE;
ALTER TABLE raw_opinions ADD COLUMN IF NOT EXISTS court TEXT;
