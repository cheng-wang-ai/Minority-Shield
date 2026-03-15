-- Migration 001: Add raw_opinions staging table for async CourtListener pipeline
-- Run this against your database after applying schema.sql

CREATE TABLE IF NOT EXISTS raw_opinions (
    id                  SERIAL PRIMARY KEY,
    courtlistener_id    INTEGER NOT NULL UNIQUE,   -- CourtListener opinion pk
    case_url            TEXT,                       -- canonical URL for reference
    raw_text            TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_msg           TEXT,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at        TIMESTAMPTZ
);

CREATE INDEX idx_raw_opinions_status ON raw_opinions (status);
CREATE INDEX idx_raw_opinions_fetched ON raw_opinions (fetched_at);
