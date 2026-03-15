-- Add is_private_firm flag to law_firms.
-- TRUE  = private law firm (included in rankings)
-- FALSE = government agency, public office, or NGO (excluded from rankings)

ALTER TABLE law_firms
    ADD COLUMN IF NOT EXISTS is_private_firm BOOLEAN NOT NULL DEFAULT TRUE;
