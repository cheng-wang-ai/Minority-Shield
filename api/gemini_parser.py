"""
Gemini API integration for extracting structured metadata from US Federal Court Opinions.

Input:  Raw text or PDF content of a court opinion.
Output: JSON with plaintiff_firm, defendant_firm, case_type, outcome, minority_focus.
"""

import json
import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

from google import genai
from google.genai import types
from pydantic import BaseModel, field_validator, ValidationError


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

CASE_TYPES = ["Civil Rights", "Contracts", "Labor", "Torts", "Other"]


class CaseMetadata(BaseModel):
    plaintiff_firm: str
    defendant_firm: str
    case_type: str
    outcome: int          # 1 = plaintiff wins, 0 = defendant wins
    minority_focus: bool  # True if case involves minority rights / Title VII / discrimination / language access

    @field_validator("case_type")
    @classmethod
    def validate_case_type(cls, v: str) -> str:
        if v not in CASE_TYPES:
            raise ValueError(f"case_type must be one of {CASE_TYPES}, got '{v}'")
        return v

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {v}")
        return v


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a legal data extraction specialist. Analyze the court opinion below and extract the following structured information.

Return ONLY a JSON object with these exact fields:
{{
  "plaintiff_firm": "<full legal name of law firm representing the plaintiff>",
  "defendant_firm": "<full legal name of law firm representing the defendant>",
  "case_type": "<one of: Civil Rights, Contracts, Labor, Torts, Other>",
  "outcome": <1 if plaintiff won, 0 if defendant won>,
  "minority_focus": <true if the case specifically involves minority rights, Title VII, racial/ethnic/national origin discrimination, or language access issues; false otherwise>
}}

Rules:
- If a party is self-represented (pro se), use "Pro Se" as the firm name.
- If the firm name cannot be determined, use "Unknown".
- case_type must be exactly one of the five listed values.
- outcome must be 1 (plaintiff victory) or 0 (defendant victory). If the outcome is a partial win or remand, use 0.
- minority_focus is true ONLY if the case explicitly involves minority civil rights, not general employment or contract disputes.

Court Opinion:
\"\"\"
{opinion_text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Model fallback chain — tried in order when quota (429) is exhausted
# or when a model's quality failure rate exceeds the threshold.
# Ordered from fastest/cheapest to most capable.
# ---------------------------------------------------------------------------

FALLBACK_MODELS = [
    "gemini-2.0-flash",   # Gemini 2.0 Flash — primary
    "gemini-2.5-flash",   # Gemini 2.5 Flash — fallback
]

# Minimum attempts before evaluating a model's failure rate
_MIN_ATTEMPTS_FOR_EVAL = 10
# Abandon a model if its non-429 failure rate exceeds this threshold
_MAX_FAILURE_RATE = 0.6


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GeminiParser:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key to GeminiParser."
            )
        self.client = genai.Client(api_key=key)

        # Build rotation list starting from the requested model.
        # If the model is in FALLBACK_MODELS, use it as the starting point
        # so we still fall back to the heavier models in order.
        if model_name in FALLBACK_MODELS:
            self._models = FALLBACK_MODELS[FALLBACK_MODELS.index(model_name):]
        else:
            # Custom model: prepend it, then fall through to the standard chain
            self._models = [model_name] + FALLBACK_MODELS

        self._model_idx = 0
        self.model_name = self._models[0]

        # Per-model quality stats: {model_name: {"attempts": int, "failures": int}}
        # Only non-429 failures (bad output, validation errors) count toward failure rate.
        self._stats: dict[str, dict] = {}

    def record_result(self, success: bool) -> None:
        """
        Record the outcome of a completed request for the current model.
        Pass success=False for quality failures (bad JSON, Unknown firms, etc.).
        Do NOT call this for 429 errors — those are quota issues, not quality issues.
        """
        s = self._stats.setdefault(self.model_name, {"attempts": 0, "failures": 0})
        s["attempts"] += 1
        if not success:
            s["failures"] += 1

    def is_current_model_reliable(self) -> bool:
        """
        Return False if the current model's quality failure rate exceeds the threshold.
        Returns True if there is not yet enough data to judge.
        """
        s = self._stats.get(self.model_name, {"attempts": 0, "failures": 0})
        if s["attempts"] < _MIN_ATTEMPTS_FOR_EVAL:
            return True
        return (s["failures"] / s["attempts"]) < _MAX_FAILURE_RATE

    def switch_to_next_model(self) -> bool:
        """
        Advance to the next model in the fallback chain.
        Skips any model already deemed unreliable by quality stats.
        Returns True if a new model is available, False if the chain is exhausted.
        """
        while self._model_idx + 1 < len(self._models):
            self._model_idx += 1
            self.model_name = self._models[self._model_idx]
            # Skip models already known to be unreliable
            if self.is_current_model_reliable():
                return True
            log.warning("Skipping unreliable model: %s", self.model_name)
        return False

    async def parse_async(self, opinion_text: str) -> CaseMetadata:
        """Async version of parse() for concurrent processing."""
        prompt = EXTRACTION_PROMPT.format(opinion_text=opinion_text[:15_000])

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        raw = _strip_markdown_fences(raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {e}\nRaw response:\n{raw}") from e

        try:
            return CaseMetadata(**data)
        except ValidationError as e:
            raise ValueError(f"Model output failed schema validation: {e}\nData: {data}") from e

    def parse(self, opinion_text: str) -> CaseMetadata:
        """
        Extract structured metadata from a raw court opinion.

        Returns:
            Validated CaseMetadata instance.

        Raises:
            ValueError: If the model response cannot be parsed or validated.
        """
        prompt = EXTRACTION_PROMPT.format(opinion_text=opinion_text[:15_000])

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        raw = _strip_markdown_fences(raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {e}\nRaw response:\n{raw}") from e

        try:
            return CaseMetadata(**data)
        except ValidationError as e:
            raise ValueError(f"Model output failed schema validation: {e}\nData: {data}") from e

    def parse_batch(self, opinions: list[str]) -> list[CaseMetadata | Exception]:
        """
        Parse a list of opinions, returning CaseMetadata or the Exception on failure.
        """
        results = []
        for opinion in opinions:
            try:
                results.append(self.parse(opinion))
            except Exception as exc:
                results.append(exc)
        return results


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()
