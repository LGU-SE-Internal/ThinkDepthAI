"""Validate the synthesizer's RCA JSON output.

The synthesis node emits one JSON object describing root causes and
propagation. This module both extracts that JSON from the model's text
response and checks it against the v2 RCA schema, returning structured
errors that callers can either log or feed back to the LLM as a
correction prompt.

Error specificity matters: the repair loop in graph.py forwards
``errors`` to the model verbatim. A vague "could not find JSON" when
the real failure is "JSON started but never closed" wastes a retry, so
the extractor distinguishes empty / no-brace / never-closed /
decode-error / shape-error and surfaces each separately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class _ExtractResult:
    json_str: str | None
    reason: str | None


def _extract_json_from_text(text: str) -> _ExtractResult:
    if not text or not text.strip():
        return _ExtractResult(None, "Output was empty.")
    text = text.strip()

    for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted.startswith("{"):
                return _ExtractResult(extracted, None)

    first_brace = text.find("{")
    if first_brace == -1:
        return _ExtractResult(None, "Output contained no '{' — emit a single JSON object.")

    depth = 0
    in_string = False
    escape_next = False
    end_pos = -1
    for i in range(first_brace, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_pos = i
                break

    if end_pos != -1:
        return _ExtractResult(text[first_brace : end_pos + 1], None)

    return _ExtractResult(
        None,
        "JSON started but never closed — output appears truncated mid-stream "
        f"(brace-depth at end-of-text was {depth}). Likely cause: max_tokens "
        "cap. Re-emit a more concise version, e.g. shorter SQL strings.",
    )


@dataclass
class RCAValidationOutcome:
    """Outcome of validating an RCA synthesis output.

    ``retry_warranted`` is True when the envelope has no usable
    ``root_causes`` and the agent should be asked to repair and re-emit.
    It is False when at least one ``root_cause`` survived; warnings are
    still recorded on the envelope but the output is accepted.

    ``errors`` is a list of agent-actionable strings, formatted to be
    pasted directly into a follow-up prompt.
    """

    envelope: dict
    errors: list[str]
    retry_warranted: bool


def validate_rca_output(text: str) -> RCAValidationOutcome:
    empty = {"root_causes": [], "propagation": []}
    extract = _extract_json_from_text(text)
    if extract.json_str is None:
        assert extract.reason is not None
        return RCAValidationOutcome(
            envelope={**empty, "parse_error": extract.reason, "raw_output": (text or "")[:500]},
            errors=[extract.reason],
            retry_warranted=True,
        )

    try:
        parsed = json.loads(extract.json_str)
    except json.JSONDecodeError as e:
        actionable = (
            f"JSON decode error: {e}. Common causes: trailing commas, "
            "single quotes, unescaped quotes inside SQL strings."
        )
        return RCAValidationOutcome(
            envelope={**empty, "parse_error": f"JSON decode error: {e}", "raw_output": text[:500]},
            errors=[actionable],
            retry_warranted=True,
        )

    if not isinstance(parsed, dict):
        actionable = (
            "Top-level output must be a JSON object with keys "
            "`root_causes` and `propagation`. Yours was not an object."
        )
        return RCAValidationOutcome(
            envelope={**empty, "parse_error": "Output is not a JSON object", "raw_output": text[:500]},
            errors=[actionable],
            retry_warranted=True,
        )

    fatal: list[str] = []
    warnings: list[str] = []

    root_causes = parsed.get("root_causes")
    if not isinstance(root_causes, list) or not root_causes:
        fatal.append(
            "`root_causes` is missing or empty. Provide at least one root "
            "cause with `service`, `fault_kind`, and ≥1 `evidence` item."
        )
        root_causes = []

    cleaned: list[dict] = []
    for i, rc in enumerate(root_causes):
        if not isinstance(rc, dict):
            warnings.append(f"root_causes[{i}] is not an object")
            continue
        if not rc.get("service"):
            warnings.append(f"root_causes[{i}] missing `service`")
            continue
        if not rc.get("fault_kind"):
            warnings.append(f"root_causes[{i}] missing `fault_kind`")
            continue
        evidence = rc.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            warnings.append(f"root_causes[{i}] has no evidence")
            continue
        cleaned.append(rc)

    propagation = parsed.get("propagation")
    if not isinstance(propagation, list):
        propagation = []

    envelope: dict = {"root_causes": cleaned, "propagation": propagation}
    if warnings:
        envelope["parse_warning"] = "; ".join(warnings)

    if not cleaned:
        errors = list(fatal)
        if warnings:
            errors.append(f"Every root_cause was dropped: {'; '.join(warnings)}.")
        return RCAValidationOutcome(envelope=envelope, errors=errors, retry_warranted=True)

    return RCAValidationOutcome(envelope=envelope, errors=[], retry_warranted=False)
