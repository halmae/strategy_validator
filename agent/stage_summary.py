"""Stage handoff summaries used to compress session context."""
from __future__ import annotations

import json
from typing import Any

from .kg_store import KGState


_PLACEHOLDER_VALUES = {
    "unknown",
    "unk",
    "tbd",
    "n/a",
    "na",
    "not sure",
    "unsure",
    "to be determined",
    "to be decided",
}

_OBJECTIVES = {
    0: "Resolve the strategy type vector and scope eligibility.",
    1: "Establish a falsifiable edge hypothesis and the market inefficiency it exploits.",
    2: "Turn the Stage 1 hypothesis into a concrete decomposition plan and bias-check plan.",
}

_NEXT_STAGE_HANDOFF = {
    0: [
        "Resolve Stage 1 deferred fields such as universe scope and anchor.",
        "Pin down the edge claim, mechanism, falsification condition, and inefficiency persistence.",
    ],
    1: [
        "Resolve signal_source first because it activates Stage 2 checks.",
        "Define ReturnDecomposition.method and sample_period.",
        "For each active Stage 2 check, fill tests, evidence_type, criterion, and method.",
    ],
    2: [
        "Use the Stage 2 plan to drive signal-quality validation once Stage 3 exists.",
        "Evidence gates that remain pending should be revisited after running the planned analyses.",
    ],
}


def build_stage_summary(stage: int, schema: dict, kg_state: KGState) -> dict:
    summary = {
        "stage": stage,
        "name": schema.get("name", f"stage_{stage}"),
        "objective": _OBJECTIVES.get(stage, ""),
        "resolved_facts": _resolved_facts(stage, kg_state),
        "key_rationale": _key_rationale(stage, kg_state),
        "active_checks": list(kg_state.active_checks.get(f"stage_{stage}", [])),
        "gates": _stage_gate_summary(stage, kg_state),
        "open_questions": _open_questions(stage, kg_state),
        "next_stage_handoff": list(_NEXT_STAGE_HANDOFF.get(stage, [])),
    }
    return summary


def format_stage_summaries(stage_summaries: dict) -> str:
    if not stage_summaries:
        return "None recorded yet."

    def _stage_sort_key(item: tuple[str, Any]) -> int:
        key = item[0]
        if key.startswith("stage_"):
            try:
                return int(key.split("_", 1)[1])
            except ValueError:
                return 999
        return 999

    ordered = [value for _, value in sorted(stage_summaries.items(), key=_stage_sort_key)]
    return json.dumps(ordered, ensure_ascii=False, indent=2)


def _resolved_facts(stage: int, kg_state: KGState) -> dict:
    if stage == 0:
        facts = {"type_vector": dict(kg_state.type_vector)}
        if kg_state.out_of_scope:
            facts["out_of_scope_reason"] = kg_state.out_of_scope_reason
        return facts

    if stage == 1:
        return {
            "deferred": {
                key: value
                for key, value in kg_state.deferred.items()
                if key in ("universe_scope", "universe_anchor")
            },
            "edge": _meaningful_fields(
                kg_state.entities.get("Edge", {}),
                ("type", "direction", "horizon", "capacity"),
            ),
            "hypothesis": _meaningful_fields(
                kg_state.entities.get("Hypothesis", {}),
                ("claim", "mechanism", "falsifiable", "falsification_condition"),
            ),
            "market_inefficiency": _meaningful_fields(
                kg_state.entities.get("MarketInefficiency", {}),
                ("persistence", "structural_barrier", "decay_risk"),
            ),
        }

    if stage == 2:
        active_checks = kg_state.active_checks.get("stage_2", [])
        check_plans = {
            name: _meaningful_fields(
                kg_state.entities.get(name, {}),
                ("tests", "evidence_type", "criterion", "method"),
            )
            for name in active_checks
        }
        return {
            "signal_source": kg_state.deferred.get("signal_source"),
            "return_decomposition": _meaningful_fields(
                kg_state.entities.get("ReturnDecomposition", {}),
                (
                    "method",
                    "sample_period",
                    "market_beta_description",
                    "factor_components",
                    "residual_alpha",
                    "supports_hypothesis",
                ),
            ),
            "check_plans": check_plans,
        }

    return {}


def _key_rationale(stage: int, kg_state: KGState) -> list[str]:
    rationale: list[str] = []

    if stage == 1:
        rationale.extend(
            _collect_strings(
                kg_state.entities.get("Hypothesis", {}),
                ("claim", "mechanism", "falsification_condition"),
            )
        )
        rationale.extend(
            _collect_strings(
                kg_state.entities.get("MarketInefficiency", {}),
                ("persistence",),
            )
        )

    if stage == 2:
        rationale.extend(
            _collect_strings(
                kg_state.entities.get("ReturnDecomposition", {}),
                ("reasoning",),
            )
        )
        for name in kg_state.active_checks.get("stage_2", []):
            rationale.extend(
                _collect_strings(
                    kg_state.entities.get(name, {}),
                    ("evidence_summary", "reasoning"),
                )
            )

    rationale.extend(
        note["note"]
        for note in kg_state.notes
        if note.get("stage") == stage and _is_meaningful_value(note.get("note"))
    )

    return _dedupe_preserve_order(rationale)


def _open_questions(stage: int, kg_state: KGState) -> list[str]:
    gate_info = _stage_gate_summary(stage, kg_state)
    pending_or_failed = gate_info["pending"] + gate_info["failed"]
    return [item["reason"] for item in pending_or_failed if item.get("reason")]


def _stage_gate_summary(stage: int, kg_state: KGState) -> dict:
    prefix = f"G{stage}_"
    summary = {"passed": [], "pending": [], "failed": []}
    status_to_bucket = {
        "pass": "passed",
        "pending": "pending",
        "fail": "failed",
    }

    for gate_id, gate_state in sorted(kg_state.gates.items()):
        if not gate_id.startswith(prefix):
            continue
        status = gate_state.get("status", "pending")
        bucket = summary.get(status_to_bucket.get(status, ""))
        if bucket is None:
            continue
        bucket.append({"id": gate_id, "reason": gate_state.get("reason", "")})

    return summary


def _meaningful_fields(source: dict, field_names: tuple[str, ...]) -> dict:
    result = {}
    for field_name in field_names:
        value = source.get(field_name)
        if _is_meaningful_value(value):
            result[field_name] = value
    return result


def _collect_strings(source: dict, field_names: tuple[str, ...]) -> list[str]:
    values = []
    for field_name in field_names:
        value = source.get(field_name)
        if _is_meaningful_value(value):
            values.append(str(value).strip())
    return values


def _is_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False
        return normalized.lower() not in _PLACEHOLDER_VALUES
    return True


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
