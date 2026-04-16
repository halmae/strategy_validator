"""Deterministic runtime helpers for stages 0 through 2.

The YAML files describe the workflow, but stage-critical rules for the
implemented scope (classification + edge existence) are enforced here so the
runtime does not rely purely on model compliance.
"""
from __future__ import annotations

import re
from typing import Any

from .kg_store import KGState


class RuntimeValidationError(ValueError):
    """Raised when a model tool call violates the stage runtime contract."""


_TRUE_VALUES = {"true", "yes", "y", "1"}
_FALSE_VALUES = {"false", "no", "n", "0"}
_PLACEHOLDER_STRINGS = {
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


def sync_runtime_state(stage: int, schema: dict, routing: dict, kg_state: KGState) -> None:
    """Synchronize deterministic stage state after any KG mutation."""
    if stage == 0:
        _sync_stage0_scope_status(schema, kg_state)
        return

    if stage == 1:
        _sync_stage_1_checks(schema, routing, kg_state)
        _sync_stage_1_gates(schema, kg_state)
        return

    if stage == 2:
        _sync_stage_2_checks(routing, kg_state)
        _sync_stage_2_gates(schema, kg_state)


def validate_type_vector_update(schema: dict, dimension: str, value: Any) -> str:
    dimensions = schema.get("dimensions", {})
    if dimension not in dimensions:
        raise RuntimeValidationError(f"Unknown type_vector dimension: {dimension}")
    return _normalize_dimension_value(
        dimensions[dimension],
        value,
        label=f"type_vector.{dimension}",
    )


def validate_deferred_update(schema: dict, dimension: str, value: Any) -> str:
    deferred = schema.get("resolves_deferred", {})
    if dimension not in deferred:
        raise RuntimeValidationError(
            f"Unknown deferred dimension for this stage: {dimension}"
        )
    return _normalize_dimension_value(
        deferred[dimension],
        value,
        label=f"deferred.{dimension}",
    )


def validate_entity_update(
    schema: dict,
    entity: str,
    prop: str,
    value: Any,
    kg_state: KGState | None = None,
    stage: int | None = None,
) -> Any:
    entities = schema.get("entities", {})
    entity_schema = entities.get(entity)

    # Stage 2: if `entity` is not explicitly declared but IS an activated check
    # name, fall back to the generic `Check` template schema.
    if entity_schema is None and stage == 2 and kg_state is not None:
        active = set(kg_state.active_checks.get("stage_2", []))
        if entity in active and "Check" in entities:
            entity_schema = entities["Check"]

    if entity_schema is None:
        raise RuntimeValidationError(f"Unknown entity for this stage: {entity}")

    prop_schema = entity_schema.get("properties", {}).get(prop)
    if prop_schema is None:
        raise RuntimeValidationError(f"Unknown property: {entity}.{prop}")

    expected_type = prop_schema.get("type", "str")
    if expected_type == "enum":
        normalized = _normalize_string(value, label=f"{entity}.{prop}")
        allowed = prop_schema.get("values", [])
        if normalized not in allowed:
            raise RuntimeValidationError(
                f"Invalid value for {entity}.{prop}: {normalized!r}. "
                f"Allowed: {allowed}"
            )
        return normalized

    if expected_type == "bool":
        return _normalize_bool(value, label=f"{entity}.{prop}")

    if expected_type == "str":
        return _normalize_string(value, label=f"{entity}.{prop}")

    raise RuntimeValidationError(
        f"Unsupported property type for {entity}.{prop}: {expected_type}"
    )


def validate_gate_update(stage: int, schema: dict, kg_state: KGState, gate_id: str) -> None:
    allowed = set(_base_gate_ids(schema))
    allowed.update(_active_gate_ids_for_stage(stage, schema, kg_state))
    if gate_id not in allowed:
        raise RuntimeValidationError(f"Unknown gate for stage {stage}: {gate_id}")


def attempt_stage_advance(
    schema: dict,
    routing: dict,
    kg_state: KGState,
    from_stage: int,
    to_stage: int,
    max_implemented_stage: int,
) -> str:
    if from_stage != kg_state.stage:
        raise RuntimeValidationError(
            f"Stage transition mismatch: current stage is {kg_state.stage}, "
            f"but advance requested from {from_stage}"
        )

    if to_stage != from_stage + 1:
        raise RuntimeValidationError(
            f"Only sequential stage transitions are allowed: {from_stage} -> {to_stage}"
        )

    sync_runtime_state(kg_state.stage, schema, routing, kg_state)

    if from_stage == 0:
        _ensure_stage_0_ready(schema, kg_state)
    elif from_stage == 1:
        _ensure_stage_1_ready(schema, kg_state)
    elif from_stage == 2:
        _ensure_stage_2_ready(schema, kg_state)
    else:
        raise RuntimeValidationError(f"Stage {from_stage} is not supported yet")

    kg_state.mark_stage_completed(from_stage)

    if to_stage > max_implemented_stage:
        kg_state.workflow_complete = True
        return (
            f"Stage {from_stage} complete. "
            f"Stage {to_stage} is not implemented yet."
        )

    kg_state.advance_stage(to_stage)
    kg_state.workflow_complete = False
    return f"Stage {from_stage} -> {to_stage}"


def _normalize_dimension_value(dimension_schema: dict, value: Any, label: str) -> str:
    normalized = _normalize_string(value, label=label)
    values = dimension_schema.get("values")

    if isinstance(values, dict):
        if normalized in values:
            return normalized
        if dimension_schema.get("extensible"):
            return normalized
        raise RuntimeValidationError(
            f"Invalid value for {label}: {normalized!r}. "
            f"Allowed: {list(values.keys())}"
        )

    if isinstance(values, list):
        if normalized in values:
            return normalized
        raise RuntimeValidationError(
            f"Invalid value for {label}: {normalized!r}. Allowed: {values}"
        )

    return normalized


def _normalize_string(value: Any, label: str) -> str:
    if value is None:
        raise RuntimeValidationError(f"{label} cannot be null")
    normalized = str(value).strip()
    if not normalized:
        raise RuntimeValidationError(f"{label} cannot be empty")
    return normalized


def _normalize_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value

    normalized = _normalize_string(value, label=label).lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise RuntimeValidationError(
        f"{label} must be a boolean-like value ('true'/'false')"
    )


def _sync_stage0_scope_status(schema: dict, kg_state: KGState) -> None:
    execution_mode = kg_state.type_vector.get("execution_mode")
    policy = (
        schema.get("dimensions", {})
        .get("execution_mode", {})
        .get("scope_policy", {})
    )
    out_of_scope = set(policy.get("out_of_scope", []))

    if execution_mode in out_of_scope:
        kg_state.out_of_scope = True
        kg_state.out_of_scope_reason = policy.get(
            "on_detect",
            policy.get("reason", "This strategy is out of scope."),
        ).strip()
        return

    kg_state.out_of_scope = False
    kg_state.out_of_scope_reason = ""


def _sync_stage_1_checks(schema: dict, routing: dict, kg_state: KGState) -> None:
    stage_key = "stage_1"
    modulation = routing.get("modulation", {})
    active: set[str] = set()

    for dimension, mapping in modulation.items():
        if dimension not in kg_state.type_vector:
            continue

        current_value = kg_state.type_vector[dimension]
        value_mapping = mapping.get(current_value)
        if not isinstance(value_mapping, dict):
            continue

        stage_mapping = value_mapping.get(stage_key)
        if not isinstance(stage_mapping, dict):
            continue

        active.update(stage_mapping.get("add_checks", []) or [])

    for check_name, check_schema in schema.get("add_checks", {}).items():
        trigger = check_schema.get("trigger")
        if trigger and _trigger_matches(trigger, kg_state):
            active.add(check_name)

    kg_state.set_checks(stage_key, sorted(active))


def _sync_stage_1_gates(schema: dict, kg_state: KGState) -> None:
    edge = kg_state.entities.get("Edge", {})
    hypothesis = kg_state.entities.get("Hypothesis", {})
    inefficiency = kg_state.entities.get("MarketInefficiency", {})

    _set_gate_status(
        kg_state,
        "G1_1",
        _required_string_status(edge.get("type")),
        _required_string_reason("Edge.type", edge.get("type")),
    )
    _set_gate_status(
        kg_state,
        "G1_2",
        _required_string_status(edge.get("direction")),
        _required_string_reason("Edge.direction", edge.get("direction")),
    )

    consistency = _horizon_consistency(
        kg_state.type_vector.get("decision_cadence"),
        edge.get("horizon"),
    )
    _set_gate_status(kg_state, "G1_3", consistency[0], consistency[1])

    _set_gate_status(
        kg_state,
        "G1_4",
        _required_string_status(hypothesis.get("claim")),
        _required_string_reason("Hypothesis.claim", hypothesis.get("claim")),
    )
    _set_gate_status(
        kg_state,
        "G1_5",
        _required_string_status(hypothesis.get("mechanism")),
        _required_string_reason("Hypothesis.mechanism", hypothesis.get("mechanism")),
    )

    falsifiable = hypothesis.get("falsifiable")
    if falsifiable is True:
        _set_gate_status(kg_state, "G1_6", "pass", "Hypothesis is falsifiable.")
    elif falsifiable is False:
        _set_gate_status(
            kg_state,
            "G1_6",
            "fail",
            "Hypothesis.falsifiable is false, which blocks Stage 1.",
        )
    else:
        _set_gate_status(
            kg_state,
            "G1_6",
            "pending",
            "Hypothesis.falsifiable has not been resolved yet.",
        )

    _set_gate_status(
        kg_state,
        "G1_7",
        _required_string_status(hypothesis.get("falsification_condition")),
        _required_string_reason(
            "Hypothesis.falsification_condition",
            hypothesis.get("falsification_condition"),
        ),
    )
    _set_gate_status(
        kg_state,
        "G1_8",
        _required_string_status(inefficiency.get("persistence")),
        _required_string_reason(
            "MarketInefficiency.persistence",
            inefficiency.get("persistence"),
        ),
    )
    _set_gate_status(
        kg_state,
        "G1_9",
        _required_string_status(inefficiency.get("structural_barrier")),
        _required_string_reason(
            "MarketInefficiency.structural_barrier",
            inefficiency.get("structural_barrier"),
        ),
    )

    for gate_id in _active_gate_ids_for_stage(1, schema, kg_state):
        kg_state.gates.setdefault(
            gate_id,
            {
                "status": "pending",
                "reason": "Activated by routing modulation and not resolved yet.",
            },
        )


def _ensure_stage_0_ready(schema: dict, kg_state: KGState) -> None:
    if kg_state.out_of_scope:
        raise RuntimeValidationError(kg_state.out_of_scope_reason)

    fields = schema.get("output", {}).get("type_vector", {}).get("fields", [])
    missing = [field for field in fields if not kg_state.type_vector.get(field)]
    if missing:
        raise RuntimeValidationError(
            f"Stage 0 is incomplete. Missing type_vector fields: {missing}"
        )

    for field in fields:
        validate_type_vector_update(schema, field, kg_state.type_vector[field])


def _ensure_stage_1_ready(schema: dict, kg_state: KGState) -> None:
    required_deferred = list(schema.get("resolves_deferred", {}).keys())
    missing_deferred = [
        name for name in required_deferred if not kg_state.deferred.get(name)
    ]
    if missing_deferred:
        raise RuntimeValidationError(
            f"Stage 1 is incomplete. Missing deferred values: {missing_deferred}"
        )

    required = _base_gate_ids(schema) + _active_gate_ids_for_stage(1, schema, kg_state)
    unresolved = [
        gate_id
        for gate_id in required
        if kg_state.gates.get(gate_id, {}).get("status") != "pass"
    ]
    if unresolved:
        raise RuntimeValidationError(
            f"Stage 1 is incomplete. Gates not passed: {unresolved}"
        )


def _required_string_status(value: Any) -> str:
    return "pass" if _is_meaningful_string(value) else "pending"


def _required_string_reason(label: str, value: Any) -> str:
    if _is_meaningful_string(value):
        return f"{label} is defined."
    if isinstance(value, str) and value.strip():
        return f"{label} is still unresolved (placeholder value: {value!r})."
    return f"{label} is still unresolved."


def _set_gate_status(kg_state: KGState, gate_id: str, status: str, reason: str) -> None:
    kg_state.gates[gate_id] = {"status": status, "reason": reason}


def _base_gate_ids(schema: dict) -> list[str]:
    return [gate["id"] for gate in schema.get("gate_conditions", [])]


def _active_gate_ids_for_stage(stage: int, schema: dict, kg_state: KGState) -> list[str]:
    stage_key = f"stage_{stage}"
    active_gate_ids: list[str] = []
    active_checks = kg_state.active_checks.get(stage_key, [])
    add_checks = schema.get("add_checks", {})

    for check_name in active_checks:
        check_schema = add_checks.get(check_name, {})
        for gate in check_schema.get("gate_addition", []) or []:
            gate_id = gate.get("id")
            if gate_id:
                active_gate_ids.append(gate_id)

    return active_gate_ids


def _horizon_consistency(decision_cadence: Any, horizon: Any) -> tuple[str, str]:
    if not isinstance(decision_cadence, str) or not decision_cadence.strip():
        return ("pending", "decision_cadence is unresolved, so horizon consistency is unknown.")

    if not isinstance(horizon, str) or not horizon.strip():
        return ("pending", "Edge.horizon is unresolved.")

    horizon_rank = _infer_horizon_rank(horizon)
    if horizon_rank is None:
        return (
            "pending",
            "Edge.horizon could not be interpreted into a comparable time bucket.",
        )

    minimum_rank = {
        "intraday": 0,
        "daily": 1,
        "weekly": 2,
        "monthly": 3,
        "quarterly_plus": 3,
    }.get(decision_cadence)

    if minimum_rank is None:
        return (
            "pending",
            f"Unknown decision_cadence value: {decision_cadence!r}",
        )

    if horizon_rank >= minimum_rank:
        return ("pass", "Edge.horizon is consistent with decision_cadence.")

    return (
        "fail",
        "Edge.horizon is shorter than the strategy's decision cadence, "
        "which is inconsistent for Stage 1.",
    )


def _infer_horizon_rank(horizon: str) -> int | None:
    normalized = horizon.lower().strip()

    if any(token in normalized for token in ("minute", "minutes", "hour", "hours", "intraday", "same day", "same-day")):
        return 0

    if "overnight" in normalized:
        return 1

    match = re.search(r"\b(\d+)\s*day", normalized)
    if match:
        days = int(match.group(1))
        if days <= 5:
            return 1
        if days < 20:
            return 2
        return 3

    match = re.search(r"\b(\d+)\s*d\b", normalized)
    if match:
        days = int(match.group(1))
        if days <= 5:
            return 1
        if days < 20:
            return 2
        return 3

    if any(token in normalized for token in ("day", "daily", "next day", "next-day")):
        return 1

    if any(token in normalized for token in ("week", "weekly")):
        return 2

    if any(token in normalized for token in ("month", "monthly", "quarter", "year", "annual")):
        return 3

    return None


def _trigger_matches(trigger: str, kg_state: KGState) -> bool:
    match = re.fullmatch(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*==\s*([a-zA-Z0-9_]+)\s*", trigger)
    if not match:
        return False

    field_name, expected_value = match.groups()

    if field_name in kg_state.deferred:
        actual_value = kg_state.deferred.get(field_name)
    else:
        actual_value = kg_state.type_vector.get(field_name)

    return actual_value == expected_value


# -----------------------------------------------------------------------------
# Stage 2: Decomposition
# -----------------------------------------------------------------------------
# Plan fields (required for plan gate per Check)
_CHECK_PLAN_FIELDS = ("tests", "evidence_type", "criterion", "method")

# Evidence fields always required regardless of evidence_type
_CHECK_EVIDENCE_CORE_FIELDS = ("evidence_summary", "passes_criterion", "reasoning")

# ReturnDecomposition plan + evidence field groups
_DECOMP_PLAN_FIELDS = ("method", "sample_period")
_DECOMP_EVIDENCE_FIELDS = (
    "market_beta_description",
    "factor_components",
    "residual_alpha",
    "supports_hypothesis",
    "reasoning",
)


def _sync_stage_2_checks(routing: dict, kg_state: KGState) -> None:
    """Recompute active checks for Stage 2 from routing modulation.

    Unlike Stage 1's sync, this reads BOTH type_vector AND deferred —
    signal_source lives in deferred but drives Stage 2 bias checks.
    """
    stage_key = "stage_2"
    modulation = routing.get("modulation", {})
    active: set[str] = set()

    for dimension, mapping in modulation.items():
        if not isinstance(mapping, dict):
            continue

        # signal_source modulation lives in the same structure but the
        # dimension value comes from kg_state.deferred
        current_value = kg_state.type_vector.get(dimension)
        if current_value is None:
            current_value = kg_state.deferred.get(dimension)
        if current_value is None:
            continue

        value_mapping = mapping.get(current_value)
        if not isinstance(value_mapping, dict):
            continue

        stage_mapping = value_mapping.get(stage_key)
        if not isinstance(stage_mapping, dict):
            continue

        active.update(stage_mapping.get("add_checks", []) or [])

    kg_state.set_checks(stage_key, sorted(active))


def _sync_stage_2_gates(schema: dict, kg_state: KGState) -> None:
    """Derive Stage 2 gate statuses from KG state deterministically."""
    # G2_P0: signal_source resolved
    signal_source = kg_state.deferred.get("signal_source")
    _set_gate_status(
        kg_state,
        "G2_P0",
        "pass" if isinstance(signal_source, str) and signal_source.strip() else "pending",
        (
            f"signal_source resolved as {signal_source!r}."
            if signal_source
            else "signal_source has not been resolved yet."
        ),
    )

    decomp = kg_state.entities.get("ReturnDecomposition", {})

    # G2_P1: ReturnDecomposition.method
    _set_gate_status(
        kg_state,
        "G2_P1",
        _required_string_status(decomp.get("method")),
        _required_string_reason("ReturnDecomposition.method", decomp.get("method")),
    )

    # G2_P2: ReturnDecomposition.sample_period
    _set_gate_status(
        kg_state,
        "G2_P2",
        _required_string_status(decomp.get("sample_period")),
        _required_string_reason(
            "ReturnDecomposition.sample_period", decomp.get("sample_period")
        ),
    )

    # G2_P3: every activated Check has all plan fields defined
    active = list(kg_state.active_checks.get("stage_2", []))
    if not active:
        _set_gate_status(
            kg_state,
            "G2_P3",
            "pass",
            "No Stage 2 checks are active for this route, so no per-check plan is required.",
        )
    else:
        incomplete = []
        for name in active:
            check = kg_state.entities.get(name, {})
            missing = [f for f in _CHECK_PLAN_FIELDS if not _nonempty(check.get(f))]
            if missing:
                incomplete.append(f"{name}: missing {missing}")
        if incomplete:
            _set_gate_status(
                kg_state,
                "G2_P3",
                "pending",
                "Plan fields incomplete — " + "; ".join(incomplete),
            )
        else:
            _set_gate_status(
                kg_state,
                "G2_P3",
                "pass",
                f"All {len(active)} Stage 2 checks have plan fields defined.",
            )

    # G2_E1: ReturnDecomposition evidence (advisory)
    missing_decomp_ev = [f for f in _DECOMP_EVIDENCE_FIELDS if not _nonempty(decomp.get(f))]
    if missing_decomp_ev:
        _set_gate_status(
            kg_state,
            "G2_E1",
            "pending",
            f"ReturnDecomposition evidence incomplete: missing {missing_decomp_ev}.",
        )
    else:
        _set_gate_status(
            kg_state,
            "G2_E1",
            "pass",
            "ReturnDecomposition evidence fields are filled.",
        )

    # G2_E2: each Check has evidence fields (advisory)
    if not active:
        _set_gate_status(
            kg_state,
            "G2_E2",
            "pending",
            "No active Stage 2 checks yet.",
        )
    else:
        incomplete_ev = []
        for name in active:
            check = kg_state.entities.get(name, {})
            missing = _missing_check_evidence_fields(check)
            if missing:
                incomplete_ev.append(f"{name}: missing {missing}")
        if incomplete_ev:
            _set_gate_status(
                kg_state,
                "G2_E2",
                "pending",
                "Evidence incomplete — " + "; ".join(incomplete_ev),
            )
        else:
            _set_gate_status(
                kg_state,
                "G2_E2",
                "pass",
                f"All {len(active)} Stage 2 checks have evidence fields filled.",
            )


def _missing_check_evidence_fields(check: dict) -> list[str]:
    """Determine which evidence fields are missing given evidence_type."""
    missing: list[str] = []
    for field in _CHECK_EVIDENCE_CORE_FIELDS:
        if field == "passes_criterion":
            if check.get(field) is None:
                missing.append(field)
        elif not _nonempty(check.get(field)):
            missing.append(field)

    evidence_type = check.get("evidence_type")
    if evidence_type == "quantitative" and not _nonempty(check.get("metrics")):
        missing.append("metrics")
    if evidence_type == "artifact_ref" and not _nonempty(check.get("artifact_ref")):
        missing.append("artifact_ref")

    return missing


def _ensure_stage_2_ready(schema: dict, kg_state: KGState) -> None:
    """Stage 2 advancement: plan gates blocking, evidence gates advisory."""
    plan_gates = [
        gate["id"]
        for gate in schema.get("gate_conditions", [])
        if gate.get("kind") == "plan"
    ]

    unresolved_plan = [
        gid
        for gid in plan_gates
        if kg_state.gates.get(gid, {}).get("status") != "pass"
    ]
    if unresolved_plan:
        raise RuntimeValidationError(
            "Stage 2 plan gates not passed: "
            + ", ".join(unresolved_plan)
            + ". Evidence gates are advisory and may remain pending."
        )


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return _is_meaningful_string(value)
    return True


def _is_meaningful_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    normalized = value.strip()
    if not normalized:
        return False

    return normalized.lower() not in _PLACEHOLDER_STRINGS
