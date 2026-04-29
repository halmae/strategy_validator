"""Deterministic runtime helpers for stages 0 through 3.

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
        _sync_stage_relations(stage, schema, kg_state)
        return

    if stage == 2:
        _sync_stage_2_checks(routing, kg_state)
        _sync_stage_2_gates(schema, kg_state)
        _sync_stage_relations(stage, schema, kg_state)
        return

    if stage == 3:
        _sync_stage_3_checks(routing, kg_state)
        _sync_stage_3_gates(schema, kg_state)
        _sync_stage_relations(stage, schema, kg_state)


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

    # Stages 2+: if `entity` is not explicitly declared but IS an activated
    # check name, fall back to the generic `Check` template schema.
    if entity_schema is None and stage in (2, 3) and kg_state is not None:
        active = set(kg_state.active_checks.get(f"stage_{stage}", []))
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

    prerequisite_messages = _missing_prerequisite_messages(
        from_stage,
        routing,
        kg_state,
    )
    if prerequisite_messages:
        raise RuntimeValidationError(
            "Routing prerequisites are not satisfied: "
            + " ".join(prerequisite_messages)
        )

    resolved_to_stage = _next_stage_after(from_stage, routing, kg_state)
    allowed_requests = {from_stage + 1, resolved_to_stage}
    if to_stage not in allowed_requests:
        raise RuntimeValidationError(
            f"Invalid stage transition: {from_stage} -> {to_stage}. "
            f"Next routed stage is {resolved_to_stage}."
        )

    sync_runtime_state(kg_state.stage, schema, routing, kg_state)

    if from_stage == 0:
        _ensure_stage_0_ready(schema, kg_state)
    elif from_stage == 1:
        _ensure_stage_1_ready(schema, kg_state)
    elif from_stage == 2:
        _ensure_stage_2_ready(schema, kg_state)
    elif from_stage == 3:
        _ensure_stage_3_ready(schema, kg_state)
    else:
        raise RuntimeValidationError(f"Stage {from_stage} is not supported yet")

    kg_state.mark_stage_completed(from_stage)

    skipped = [
        stage_id
        for stage_id in range(from_stage + 1, resolved_to_stage)
        if not _is_stage_required(stage_id, routing, kg_state)
    ]
    for stage_id in skipped:
        kg_state.mark_stage_skipped(stage_id)

    if resolved_to_stage > max_implemented_stage:
        kg_state.workflow_complete = True
        skip_note = (
            " " + " ".join(f"Stage {stage_id} skipped." for stage_id in skipped)
            if skipped
            else ""
        )
        requested_note = (
            f" Requested Stage {to_stage}, but routing resolved next stage "
            f"to {resolved_to_stage}."
            if to_stage != resolved_to_stage
            else ""
        )
        return (
            f"Stage {from_stage} complete.{skip_note} "
            f"Stage {resolved_to_stage} is not implemented yet.{requested_note}"
        )

    kg_state.advance_stage(resolved_to_stage)
    kg_state.workflow_complete = False
    if skipped:
        skipped_text = ", ".join(str(stage_id) for stage_id in skipped)
        requested_note = (
            f" Requested Stage {to_stage}, but routing resolved next stage "
            f"to {resolved_to_stage}."
            if to_stage != resolved_to_stage
            else ""
        )
        return (
            f"Stage {from_stage} -> {resolved_to_stage} "
            f"(skipped {skipped_text}).{requested_note}"
        )
    return f"Stage {from_stage} -> {resolved_to_stage}"


def _next_stage_after(from_stage: int, routing: dict, kg_state: KGState) -> int:
    next_stage = from_stage + 1
    max_stage = _routing_max_stage(routing)
    while next_stage <= max_stage and not _is_stage_required(
        next_stage,
        routing,
        kg_state,
    ):
        next_stage += 1
    return next_stage


def _is_stage_required(stage: int, routing: dict, kg_state: KGState) -> bool:
    default_status = _stage_catalog_status(stage, routing)
    rules = [
        rule for rule in routing.get("rules", []) or []
        if rule.get("stage") == stage
    ]
    for rule in rules:
        for condition in rule.get("conditions", []) or []:
            if "if" in condition and _route_condition_matches(
                condition["if"],
                kg_state,
            ):
                return condition.get("then", default_status) != "skip"
            if "default" in condition:
                return condition["default"] != "skip"
    return default_status != "skip"


def _stage_catalog_status(stage: int, routing: dict) -> str:
    for stage_spec in routing.get("stages", []) or []:
        if stage_spec.get("id") == stage:
            return stage_spec.get("status", "required")
    return "required"


def _routing_max_stage(routing: dict) -> int:
    stage_ids = [
        stage_spec.get("id")
        for stage_spec in routing.get("stages", []) or []
        if isinstance(stage_spec.get("id"), int)
    ]
    return max(stage_ids, default=0)


def _route_condition_matches(condition: dict, kg_state: KGState) -> bool:
    entity_name = condition.get("kg_entity_exists")
    if entity_name:
        return _entity_exists(kg_state, entity_name)
    for key, expected in condition.items():
        actual = kg_state.type_vector.get(key)
        if actual is None:
            actual = kg_state.deferred.get(key)
        if actual != expected:
            return False
    return True


def _missing_prerequisite_messages(
    from_stage: int,
    routing: dict,
    kg_state: KGState,
) -> list[str]:
    next_stage = from_stage + 1
    rule_entities = _rule_entities_for_stage(next_stage, routing)
    messages: list[str] = []

    for prerequisite in routing.get("prerequisites", []) or []:
        if not _route_condition_matches(prerequisite.get("condition", {}), kg_state):
            continue
        required_entity = (prerequisite.get("requires") or {}).get("kg_entity")
        if not required_entity or required_entity not in rule_entities:
            continue
        if _entity_exists(kg_state, required_entity):
            continue

        message = str(prerequisite.get("message", "")).strip()
        if message:
            messages.append(message)
        else:
            messages.append(f"{required_entity} is required before Stage {next_stage}.")

    return messages


def _rule_entities_for_stage(stage: int, routing: dict) -> set[str]:
    entities: set[str] = set()
    for rule in routing.get("rules", []) or []:
        if rule.get("stage") != stage:
            continue
        for condition in rule.get("conditions", []) or []:
            if_condition = condition.get("if")
            if isinstance(if_condition, dict) and if_condition.get("kg_entity_exists"):
                entities.add(if_condition["kg_entity_exists"])
    return entities


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

    required_deferred = list(schema.get("resolves_deferred", {}).keys())
    missing_deferred = [
        name for name in required_deferred if not kg_state.deferred.get(name)
    ]
    if missing_deferred:
        _set_gate_status(
            kg_state,
            "G1_0",
            "pending",
            f"Stage 1 deferred values are unresolved: missing {missing_deferred}.",
        )
    else:
        _set_gate_status(
            kg_state,
            "G1_0",
            "pass",
            "Stage 1 deferred values are resolved.",
        )

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
    for name in required_deferred:
        if kg_state.deferred.get(name):
            validate_deferred_update(schema, name, kg_state.deferred[name])

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

    _ensure_required_relations(1, schema, kg_state)


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


def _sync_stage_relations(stage: int, schema: dict, kg_state: KGState) -> None:
    """Derive stage-scoped KG relations from the current entity/property state."""
    relation_specs = schema.get("relationships", []) or []
    derived: list[dict] = []

    for spec in relation_specs:
        kind = spec.get("kind", "fixed")
        if kind == "fixed":
            relation = _derive_fixed_relation(stage, spec, kg_state)
            if relation is not None:
                derived.append(relation)
        elif kind == "active_check_field":
            derived.extend(_derive_active_check_relations(stage, spec, kg_state))
        elif kind == "mapped_field_predicate":
            relation = _derive_mapped_field_relation(stage, spec, kg_state)
            if relation is not None:
                derived.append(relation)
        else:
            raise RuntimeValidationError(
                f"Unsupported relationship kind for stage {stage}: {kind}"
            )

    kg_state.set_stage_relations(stage, _dedupe_relations(derived))


def _derive_fixed_relation(stage: int, spec: dict, kg_state: KGState) -> dict | None:
    subject = spec.get("subject")
    predicate = spec.get("predicate")
    object_name = spec.get("object")
    object_ref = spec.get("object_ref")

    if not subject or not predicate or not _entity_exists(kg_state, subject):
        return None

    relation = {
        "stage": stage,
        "subject": subject,
        "predicate": predicate,
        "scope": spec.get("scope", "semantic"),
    }

    if object_name:
        if not _relation_object_ready(kg_state, object_name):
            return None
        relation["object"] = object_name
        return relation

    if object_ref:
        value = _entity_field_value(kg_state, object_ref)
        if not _nonempty(value):
            return None
        relation["object"] = object_ref
        relation["object_value"] = value
        return relation

    return None


def _derive_active_check_relations(stage: int, spec: dict, kg_state: KGState) -> list[dict]:
    predicate = spec.get("predicate")
    field_name = spec.get("field")
    if not predicate or not field_name:
        return []

    derived: list[dict] = []
    for check_name in kg_state.active_checks.get(f"stage_{stage}", []):
        field_value = kg_state.entities.get(check_name, {}).get(field_name)
        if not _nonempty(field_value):
            continue
        derived.append(
            {
                "stage": stage,
                "subject": check_name,
                "predicate": predicate,
                "object": str(field_value).strip(),
                "scope": spec.get("scope", "semantic"),
            }
        )

    return derived


def _derive_mapped_field_relation(stage: int, spec: dict, kg_state: KGState) -> dict | None:
    subject = spec.get("subject")
    field_name = spec.get("predicate_field")
    predicate_values = spec.get("predicate_values", {})
    object_name = spec.get("object")
    object_ref = spec.get("object_ref")

    if not subject or not field_name or not _entity_exists(kg_state, subject):
        return None

    field_value = kg_state.entities.get(subject, {}).get(field_name)
    if not _nonempty(field_value):
        return None

    predicate = predicate_values.get(str(field_value).strip())
    if not predicate:
        return None

    relation = {
        "stage": stage,
        "subject": subject,
        "predicate": predicate,
        "scope": spec.get("scope", "semantic"),
    }

    if object_name:
        if not _relation_object_ready(kg_state, object_name):
            return None
        relation["object"] = object_name
        return relation

    if object_ref:
        value = _entity_field_value(kg_state, object_ref)
        if not _nonempty(value):
            return None
        relation["object"] = object_ref
        relation["object_value"] = value
        return relation

    return None


def _ensure_required_relations(stage: int, schema: dict, kg_state: KGState) -> None:
    missing: list[str] = []
    active_checks = kg_state.active_checks.get(f"stage_{stage}", [])

    for spec in schema.get("relationships", []) or []:
        kind = spec.get("kind", "fixed")
        if kind == "fixed" and spec.get("required"):
            if not _has_stage_relation(
                kg_state,
                stage,
                subject=spec.get("subject"),
                predicate=spec.get("predicate"),
                object_name=spec.get("object") or spec.get("object_ref"),
            ):
                missing.append(
                    f"{spec.get('subject')} --{spec.get('predicate')}--> "
                    f"{spec.get('object') or spec.get('object_ref')}"
                )
        elif kind == "active_check_field" and spec.get("required_per_active_check"):
            predicate = spec.get("predicate")
            field_name = spec.get("field")
            for check_name in active_checks:
                field_value = kg_state.entities.get(check_name, {}).get(field_name)
                if not _nonempty(field_value):
                    missing.append(f"{check_name} --{predicate}--> <unresolved>")
                    continue
                if not _has_stage_relation(
                    kg_state,
                    stage,
                    subject=check_name,
                    predicate=predicate,
                    object_name=str(field_value).strip(),
                ):
                    missing.append(
                        f"{check_name} --{predicate}--> {str(field_value).strip()}"
                    )

    if missing:
        raise RuntimeValidationError(
            "Required stage relations are missing: " + "; ".join(missing)
        )


def _has_stage_relation(
    kg_state: KGState,
    stage: int,
    subject: str | None,
    predicate: str | None,
    object_name: str | None,
) -> bool:
    for relation in kg_state.relations:
        if relation.get("stage") != stage:
            continue
        if relation.get("subject") != subject:
            continue
        if relation.get("predicate") != predicate:
            continue
        if relation.get("object") != object_name:
            continue
        return True
    return False


def _dedupe_relations(relations: list[dict]) -> list[dict]:
    seen: set[tuple[Any, ...]] = set()
    ordered: list[dict] = []
    for relation in relations:
        key = (
            relation.get("stage"),
            relation.get("subject"),
            relation.get("predicate"),
            relation.get("object"),
            relation.get("object_value"),
            relation.get("scope"),
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(relation)
    return ordered


def _entity_exists(kg_state: KGState, entity_name: str) -> bool:
    return bool(kg_state.entities.get(entity_name))


def _relation_object_ready(kg_state: KGState, object_name: str) -> bool:
    if "." in object_name:
        return _nonempty(_entity_field_value(kg_state, object_name))
    return _entity_exists(kg_state, object_name)


def _entity_field_value(kg_state: KGState, ref: str) -> Any:
    if "." not in ref:
        return None
    entity_name, field_name = ref.split(".", 1)
    return kg_state.entities.get(entity_name, {}).get(field_name)


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

    if any(
        token in normalized
        for token in (
            "month",
            "monthly",
            "quarter",
            "year",
            "annual",
            "개월",
            "분기",
            "월간",
            "연간",
            "년",
        )
    ) or re.search(r"\d+\s*(개월|월|년)", normalized):
        return 3

    if any(
        token in normalized
        for token in (
            "minute",
            "minutes",
            "min",
            "mins",
            "분봉",
            "hour",
            "hours",
            "hr",
            "hrs",
            "시간",
            "intraday",
            "same day",
            "same-day",
            "당일",
            "장중",
            "인트라데이",
        )
    ) or re.search(r"\d+\s*분(?!기)", normalized):
        return 0

    if any(
        token in normalized for token in ("overnight", "오버나이트", "익일", "다음날")
    ):
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

    if any(
        token in normalized
        for token in (
            "day",
            "daily",
            "next day",
            "next-day",
            "하루",
            "수일",
            "며칠",
            "일간",
            "영업일",
            "거래일",
        )
    ) or re.search(r"\d+\s*일", normalized):
        return 1

    if any(
        token in normalized for token in ("week", "weekly", "주간", "주일")
    ) or re.search(r"\d+\s*주", normalized):
        return 2

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

_SIGNAL_SCORE_PLAN_FIELDS = (
    "definition",
    "signal_direction",
    "measurement_method",
    "evaluation_window",
)
_SIGNAL_SCORE_EVIDENCE_FIELDS = (
    "ic_metric",
    "hit_rate",
    "decay_profile",
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

    _ensure_required_relations(2, schema, kg_state)


# -----------------------------------------------------------------------------
# Stage 3: Signal Quality
# -----------------------------------------------------------------------------
def _sync_stage_3_checks(routing: dict, kg_state: KGState) -> None:
    stage_key = "stage_3"
    modulation = routing.get("modulation", {})
    active: set[str] = set()

    for dimension, mapping in modulation.items():
        if not isinstance(mapping, dict):
            continue

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


def _sync_stage_3_gates(schema: dict, kg_state: KGState) -> None:
    signal_score = kg_state.entities.get("SignalScore", {})

    if signal_score:
        _set_gate_status(
            kg_state,
            "G3_0",
            "pass",
            "SignalScore entity exists.",
        )
    else:
        _set_gate_status(
            kg_state,
            "G3_0",
            "pending",
            "SignalScore entity has not been defined.",
        )

    missing_signal_plan = [
        field for field in _SIGNAL_SCORE_PLAN_FIELDS
        if not _nonempty(signal_score.get(field))
    ]
    if missing_signal_plan:
        _set_gate_status(
            kg_state,
            "G3_P1",
            "pending",
            f"SignalScore plan fields incomplete: missing {missing_signal_plan}.",
        )
    else:
        _set_gate_status(
            kg_state,
            "G3_P1",
            "pass",
            "SignalScore plan fields are defined.",
        )

    active = list(kg_state.active_checks.get("stage_3", []))
    if not active:
        _set_gate_status(
            kg_state,
            "G3_P2",
            "pass",
            "No Stage 3 checks are active for this route, so no per-check plan is required.",
        )
    else:
        incomplete = []
        for name in active:
            check = kg_state.entities.get(name, {})
            missing = [
                field for field in _CHECK_PLAN_FIELDS
                if not _nonempty(check.get(field))
            ]
            if missing:
                incomplete.append(f"{name}: missing {missing}")
        if incomplete:
            _set_gate_status(
                kg_state,
                "G3_P2",
                "pending",
                "Plan fields incomplete — " + "; ".join(incomplete),
            )
        else:
            _set_gate_status(
                kg_state,
                "G3_P2",
                "pass",
                f"All {len(active)} Stage 3 checks have plan fields defined.",
            )

    missing_signal_ev = [
        field for field in _SIGNAL_SCORE_EVIDENCE_FIELDS
        if not _nonempty(signal_score.get(field))
    ]
    if missing_signal_ev:
        _set_gate_status(
            kg_state,
            "G3_E1",
            "pending",
            f"SignalScore evidence incomplete: missing {missing_signal_ev}.",
        )
    else:
        _set_gate_status(
            kg_state,
            "G3_E1",
            "pass",
            "SignalScore evidence fields are filled.",
        )

    if not active:
        _set_gate_status(
            kg_state,
            "G3_E2",
            "pending",
            "No active Stage 3 checks yet.",
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
                "G3_E2",
                "pending",
                "Evidence incomplete — " + "; ".join(incomplete_ev),
            )
        else:
            _set_gate_status(
                kg_state,
                "G3_E2",
                "pass",
                f"All {len(active)} Stage 3 checks have evidence fields filled.",
            )


def _ensure_stage_3_ready(schema: dict, kg_state: KGState) -> None:
    blocking_gates = [
        gate["id"]
        for gate in schema.get("gate_conditions", [])
        if gate.get("kind") in ("entry", "plan")
    ]

    unresolved = [
        gid
        for gid in blocking_gates
        if kg_state.gates.get(gid, {}).get("status") != "pass"
    ]
    if unresolved:
        raise RuntimeValidationError(
            "Stage 3 entry/plan gates not passed: "
            + ", ".join(unresolved)
            + ". Evidence gates are advisory and may remain pending."
        )

    _ensure_required_relations(3, schema, kg_state)


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
