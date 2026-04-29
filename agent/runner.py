"""Agent runner — Gemini API integration with function calling for KG state management.

Design:
- Conversation is natural language
- KG state changes happen via function calls (structured)
- System prompt assembled from config/prompts/ files + dynamic KG context
- Two modes: question_mode (default), proposal_mode (when user is stuck)
"""
import json
import yaml

from google import genai
from google.genai import types

from .kg_store import KGState
from .config_loader import load_settings, load_prompt
from .stage_summary import build_prompt_kg_snapshot, format_stage_summaries
from .stage_runtime import (
    RuntimeValidationError,
    attempt_stage_advance,
    sync_runtime_state,
    validate_deferred_update,
    validate_entity_update,
    validate_gate_update,
    validate_type_vector_update,
)

# ---------------------------------------------------------------------------
# Tool / Function declarations
# ---------------------------------------------------------------------------
FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="update_type_vector",
        description=(
            "Stage 0에서 type_vector의 dimension 값을 확정할 때 사용합니다. "
            "사용자 답변 또는 추론을 통해 dimension이 결정될 때마다 호출하세요."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string",
                    "description": "classification.yaml의 dimension 이름",
                },
                "value": {
                    "type": "string",
                    "description": "선택된 값",
                },
                "reasoning": {
                    "type": "string",
                    "description": "이 값을 선택한 근거",
                },
            },
            "required": ["dimension", "value"],
        },
    ),
    types.FunctionDeclaration(
        name="update_kg_entity",
        description=(
            "Stage 1과 Stage 2에서 KG entity property를 업데이트할 때 사용합니다. "
            "예: Stage 1의 Edge/Hypothesis/MarketInefficiency, "
            "Stage 2의 ReturnDecomposition 또는 활성화된 check entity."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity 이름 (예: Edge, Hypothesis, MarketInefficiency)",
                },
                "property": {"type": "string"},
                "value": {
                    "type": "string",
                    "description": "속성 값. bool 속성은 'true' 또는 'false' 문자열로 전달합니다.",
                },
                "note": {"type": "string"},
            },
            "required": ["entity", "property", "value"],
        },
    ),
    types.FunctionDeclaration(
        name="update_deferred",
        description=(
            "classification.yaml에서 deferred로 표시된 dimension을 해당 stage에서 해소합니다. "
            "(예: universe_scope, universe_anchor → Stage 1, signal_source → Stage 2)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "dimension": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["dimension", "value"],
        },
    ),
    types.FunctionDeclaration(
        name="mark_gate",
        description=(
            "Gate condition의 충족 여부를 기록합니다. "
            "pass: 충족됨, fail: 불충족(blocking), pending: 모름(proposal_mode 전환)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "gate_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pass", "fail", "pending"],
                },
                "reason": {"type": "string"},
            },
            "required": ["gate_id", "status"],
        },
    ),
    types.FunctionDeclaration(
        name="add_note",
        description=(
            "이후 stage에서 확인이 필요한 관찰, 위험 요소, 플래그를 기록합니다."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target_stage": {"type": "integer"},
                "note_type": {
                    "type": "string",
                    "enum": ["flag", "observation", "risk"],
                },
                "note": {"type": "string"},
            },
            "required": ["target_stage", "note_type", "note"],
        },
    ),
    types.FunctionDeclaration(
        name="advance_stage",
        description=(
            "현재 stage의 모든 required gate가 통과됐을 때 다음 stage로 전환합니다."
        ),
        parameters={
            "type": "object",
            "properties": {
                "from_stage": {"type": "integer"},
                "to_stage": {"type": "integer"},
                "summary": {"type": "string"},
            },
            "required": ["from_stage", "to_stage", "summary"],
        },
    ),
]

TOOLS = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]


# ---------------------------------------------------------------------------
# Prompt snapshots
# ---------------------------------------------------------------------------
def build_routing_prompt_snapshot(stage: int, kg_state: KGState) -> dict:
    return {
        "stage": stage,
        "active_checks": list(kg_state.active_checks.get(f"stage_{stage}", [])),
        "resolved_by_runtime": True,
        "note": (
            "Routing modulation has already been resolved deterministically. "
            "Use the active checks in this snapshot instead of re-deriving routes."
        ),
    }


# ---------------------------------------------------------------------------
# System prompt — assembled from config/prompts/ files + dynamic context
# ---------------------------------------------------------------------------
def build_system_prompt(
    stage: int,
    schema: dict,
    routing: dict,
    kg_state: KGState,
    strategy_text: str,
    settings: dict | None = None,
) -> str:
    if settings is None:
        settings = load_settings()

    language = settings.get("language", "english")

    # Static behavioral rules (config/prompts/system.md)
    system_rules = load_prompt("system").format(language=language)

    # Dynamic context (config/prompts/stage_context.md)
    context = load_prompt("stage_context").format(
        stage=stage,
        strategy_text=strategy_text,
        schema_yaml=yaml.dump(schema, allow_unicode=True, default_flow_style=False),
        routing_yaml=yaml.dump(
            build_routing_prompt_snapshot(stage, kg_state),
            allow_unicode=True,
            default_flow_style=False,
        ),
        kg_json=json.dumps(
            build_prompt_kg_snapshot(stage, kg_state),
            ensure_ascii=False,
            indent=2,
        ),
        stage_summaries_json=format_stage_summaries(kg_state.stage_summaries),
    )

    return f"{system_rules}\n\n{context}"


# ---------------------------------------------------------------------------
# Function call processor
# ---------------------------------------------------------------------------
def process_function_call(
    name: str,
    args: dict,
    kg_state: KGState,
    stage: int,
    schema: dict,
    routing: dict,
    max_implemented_stage: int,
) -> str:
    """Process a single function call with deterministic stage validation."""
    try:
        if name == "update_type_vector":
            if stage != 0:
                raise RuntimeValidationError(
                    "update_type_vector is only allowed during Stage 0."
                )
            value = validate_type_vector_update(
                schema,
                args["dimension"],
                args["value"],
            )
            kg_state.update_type_vector(args["dimension"], value)
            sync_runtime_state(stage, schema, routing, kg_state)
            return f"type_vector.{args['dimension']} = {value}"

        if name == "update_kg_entity":
            if stage not in (1, 2):
                raise RuntimeValidationError(
                    "update_kg_entity is only supported during Stage 1 or Stage 2."
                )
            value = validate_entity_update(
                schema,
                args["entity"],
                args["property"],
                args["value"],
                kg_state=kg_state,
                stage=stage,
            )
            kg_state.update_entity(args["entity"], args["property"], value)
            sync_runtime_state(stage, schema, routing, kg_state)
            return f"{args['entity']}.{args['property']} updated"

        if name == "update_deferred":
            if stage not in (1, 2):
                raise RuntimeValidationError(
                    "update_deferred is only allowed during Stage 1 or Stage 2."
                )
            value = validate_deferred_update(
                schema,
                args["dimension"],
                args["value"],
            )
            kg_state.deferred[args["dimension"]] = value
            sync_runtime_state(stage, schema, routing, kg_state)
            return f"deferred.{args['dimension']} = {value}"

        if name == "mark_gate":
            if stage == 0:
                sync_runtime_state(stage, schema, routing, kg_state)
                return "Stage 0 has no gates; mark_gate ignored."
            if stage not in (1, 2):
                raise RuntimeValidationError(
                    "mark_gate is only supported during Stage 1 or Stage 2."
                )
            validate_gate_update(stage, schema, kg_state, args["gate_id"])
            kg_state.mark_gate(args["gate_id"], args["status"], args.get("reason", ""))
            sync_runtime_state(stage, schema, routing, kg_state)
            return f"Gate {args['gate_id']}: {args['status']}"

        if name == "add_note":
            # Notes may intentionally target future stages before they exist.
            kg_state.add_note(args["target_stage"], args["note_type"], args["note"])
            return "Note recorded"

        if name == "advance_stage":
            result = attempt_stage_advance(
                schema=schema,
                routing=routing,
                kg_state=kg_state,
                from_stage=args["from_stage"],
                to_stage=args["to_stage"],
                max_implemented_stage=max_implemented_stage,
            )

            if kg_state.stage == stage:
                sync_runtime_state(stage, schema, routing, kg_state)

            return result

        return "unknown function"
    except RuntimeValidationError as exc:
        return f"Validation error: {exc}"


# ---------------------------------------------------------------------------
# Single turn: send message, handle function call loop, return final text
# ---------------------------------------------------------------------------
def run_turn(
    client: genai.Client,
    contents: list,
    system_prompt: str,
    kg_state: KGState,
    stage: int,
    schema: dict,
    routing: dict,
    max_implemented_stage: int,
    model_config: dict | None = None,
) -> tuple[str, list]:
    """
    Run one conversation turn with Gemini.
    Handles function call loop internally.
    Returns (assistant_text, updated_contents).

    model_config: dict from settings.yaml's `model` section.
        Expected keys: name, max_output_tokens, temperature.
    """
    from .config_loader import get_model_config

    if model_config is None:
        model_config = get_model_config()

    model_name = model_config["name"]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=TOOLS,
        max_output_tokens=model_config.get("max_output_tokens", 2048),
        temperature=model_config.get("temperature", 0.3),
    )

    # Accumulate text across ALL loop iterations.
    # The model may generate text in an early iteration (alongside function calls)
    # and return nothing in the final iteration — we want to keep all of it.
    accumulated_text: list[str] = []

    while True:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # Defensive: handle empty or missing candidates/content
        if not response.candidates:
            break

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            break

        model_parts = []
        text_parts = []
        function_calls = []

        for part in candidate.content.parts:
            model_parts.append(part)
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                function_calls.append(part.function_call)

        accumulated_text.extend(text_parts)

        # Append model turn to contents
        contents = contents + [
            types.Content(role="model", parts=model_parts)
        ]

        if not function_calls:
            # No more function calls — return everything collected so far
            return "\n".join(accumulated_text), contents

        # Process function calls, build function_response parts
        response_parts = []
        for fc in function_calls:
            result = process_function_call(
                fc.name,
                dict(fc.args),
                kg_state,
                stage=stage,
                schema=schema,
                routing=routing,
                max_implemented_stage=max_implemented_stage,
            )
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        # Append function responses as user turn, then loop
        contents = contents + [
            types.Content(role="user", parts=response_parts)
        ]

    # Reached only if loop exited via break (empty response after function calls)
    return "\n".join(accumulated_text), contents
