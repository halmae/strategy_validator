"""KG state management and persistence.

KG state is saved as a JSON file alongside the strategy .md file.
e.g. strategies/my_strategy.md → strategies/my_strategy.kg.json

State persists between sessions — the agent resumes from where it left off.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class KGState:
    # Current stage (0 = classification, 1 = edge_existence, ...)
    stage: int = 0

    # Stage 0 output: 6-dimensional type vector
    type_vector: dict = field(default_factory=dict)

    # KG entities populated during Stage 1+
    # e.g. {"Edge": {"type": "behavioral", ...}, "Hypothesis": {...}}
    entities: dict = field(default_factory=dict)

    # Gate condition statuses per stage
    # e.g. {"G1_1": {"status": "pass", "reason": "..."}}
    gates: dict = field(default_factory=dict)

    # Deferred dimensions resolved after Stage 0
    # e.g. {"universe_scope": "concentrated", "universe_anchor": "KOSPI"}
    deferred: dict = field(default_factory=dict)

    # Agent notes and flags for later stages
    # e.g. [{"stage": 2, "type": "risk", "note": "informed trading decay risk"}]
    notes: list = field(default_factory=list)

    # Derived KG relations with stage-scoped fixed predicates
    # e.g. [{"stage": 1, "subject": "Edge", "predicate": "grounded_in", "object": "Hypothesis"}]
    relations: list = field(default_factory=list)

    # add_checks activated by routing modulation
    # e.g. {"stage_2": ["regime_sensitivity", "event_definition_consistency"]}
    active_checks: dict = field(default_factory=dict)

    # Implemented stage completion tracker
    completed_stages: list = field(default_factory=list)

    # Conditional stages skipped by routing
    skipped_stages: list = field(default_factory=list)

    # Stage handoff summaries used to compress long session history
    stage_summaries: dict = field(default_factory=dict)

    # Runtime terminal states for the currently implemented scope
    workflow_complete: bool = False
    out_of_scope: bool = False
    out_of_scope_reason: str = ""

    def update_entity(self, entity: str, prop: str, value) -> None:
        if entity not in self.entities:
            self.entities[entity] = {}
        self.entities[entity][prop] = value

    def update_type_vector(self, dimension: str, value: str) -> None:
        self.type_vector[dimension] = value

    def mark_gate(self, gate_id: str, status: str, reason: str = "") -> None:
        self.gates[gate_id] = {"status": status, "reason": reason}

    def add_note(self, stage: int, note_type: str, note: str) -> None:
        self.notes.append({"stage": stage, "type": note_type, "note": note})

    def set_stage_relations(self, stage: int, relations: list[dict]) -> None:
        retained = [item for item in self.relations if item.get("stage") != stage]
        retained.extend(relations)
        self.relations = retained

    def add_check(self, stage_key: str, check_id: str) -> None:
        if stage_key not in self.active_checks:
            self.active_checks[stage_key] = []
        if check_id not in self.active_checks[stage_key]:
            self.active_checks[stage_key].append(check_id)

    def set_checks(self, stage_key: str, check_ids: list[str]) -> None:
        self.active_checks[stage_key] = list(check_ids)

    def advance_stage(self, to_stage: int) -> None:
        self.stage = to_stage

    def mark_stage_completed(self, stage: int) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def mark_stage_skipped(self, stage: int) -> None:
        if stage not in self.skipped_stages:
            self.skipped_stages.append(stage)

    def set_stage_summary(self, stage: int, summary: dict) -> None:
        self.stage_summaries[f"stage_{stage}"] = summary

    def gate_summary(self) -> dict:
        passed = [k for k, v in self.gates.items() if v["status"] == "pass"]
        failed = [k for k, v in self.gates.items() if v["status"] == "fail"]
        pending = [k for k, v in self.gates.items() if v["status"] == "pending"]
        return {"passed": passed, "failed": failed, "pending": pending}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KGState":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in d.items() if key in fields})


def kg_path(strategy_path: Path) -> Path:
    return strategy_path.with_suffix(".kg.json")


def load_kg_state(strategy_path: Path) -> KGState:
    path = kg_path(strategy_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return KGState.from_dict(data)
    return KGState()


def save_kg_state(strategy_path: Path, state: KGState) -> None:
    path = kg_path(strategy_path)
    path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
