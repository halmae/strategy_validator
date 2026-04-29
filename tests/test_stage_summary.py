import unittest

from google.genai import types

from agent.kg_store import KGState
from agent.session_store import prune_session_contents
from agent.stage_summary import (
    build_prompt_kg_snapshot,
    build_stage_summary,
    format_stage_summaries,
)


class StageSummaryTests(unittest.TestCase):
    def test_build_stage_1_summary_collects_resolved_facts(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "event_driven",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
                "exposure_structure": "long_only",
                "asset_class": "equity",
            },
            deferred={
                "universe_scope": "concentrated",
                "universe_anchor": "KOSPI",
            },
            entities={
                "Edge": {
                    "type": "behavioral",
                    "direction": "long",
                    "horizon": "1 hour",
                },
                "Hypothesis": {
                    "claim": "Theme chasing creates temporary inefficiencies.",
                    "mechanism": "Retail attention clusters into trending names.",
                    "falsifiable": True,
                    "falsification_condition": "No abnormal return remains after the event.",
                },
                "MarketInefficiency": {
                    "persistence": "Capacity and attention constraints keep the edge alive.",
                    "structural_barrier": "behavioral",
                },
            },
            gates={
                "G1_1": {"status": "pass", "reason": "Edge.type is defined."},
                "G1_4": {"status": "pass", "reason": "Hypothesis.claim is defined."},
                "G1_WF": {"status": "pending", "reason": "Walk-forward plan still needs detail."},
            },
            active_checks={"stage_1": ["walk_forward_emphasis"]},
        )

        summary = build_stage_summary(
            1,
            {
                "name": "edge_existence",
                "summary": {
                    "objective": "Edge objective",
                    "next_stage_handoff": ["next-1"],
                },
            },
            kg_state,
        )

        self.assertEqual(summary["stage"], 1)
        self.assertEqual(summary["objective"], "Edge objective")
        self.assertEqual(summary["resolved_facts"]["deferred"]["universe_anchor"], "KOSPI")
        self.assertEqual(summary["resolved_facts"]["edge"]["type"], "behavioral")
        self.assertEqual(summary["active_checks"], ["walk_forward_emphasis"])
        self.assertEqual(summary["gates"]["passed"][0], "G1_1")
        self.assertIn("Walk-forward plan still needs detail.", summary["open_questions"])

    def test_stage_summary_includes_only_semantic_relations(self) -> None:
        kg_state = KGState(
            relations=[
                {
                    "stage": 1,
                    "subject": "Edge",
                    "predicate": "grounded_in",
                    "object": "Hypothesis",
                    "scope": "structural",
                },
                {
                    "stage": 2,
                    "subject": "ReturnDecomposition",
                    "predicate": "supports",
                    "object": "Hypothesis.claim",
                    "scope": "semantic",
                },
                {
                    "stage": 1,
                    "subject": "Hypothesis",
                    "predicate": "explains",
                    "object": "MarketInefficiency.persistence",
                    "scope": "semantic",
                },
            ]
        )

        summary = build_stage_summary(
            1,
            {
                "name": "edge_existence",
                "summary": {
                    "objective": "Edge objective",
                    "next_stage_handoff": [],
                },
            },
            kg_state,
        )

        self.assertEqual(
            summary["relations"],
            [
                {
                    "subject": "Hypothesis",
                    "predicate": "explains",
                    "object": "MarketInefficiency.persistence",
                }
            ],
        )

    def test_build_prompt_kg_snapshot_prefers_current_stage_semantic_view(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={"alpha_family": "event_driven"},
            deferred={"signal_source": "price_volume"},
            entities={
                "Hypothesis": {"claim": "Theme chasing persists."},
                "ReturnDecomposition": {"method": "event_study"},
                "event_definition_consistency": {"tests": "Hypothesis.mechanism"},
            },
            gates={
                "G2_P1": {"status": "pass", "reason": "ok"},
                "G2_P3": {"status": "pending", "reason": "missing method"},
            },
            active_checks={"stage_2": ["event_definition_consistency"]},
            relations=[
                {
                    "stage": 1,
                    "subject": "Edge",
                    "predicate": "grounded_in",
                    "object": "Hypothesis",
                    "scope": "structural",
                },
                {
                    "stage": 2,
                    "subject": "event_definition_consistency",
                    "predicate": "tests",
                    "object": "Hypothesis.mechanism",
                    "scope": "semantic",
                },
            ],
            completed_stages=[0, 1],
        )

        snapshot = build_prompt_kg_snapshot(2, kg_state)

        self.assertEqual(snapshot["stage"], 2)
        self.assertEqual(snapshot["active_checks"], ["event_definition_consistency"])
        self.assertEqual(snapshot["gates"]["passed_ids"], ["G2_P1"])
        self.assertEqual(
            snapshot["relations"],
            [
                {
                    "subject": "event_definition_consistency",
                    "predicate": "tests",
                    "object": "Hypothesis.mechanism",
                }
            ],
        )

    def test_format_stage_summaries_orders_by_stage(self) -> None:
        payload = {
            "stage_2": {"stage": 2, "name": "decomposition"},
            "stage_0": {"stage": 0, "name": "classification"},
        }

        rendered = format_stage_summaries(payload)

        self.assertLess(rendered.find('"stage": 0'), rendered.find('"stage": 2'))

    def test_prune_session_contents_keeps_recent_turns(self) -> None:
        contents = [
            types.Content(role="user", parts=[types.Part(text="turn-0 user")]),
            types.Content(role="model", parts=[types.Part(text="turn-0 model")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="update_type_vector",
                            args={"dimension": "alpha_family", "value": "event_driven"},
                        )
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="update_type_vector",
                            response={"result": "ok"},
                        )
                    )
                ],
            ),
            types.Content(role="model", parts=[types.Part(text="turn-0 done")]),
            types.Content(role="user", parts=[types.Part(text="turn-1 user")]),
            types.Content(role="model", parts=[types.Part(text="turn-1 model")]),
            types.Content(role="user", parts=[types.Part(text="turn-2 user")]),
            types.Content(role="model", parts=[types.Part(text="turn-2 model")]),
        ]

        pruned = prune_session_contents(contents, keep_recent_turns=2)

        self.assertEqual(pruned[0].parts[0].text, "turn-1 user")
        self.assertEqual(pruned[-1].parts[0].text, "turn-2 model")

    def test_prune_session_contents_drops_leading_function_response(self) -> None:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="update_type_vector",
                            response={"result": "ok"},
                        )
                    )
                ],
            ),
            types.Content(role="user", parts=[types.Part(text="turn-1 user")]),
            types.Content(role="model", parts=[types.Part(text="turn-1 model")]),
        ]

        pruned = prune_session_contents(contents, keep_recent_turns=4)

        self.assertEqual(pruned[0].role, "user")
        self.assertEqual(pruned[0].parts[0].text, "turn-1 user")


if __name__ == "__main__":
    unittest.main()
