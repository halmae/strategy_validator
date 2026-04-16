import unittest

from google.genai import types

from agent.kg_store import KGState
from agent.session_store import prune_session_contents
from agent.stage_summary import build_stage_summary, format_stage_summaries


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

        summary = build_stage_summary(1, {"name": "edge_existence"}, kg_state)

        self.assertEqual(summary["stage"], 1)
        self.assertEqual(summary["resolved_facts"]["deferred"]["universe_anchor"], "KOSPI")
        self.assertEqual(summary["resolved_facts"]["edge"]["type"], "behavioral")
        self.assertEqual(summary["active_checks"], ["walk_forward_emphasis"])
        self.assertEqual(summary["gates"]["passed"][0]["id"], "G1_1")
        self.assertIn("Walk-forward plan still needs detail.", summary["open_questions"])

    def test_format_stage_summaries_orders_by_stage(self) -> None:
        payload = {
            "stage_2": {"stage": 2, "name": "decomposition"},
            "stage_0": {"stage": 0, "name": "classification"},
        }

        rendered = format_stage_summaries(payload)

        self.assertLess(rendered.find('"stage": 0'), rendered.find('"stage": 2'))

    def test_prune_session_contents_keeps_recent_messages(self) -> None:
        contents = [
            types.Content(role="user", parts=[types.Part(text=f"message-{idx}")])
            for idx in range(10)
        ]

        pruned = prune_session_contents(contents, keep_recent_messages=4)

        self.assertEqual(len(pruned), 4)
        self.assertEqual(pruned[0].parts[0].text, "message-6")
        self.assertEqual(pruned[-1].parts[0].text, "message-9")


if __name__ == "__main__":
    unittest.main()
