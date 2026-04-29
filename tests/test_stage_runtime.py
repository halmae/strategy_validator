import unittest

from agent.kg_store import KGState
from agent.runner import process_function_call
from agent.schema_loader import load_schema
from agent.stage_runtime import RuntimeValidationError, sync_runtime_state


class StageRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classification_schema = load_schema("classification")
        cls.stage_1_schema = load_schema("stage_1")
        cls.stage_2_schema = load_schema("stage_2")
        cls.stage_3_schema = load_schema("stage_3")
        cls.stage_4_schema = load_schema("stage_4")
        cls.routing = load_schema("routing")
        cls.max_implemented_stage = 4

    def process_call(self, name: str, args: dict, kg_state: KGState, stage: int, schema: dict) -> str:
        return process_function_call(
            name,
            args,
            kg_state,
            stage=stage,
            schema=schema,
            routing=self.routing,
            max_implemented_stage=self.max_implemented_stage,
        )

    def test_stage_0_blocks_advance_until_type_vector_is_complete(self) -> None:
        kg_state = KGState(stage=0)
        result = self.process_call(
            "advance_stage",
            {"from_stage": 0, "to_stage": 1, "summary": "done"},
            kg_state,
            stage=0,
            schema=self.classification_schema,
        )

        self.assertIn("Validation error", result)
        self.assertEqual(kg_state.stage, 0)

    def test_stage_0_mark_gate_is_noop(self) -> None:
        kg_state = KGState(stage=0)

        result = self.process_call(
            "mark_gate",
            {"gate_id": "G0_1", "status": "pending"},
            kg_state,
            stage=0,
            schema=self.classification_schema,
        )

        self.assertEqual(result, "Stage 0 has no gates; mark_gate ignored.")
        self.assertEqual(kg_state.gates, {})

    def test_stage_0_out_of_scope_reason_uses_user_facing_message(self) -> None:
        kg_state = KGState(
            stage=0,
            type_vector={"execution_mode": "discretionary"},
        )

        sync_runtime_state(0, self.classification_schema, self.routing, kg_state)

        self.assertTrue(kg_state.out_of_scope)
        self.assertEqual(
            kg_state.out_of_scope_reason,
            "Discretionary 전략은 이 시스템의 검증 범위 밖입니다. "
            "Signal을 rule로 표현 가능한 경우 semi_systematic으로 재분류하세요.",
        )
        self.assertNotIn("Warn the user", kg_state.out_of_scope_reason)

    def test_stage_1_checks_are_activated_from_routing(self) -> None:
        kg_state = KGState(
            stage=1,
            type_vector={
                "alpha_family": "carry",
                "exposure_structure": "market_neutral",
                "asset_class": "futures",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
        )

        sync_runtime_state(1, self.stage_1_schema, self.routing, kg_state)

        self.assertEqual(
            kg_state.active_checks["stage_1"],
            ["beta_neutrality_assumption", "risk_premium_decomposition"],
        )

    def test_stage_1_checks_can_also_activate_from_stage_schema_triggers(self) -> None:
        kg_state = KGState(
            stage=1,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            deferred={"universe_scope": "concentrated"},
        )

        sync_runtime_state(1, self.stage_1_schema, self.routing, kg_state)

        self.assertEqual(kg_state.active_checks["stage_1"], ["walk_forward_emphasis"])

    def test_stage_1_rejects_invalid_deferred_value(self) -> None:
        kg_state = KGState(stage=1)
        result = self.process_call(
            "update_deferred",
            {"dimension": "universe_scope", "value": "mega"},
            kg_state,
            stage=1,
            schema=self.stage_1_schema,
        )

        self.assertIn("Validation error", result)
        self.assertNotIn("universe_scope", kg_state.deferred)

    def test_stage_1_deferred_values_are_exposed_as_gate(self) -> None:
        kg_state = KGState(
            stage=1,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "Edge": {
                    "type": "behavioral",
                    "direction": "long",
                    "horizon": "하루",
                },
                "Hypothesis": {
                    "claim": "This edge exists because attention reacts slowly.",
                    "mechanism": "Retail attention diffuses over the trading day.",
                    "falsifiable": True,
                    "falsification_condition": "No abnormal return remains.",
                },
                "MarketInefficiency": {
                    "persistence": "Attention constraints prevent immediate arbitrage.",
                    "structural_barrier": "behavioral",
                },
            },
        )

        sync_runtime_state(1, self.stage_1_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G1_0"]["status"], "pending")
        self.assertEqual(kg_state.gates["G1_3"]["status"], "pass")

        result = self.process_call(
            "advance_stage",
            {"from_stage": 1, "to_stage": 2, "summary": "stage done"},
            kg_state,
            stage=1,
            schema=self.stage_1_schema,
        )

        self.assertIn("G1_0", result)
        self.assertEqual(kg_state.stage, 1)

    def test_stage_1_normalizes_boolean_entity_values(self) -> None:
        kg_state = KGState(
            stage=1,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
            },
        )

        result = self.process_call(
            "update_kg_entity",
            {"entity": "Hypothesis", "property": "falsifiable", "value": "true"},
            kg_state,
            stage=1,
            schema=self.stage_1_schema,
        )

        self.assertIn("updated", result)
        self.assertIs(kg_state.entities["Hypothesis"]["falsifiable"], True)
        self.assertEqual(kg_state.gates["G1_6"]["status"], "pass")

    def test_stage_1_horizon_consistency_understands_korean_terms(self) -> None:
        kg_state = KGState(
            stage=1,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "quarterly_plus",
                "execution_mode": "systematic",
            },
            deferred={
                "universe_scope": "diversified",
                "universe_anchor": "KOSPI",
            },
            entities={
                "Edge": {
                    "type": "behavioral",
                    "direction": "long",
                    "horizon": "분기 단위",
                },
            },
        )

        sync_runtime_state(1, self.stage_1_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G1_3"]["status"], "pass")

    def test_stage_1_completion_advances_to_stage_2(self) -> None:
        kg_state = KGState(
            stage=1,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            deferred={
                "universe_scope": "diversified",
                "universe_anchor": "KOSPI",
            },
            entities={
                "Edge": {
                    "type": "behavioral",
                    "direction": "long",
                    "horizon": "3 days",
                },
                "Hypothesis": {
                    "claim": "This edge exists because post-event repricing is slow.",
                    "mechanism": "Investors underreact to the event signal.",
                    "falsifiable": True,
                    "falsification_condition": "No abnormal return remains after the event.",
                },
                "MarketInefficiency": {
                    "persistence": "Capacity and attention constraints keep the edge alive.",
                    "structural_barrier": "information",
                },
            },
        )

        sync_runtime_state(1, self.stage_1_schema, self.routing, kg_state)
        result = self.process_call(
            "advance_stage",
            {"from_stage": 1, "to_stage": 2, "summary": "stage done"},
            kg_state,
            stage=1,
            schema=self.stage_1_schema,
        )

        self.assertIn("Stage 1 -> 2", result)
        self.assertEqual(kg_state.stage, 2)
        self.assertFalse(kg_state.workflow_complete)
        self.assertIn(1, kg_state.completed_stages)
        self.assertIn(
            {
                "stage": 1,
                "subject": "Edge",
                "predicate": "grounded_in",
                "object": "Hypothesis",
                "scope": "structural",
            },
            kg_state.relations,
        )
        self.assertIn(
            {
                "stage": 1,
                "subject": "Edge",
                "predicate": "exploits",
                "object": "MarketInefficiency",
                "scope": "structural",
            },
            kg_state.relations,
        )

    def test_stage_2_with_no_active_checks_can_advance(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "carry",
                "exposure_structure": "long_only",
                "asset_class": "futures",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            deferred={"signal_source": "price_volume"},
            entities={
                "ReturnDecomposition": {
                    "method": "attribution",
                    "sample_period": "2020-01 ~ 2024-12",
                }
            },
        )

        sync_runtime_state(2, self.stage_2_schema, self.routing, kg_state)
        self.assertEqual(kg_state.active_checks["stage_2"], [])
        self.assertEqual(kg_state.gates["G2_P3"]["status"], "pass")

        result = self.process_call(
            "advance_stage",
            {"from_stage": 2, "to_stage": 3, "summary": "stage done"},
            kg_state,
            stage=2,
            schema=self.stage_2_schema,
        )

        self.assertIn("Stage 2 -> 4", result)
        self.assertIn(
            "Requested Stage 3, but routing resolved next stage to 4.",
            result,
        )
        self.assertEqual(kg_state.stage, 4)
        self.assertFalse(kg_state.workflow_complete)
        self.assertIn(2, kg_state.completed_stages)
        self.assertIn(3, kg_state.skipped_stages)

    def test_stage_2_event_driven_requires_signal_score_before_stage_3_route(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            deferred={"signal_source": "price_volume"},
            entities={
                "ReturnDecomposition": {
                    "method": "event_study",
                    "sample_period": "2020-01 ~ 2024-12",
                },
                "event_definition_consistency": {
                    "tests": "Hypothesis.mechanism",
                    "evidence_type": "attestation",
                    "criterion": "Event labels are stable.",
                    "method": "Review event labels.",
                },
                "look_ahead_event_timing": {
                    "tests": "Hypothesis.falsification_condition",
                    "evidence_type": "attestation",
                    "criterion": "No future timestamps leak.",
                    "method": "Review timestamp source.",
                },
            },
        )

        sync_runtime_state(2, self.stage_2_schema, self.routing, kg_state)
        result = self.process_call(
            "advance_stage",
            {"from_stage": 2, "to_stage": 3, "summary": "stage done"},
            kg_state,
            stage=2,
            schema=self.stage_2_schema,
        )

        self.assertIn("Routing prerequisites are not satisfied", result)
        self.assertIn("SignalScore", result)
        self.assertEqual(kg_state.stage, 2)

    def test_stage_gate_errors_take_priority_over_prerequisite_errors(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            deferred={"signal_source": "price_volume"},
            entities={
                "ReturnDecomposition": {
                    "method": "event_study",
                    "sample_period": "unknown",
                },
            },
        )

        result = self.process_call(
            "advance_stage",
            {"from_stage": 2, "to_stage": 3, "summary": "stage done"},
            kg_state,
            stage=2,
            schema=self.stage_2_schema,
        )

        self.assertIn("Stage 2 plan gates not passed", result)
        self.assertNotIn("Routing prerequisites are not satisfied", result)

    def test_stage_2_advances_to_stage_3_when_signal_score_exists(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "carry",
                "exposure_structure": "long_only",
                "asset_class": "futures",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            deferred={"signal_source": "price_volume"},
            entities={
                "SignalScore": {
                    "definition": "Carry rank across contracts.",
                },
                "ReturnDecomposition": {
                    "method": "attribution",
                    "sample_period": "2020-01 ~ 2024-12",
                },
            },
        )

        sync_runtime_state(2, self.stage_2_schema, self.routing, kg_state)
        result = self.process_call(
            "advance_stage",
            {"from_stage": 2, "to_stage": 3, "summary": "stage done"},
            kg_state,
            stage=2,
            schema=self.stage_2_schema,
        )

        self.assertIn("Stage 2 -> 3", result)
        self.assertEqual(kg_state.stage, 3)
        self.assertFalse(kg_state.workflow_complete)
        self.assertNotIn(3, kg_state.skipped_stages)

    def test_stage_2_can_seed_signal_score_for_stage_3_routing(self) -> None:
        kg_state = KGState(stage=2)

        result = self.process_call(
            "update_kg_entity",
            {
                "entity": "SignalScore",
                "property": "definition",
                "value": "Signal is triggered when price reaches +10%.",
            },
            kg_state,
            stage=2,
            schema=self.stage_2_schema,
        )

        self.assertIn("updated", result)
        self.assertEqual(
            kg_state.entities["SignalScore"]["definition"],
            "Signal is triggered when price reaches +10%.",
        )

    def test_stage_2_and_stage_3_signal_direction_enums_stay_in_sync(self) -> None:
        stage_2_values = (
            self.stage_2_schema["entities"]["SignalScore"]["properties"]
            ["signal_direction"]["values"]
        )
        stage_3_values = (
            self.stage_3_schema["entities"]["SignalScore"]["properties"]
            ["signal_direction"]["values"]
        )

        self.assertEqual(stage_2_values, stage_3_values)

    def test_stage_2_placeholders_do_not_satisfy_plan_gates(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
            },
            deferred={"signal_source": "price_volume"},
            entities={
                "ReturnDecomposition": {
                    "method": "event_study",
                    "sample_period": "unknown",
                },
                "event_definition_consistency": {
                    "tests": "unknown",
                    "evidence_type": "attestation",
                    "criterion": "unknown",
                    "method": "unknown",
                },
                "look_ahead_event_timing": {
                    "tests": "tbd",
                    "evidence_type": "attestation",
                    "criterion": "not sure",
                    "method": "unknown",
                },
            },
        )

        sync_runtime_state(2, self.stage_2_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G2_P1"]["status"], "pass")
        self.assertEqual(kg_state.gates["G2_P2"]["status"], "pending")
        self.assertEqual(kg_state.gates["G2_P3"]["status"], "pending")

        result = self.process_call(
            "advance_stage",
            {"from_stage": 2, "to_stage": 3, "summary": "stage done"},
            kg_state,
            stage=2,
            schema=self.stage_2_schema,
        )

        self.assertIn("Validation error", result)
        self.assertEqual(kg_state.stage, 2)
        self.assertFalse(kg_state.workflow_complete)

    def test_stage_2_derives_fixed_predicates_from_checks_and_decomposition(self) -> None:
        kg_state = KGState(
            stage=2,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
            },
            deferred={"signal_source": "price_volume"},
            entities={
                "Hypothesis": {
                    "claim": "Theme chasing creates temporary inefficiencies.",
                },
                "ReturnDecomposition": {
                    "method": "event_study",
                    "sample_period": "2020-01 ~ 2024-12",
                    "supports_hypothesis": "supports",
                },
                "event_definition_consistency": {
                    "tests": "Hypothesis.mechanism",
                    "evidence_type": "attestation",
                    "criterion": "Event labels are stable across samples.",
                    "method": "Manual review of event tagging rules.",
                },
                "look_ahead_event_timing": {
                    "tests": "Hypothesis.falsification_condition",
                    "evidence_type": "attestation",
                    "criterion": "No future timestamps leak into event windows.",
                    "method": "Verify event timestamps against source system logs.",
                },
            },
        )

        sync_runtime_state(2, self.stage_2_schema, self.routing, kg_state)

        self.assertIn(
            {
                "stage": 2,
                "subject": "event_definition_consistency",
                "predicate": "tests",
                "object": "Hypothesis.mechanism",
                "scope": "semantic",
            },
            kg_state.relations,
        )
        self.assertIn(
            {
                "stage": 2,
                "subject": "ReturnDecomposition",
                "predicate": "supports",
                "object": "Hypothesis.claim",
                "object_value": "Theme chasing creates temporary inefficiencies.",
                "scope": "semantic",
            },
            kg_state.relations,
        )

    def test_stage_3_activates_signal_quality_checks_from_routing(self) -> None:
        kg_state = KGState(
            stage=3,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
            },
            entities={"SignalScore": {"definition": "Price reaches +10% intraday."}},
        )

        sync_runtime_state(3, self.stage_3_schema, self.routing, kg_state)

        self.assertEqual(
            kg_state.active_checks["stage_3"],
            ["event_signal_alignment", "signal_decay_intraday"],
        )

    def test_stage_3_plan_gates_can_advance_with_evidence_pending(self) -> None:
        kg_state = KGState(
            stage=3,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
            },
            entities={
                "Hypothesis": {
                    "claim": "Theme attention creates short-lived intraday continuation.",
                    "mechanism": "Retail attention clusters into stocks already up sharply.",
                },
                "SignalScore": {
                    "definition": "Signal is 1 when price first reaches +10% from prior close.",
                    "signal_direction": "higher_better",
                    "measurement_method": "Event-study return from +10% trigger to +15% or 10:00.",
                    "evaluation_window": "09:00~10:00",
                },
                "event_signal_alignment": {
                    "tests": "Hypothesis.mechanism",
                    "evidence_type": "attestation",
                    "criterion": "+10% trigger must represent theme-attention demand.",
                    "method": "Review event definition against strategy hypothesis.",
                },
                "signal_decay_intraday": {
                    "tests": "SignalScore.evaluation_window",
                    "evidence_type": "quantitative",
                    "criterion": "Forward return should decay after the first hour.",
                    "method": "Bucket forward returns by minutes since trigger.",
                },
            },
        )

        sync_runtime_state(3, self.stage_3_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G3_0"]["status"], "pass")
        self.assertEqual(kg_state.gates["G3_P1"]["status"], "pass")
        self.assertEqual(kg_state.gates["G3_P2"]["status"], "pass")
        self.assertEqual(kg_state.gates["G3_E1"]["status"], "pending")
        self.assertEqual(kg_state.gates["G3_E2"]["status"], "pending")

        result = self.process_call(
            "advance_stage",
            {"from_stage": 3, "to_stage": 4, "summary": "stage done"},
            kg_state,
            stage=3,
            schema=self.stage_3_schema,
        )

        self.assertIn("Stage 3 -> 4", result)
        self.assertEqual(kg_state.stage, 4)
        self.assertFalse(kg_state.workflow_complete)
        self.assertIn(3, kg_state.completed_stages)
        self.assertIn(
            {
                "stage": 3,
                "subject": "SignalScore",
                "predicate": "operationalizes",
                "object": "Hypothesis.mechanism",
                "object_value": "Retail attention clusters into stocks already up sharply.",
                "scope": "semantic",
            },
            kg_state.relations,
        )

    def test_stage_3_blocks_advance_until_active_check_plans_are_defined(self) -> None:
        kg_state = KGState(
            stage=3,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "intraday",
                "execution_mode": "systematic",
            },
            entities={
                "Hypothesis": {
                    "claim": "Theme attention creates short-lived intraday continuation.",
                    "mechanism": "Retail attention clusters into stocks already up sharply.",
                },
                "SignalScore": {
                    "definition": "Signal is 1 when price reaches +10%.",
                    "signal_direction": "higher_better",
                    "measurement_method": "Event-study forward returns.",
                    "evaluation_window": "09:00~10:00",
                },
            },
        )

        sync_runtime_state(3, self.stage_3_schema, self.routing, kg_state)
        result = self.process_call(
            "advance_stage",
            {"from_stage": 3, "to_stage": 4, "summary": "stage done"},
            kg_state,
            stage=3,
            schema=self.stage_3_schema,
        )

        self.assertIn("G3_P2", result)
        self.assertFalse(kg_state.workflow_complete)

    def test_stage_3_evidence_accepts_explicit_not_applicable_text(self) -> None:
        kg_state = KGState(
            stage=3,
            type_vector={
                "alpha_family": "carry",
                "exposure_structure": "long_only",
                "asset_class": "futures",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "SignalScore": {
                    "definition": "Carry rank across contracts.",
                    "signal_direction": "higher_better",
                    "measurement_method": "Hit-rate by carry decile.",
                    "evaluation_window": "1 month",
                    "ic_metric": "not applicable",
                    "hit_rate": "not applicable",
                    "decay_profile": "Monthly decay is evaluated at rebalance.",
                    "supports_hypothesis": "inconclusive",
                    "reasoning": "Evidence is pending but IC is not the chosen metric.",
                },
            },
        )

        sync_runtime_state(3, self.stage_3_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G3_E1"]["status"], "pass")

    def test_unknown_relationship_kind_fails_loudly(self) -> None:
        schema = {
            "relationships": [
                {
                    "kind": "unsupported_kind",
                    "subject": "SignalScore",
                    "predicate": "tests",
                    "object_ref": "Hypothesis.claim",
                }
            ]
        }
        kg_state = KGState(
            stage=3,
            entities={
                "SignalScore": {"definition": "A signal."},
                "Hypothesis": {"claim": "A claim."},
            },
        )

        with self.assertRaises(RuntimeValidationError):
            sync_runtime_state(3, schema, self.routing, kg_state)

    def test_stage_4_activates_exit_checks_from_routing(self) -> None:
        kg_state = KGState(
            stage=4,
            type_vector={
                "alpha_family": "stat_arb",
                "exposure_structure": "market_neutral",
                "asset_class": "futures",
                "market_scope": "us",
                "decision_cadence": "quarterly_plus",
                "execution_mode": "systematic",
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(
            kg_state.active_checks["stage_4"],
            [
                "drawdown_duration_analysis",
                "mean_reversion_timeout",
                "rebalance_frequency_alignment",
                "roll_timing_consistency",
            ],
        )

    def test_stage_4_plan_gates_can_advance_with_evidence_pending(self) -> None:
        kg_state = KGState(
            stage=4,
            type_vector={
                "alpha_family": "cross_sectional_factor",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "Hypothesis": {
                    "claim": "Slow repricing persists after signal observation.",
                    "mechanism": "Investors underreact until the repricing window closes.",
                    "falsification_condition": "No abnormal return remains after five days.",
                },
                "Edge": {"horizon": "5 days"},
                "ExitPolicy": {
                    "exit_trigger_type": "time_stop",
                    "rationale": "Exit after the repricing window should have closed.",
                    "failure_mode_addressed": "If no abnormal return appears after five days, the thesis failed.",
                    "expected_holding_period": "5 trading days",
                },
                "DrawdownProfile": {
                    "measurement_method": "Measure max drawdown during each holding window.",
                    "acceptable_drawdown": "Max single-position drawdown should stay below 8%.",
                },
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G4_P1"]["status"], "pass")
        self.assertEqual(kg_state.gates["G4_P2"]["status"], "pass")
        self.assertEqual(kg_state.gates["G4_P3"]["status"], "pass")
        self.assertEqual(kg_state.gates["G4_P4"]["status"], "pass")
        self.assertEqual(kg_state.gates["G4_E1"]["status"], "pending")
        self.assertEqual(kg_state.gates["G4_E2"]["status"], "pending")

        result = self.process_call(
            "advance_stage",
            {"from_stage": 4, "to_stage": 5, "summary": "stage done"},
            kg_state,
            stage=4,
            schema=self.stage_4_schema,
        )

        self.assertIn("Stage 5 is not implemented yet", result)
        self.assertEqual(kg_state.stage, 4)
        self.assertTrue(kg_state.workflow_complete)
        self.assertIn(4, kg_state.completed_stages)
        self.assertIn(
            {
                "stage": 4,
                "subject": "ExitPolicy",
                "predicate": "terminates",
                "object": "Hypothesis.mechanism",
                "object_value": "Investors underreact until the repricing window closes.",
                "scope": "semantic",
            },
            kg_state.relations,
        )

    def test_stage_4_signal_reversal_requires_signal_score(self) -> None:
        kg_state = KGState(
            stage=4,
            type_vector={
                "alpha_family": "cross_sectional_factor",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "Hypothesis": {
                    "mechanism": "Signal predicts short-lived repricing.",
                },
                "ExitPolicy": {
                    "exit_trigger_type": "signal_reversal",
                    "rationale": "Exit when the entry signal reverses.",
                    "failure_mode_addressed": "Signal reversal means the original edge has ended.",
                    "expected_holding_period": "Until signal reversal.",
                },
                "DrawdownProfile": {
                    "measurement_method": "Measure drawdown until signal reversal.",
                    "acceptable_drawdown": "Max drawdown below 8%.",
                },
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G4_P4"]["status"], "pending")

        result = self.process_call(
            "advance_stage",
            {"from_stage": 4, "to_stage": 5, "summary": "stage done"},
            kg_state,
            stage=4,
            schema=self.stage_4_schema,
        )

        self.assertIn("G4_P4", result)
        self.assertFalse(kg_state.workflow_complete)

    def test_stage_4_signal_reversal_gate_is_pending_until_trigger_type_defined(self) -> None:
        kg_state = KGState(
            stage=4,
            entities={
                "DrawdownProfile": {
                    "measurement_method": "Measure holding-window drawdown.",
                    "acceptable_drawdown": "Below 8%.",
                },
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G4_P4"]["status"], "pending")
        self.assertIn(
            "exit_trigger_type",
            kg_state.gates["G4_P4"]["reason"],
        )

    def test_stage_4_signal_reversal_passes_when_signal_score_exists(self) -> None:
        kg_state = KGState(
            stage=4,
            type_vector={
                "alpha_family": "cross_sectional_factor",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "SignalScore": {"definition": "Ranked factor score."},
                "ExitPolicy": {
                    "exit_trigger_type": "signal_reversal",
                    "rationale": "Exit when the score reverses.",
                    "failure_mode_addressed": "Reversal means the ranking edge has ended.",
                    "expected_holding_period": "Until score reversal.",
                },
                "DrawdownProfile": {
                    "measurement_method": "Measure drawdown until score reversal.",
                    "acceptable_drawdown": "Max drawdown below 8%.",
                },
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G4_P4"]["status"], "pass")

    def test_stage_4_blocks_until_active_check_plans_are_defined(self) -> None:
        kg_state = KGState(
            stage=4,
            type_vector={
                "alpha_family": "event_driven",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "korea",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "Hypothesis": {"mechanism": "Event impact fades after the window closes."},
                "ExitPolicy": {
                    "exit_trigger_type": "time_stop",
                    "rationale": "Exit after event impact window closes.",
                    "failure_mode_addressed": "No repricing after the window means thesis failure.",
                    "expected_holding_period": "3 trading days",
                },
                "DrawdownProfile": {
                    "measurement_method": "Event-window max drawdown.",
                    "acceptable_drawdown": "Below 10%.",
                },
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(kg_state.active_checks["stage_4"], ["event_window_close"])

        result = self.process_call(
            "advance_stage",
            {"from_stage": 4, "to_stage": 5, "summary": "stage done"},
            kg_state,
            stage=4,
            schema=self.stage_4_schema,
        )

        self.assertIn("G4_P3", result)

    def test_stage_4_evidence_accepts_explicit_not_applicable_text(self) -> None:
        kg_state = KGState(
            stage=4,
            type_vector={
                "alpha_family": "cross_sectional_factor",
                "exposure_structure": "long_only",
                "asset_class": "equity",
                "market_scope": "us",
                "decision_cadence": "daily",
                "execution_mode": "systematic",
            },
            entities={
                "ExitPolicy": {
                    "exit_trigger_type": "time_stop",
                    "rationale": "Exit after the expected repricing window.",
                    "failure_mode_addressed": "No repricing after the window.",
                    "expected_holding_period": "5 trading days",
                    "realized_holding_period": "not applicable",
                    "supports_hypothesis": "inconclusive",
                    "reasoning": "Holding-period evidence is not available yet.",
                },
                "DrawdownProfile": {
                    "measurement_method": "Measure window drawdown.",
                    "acceptable_drawdown": "Below 8%.",
                    "realized_drawdown": "not applicable",
                    "drawdown_duration": "not applicable",
                    "reasoning": "Drawdown evidence is not available yet.",
                },
            },
        )

        sync_runtime_state(4, self.stage_4_schema, self.routing, kg_state)

        self.assertEqual(kg_state.gates["G4_E1"]["status"], "pass")
        self.assertEqual(kg_state.gates["G4_E2"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
