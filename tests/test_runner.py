import unittest
from types import SimpleNamespace

from google.genai import types

from agent.kg_store import KGState
from agent.runner import run_turn
from agent.schema_loader import load_schema


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate_content(self, **_kwargs):
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


def _response(parts):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=types.Content(role="model", parts=parts))
        ]
    )


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classification_schema = load_schema("classification")
        cls.routing = load_schema("routing")
        cls.model_config = {
            "name": "fake-model",
            "max_output_tokens": 128,
            "temperature": 0,
        }

    def test_run_turn_returns_text_without_function_calls(self) -> None:
        kg_state = KGState(stage=0)
        client = _FakeClient([_response([types.Part(text="hello")])])

        text, contents = run_turn(
            client,
            [types.Content(role="user", parts=[types.Part(text="start")])],
            "system",
            kg_state,
            stage=0,
            schema=self.classification_schema,
            routing=self.routing,
            max_implemented_stage=2,
            model_config=self.model_config,
        )

        self.assertEqual(text, "hello")
        self.assertEqual(contents[-1].role, "model")

    def test_run_turn_processes_function_call_loop(self) -> None:
        kg_state = KGState(stage=0)
        client = _FakeClient(
            [
                _response(
                    [
                        types.Part(
                            function_call=types.FunctionCall(
                                name="update_type_vector",
                                args={
                                    "dimension": "alpha_family",
                                    "value": "event_driven",
                                },
                            )
                        )
                    ]
                ),
                _response([types.Part(text="updated")]),
            ]
        )

        text, contents = run_turn(
            client,
            [types.Content(role="user", parts=[types.Part(text="start")])],
            "system",
            kg_state,
            stage=0,
            schema=self.classification_schema,
            routing=self.routing,
            max_implemented_stage=2,
            model_config=self.model_config,
        )

        self.assertEqual(text, "updated")
        self.assertEqual(kg_state.type_vector["alpha_family"], "event_driven")
        self.assertEqual(contents[-2].role, "user")
        self.assertEqual(
            contents[-2].parts[0].function_response.name,
            "update_type_vector",
        )


if __name__ == "__main__":
    unittest.main()
