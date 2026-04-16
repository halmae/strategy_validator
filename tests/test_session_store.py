import tempfile
import unittest
from pathlib import Path

from google.genai import types

from agent.session_store import load_session_contents, save_session_contents


class SessionStoreTests(unittest.TestCase):
    def test_session_contents_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_path = Path(tmpdir) / "demo.md"
            strategy_path.write_text("# Demo", encoding="utf-8")

            contents = [
                types.Content(role="user", parts=[types.Part(text="hello")]),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(text="world"),
                        types.Part(
                            function_call=types.FunctionCall(
                                name="update_type_vector",
                                args={"dimension": "alpha_family", "value": "event_driven"},
                            )
                        ),
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
            ]

            save_session_contents(strategy_path, contents)
            restored = load_session_contents(strategy_path)

            self.assertIsNotNone(restored)
            self.assertEqual(restored[0].parts[0].text, "hello")
            self.assertEqual(restored[1].parts[0].text, "world")
            self.assertEqual(restored[1].parts[1].function_call.name, "update_type_vector")
            self.assertEqual(restored[2].parts[0].function_response.name, "update_type_vector")


if __name__ == "__main__":
    unittest.main()
