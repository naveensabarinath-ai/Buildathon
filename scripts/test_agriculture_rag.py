import unittest
from unittest.mock import patch

from scripts import agriculture_rag


class AgricultureRagTests(unittest.TestCase):
    def test_scheme_discovery_question_ignores_selected_scheme_filter(self):
        with patch.object(
            agriculture_rag.graph_retriever,
            "retrieve_documents",
            return_value=([], {"graph_matches": 0}, {"prompt_tokens": 0, "completion_tokens": 0}),
        ) as retrieve:
            agriculture_rag.select_evidence(
                object(),
                object(),
                "Any Schemes related to Chief Minister's name?",
                k=5,
                override_scheme_id="65",
                catalog={"65": "Uzhavar sandhai"},
            )

        self.assertIsNone(retrieve.call_args.kwargs["scheme_id"])
        self.assertTrue(retrieve.call_args.kwargs["scheme_discovery"])


if __name__ == "__main__":
    unittest.main()
