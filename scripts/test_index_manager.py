import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from scripts import graph_retriever, index_manager, scheme_documents


class IndexManagerTests(unittest.TestCase):
    def test_empty_subsidy_pattern_is_explicit_in_source_chunk(self):
        documents = scheme_documents.scheme_to_documents(
            {
                "scheme_id": "65",
                "scheme_name": "Uzhavar sandhai",
                "details": [
                    {
                        "Input Type": "Farmer Producer Organisations",
                        "Componenet": "Financial assistance to FPOs",
                        "Pattern of Subsidy": "",
                    }
                ],
            }
        )

        detail = next(
            document
            for document in documents
            if document.metadata["content_type"] == "scheme_detail"
        )
        self.assertIn("Pattern of Subsidy: Not specified in source.", detail.page_content)

    def test_build_extracts_in_parallel_reports_progress_and_publishes(self):
        documents = [
            Document(
                page_content=f"Scheme {index} provides an agriculture benefit.",
                metadata={
                    "scheme_id": str(index),
                    "scheme_name": f"Scheme {index}",
                    "content_type": "scheme_description",
                },
            )
            for index in range(33)
        ]
        source_data = [
            {"scheme_id": str(index), "scheme_name": f"Scheme {index}"}
            for index in range(33)
        ]
        active_calls = 0
        max_active_calls = 0
        call_lock = threading.Lock()
        progress = []

        def fake_extract_facts(_llm, _text):
            nonlocal active_calls, max_active_calls
            with call_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            time.sleep(0.02)
            with call_lock:
                active_calls -= 1
            return graph_retriever.ExtractedFacts(), {
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "indexes").mkdir()
            graph_path = root / "indexes" / "graph.json"
            manifest_path = root / "indexes" / "manifest.txt"
            database_path = root / "data" / "admin.db"
            with (
                patch.object(index_manager, "ROOT", root),
                patch.object(index_manager, "GRAPH", graph_path),
                patch.object(index_manager, "MANIFEST", manifest_path),
                patch.object(index_manager, "DB", database_path),
                patch.object(index_manager, "load_chunks", return_value=(source_data, documents)),
                patch.object(index_manager, "ChatOpenAI", return_value=object()),
                patch.object(index_manager, "EXTRACTION_BATCH_SIZE", 8),
                patch.object(
                    graph_retriever,
                    "extract_facts",
                    side_effect=fake_extract_facts,
                ),
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            ):
                result = index_manager.build(
                    progress_callback=lambda completed, total: progress.append(
                        (completed, total)
                    )
                )
                loaded = graph_retriever.load_graph(
                    graph_path,
                    expected_digest=manifest_path.read_text(encoding="utf-8"),
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extraction_calls"], len(documents))
        self.assertEqual(progress, [(index, len(documents)) for index in range(1, 34)])
        self.assertGreater(max_active_calls, 1)
        self.assertLessEqual(max_active_calls, index_manager.EXTRACTION_WORKERS)
        self.assertEqual(len(graph_retriever.scheme_catalog(loaded)), len(documents))


if __name__ == "__main__":
    unittest.main()
