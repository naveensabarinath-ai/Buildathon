import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from scripts import graph_retriever


class _StructuredExtractor:
    def __init__(self, facts):
        self.facts = facts

    def invoke(self, messages):
        return {
            "parsed": self.facts,
            "raw": type("Response", (), {"usage_metadata": {"input_tokens": 7, "output_tokens": 3}})(),
            "parsing_error": None,
        }


class _FakeLLM:
    def __init__(self, facts):
        self.facts = facts

    def with_structured_output(self, schema, include_raw):
        return _StructuredExtractor(self.facts)


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.irrigation = Document(
            page_content=(
                "Scheme name: Micro Irrigation Support\n"
                "Farmers can receive subsidy for drip irrigation equipment."
            ),
            metadata={
                "scheme_id": "101",
                "scheme_name": "Micro Irrigation Support",
                "content_type": "scheme_description",
                "listing_field": "Scheme Description",
            },
        )
        self.documents = Document(
            page_content="Scheme name: Seed Support\nDocuments required for seed assistance.",
            metadata={
                "scheme_id": "202",
                "scheme_name": "Seed Support",
                "content_type": "required_documents",
                "listing_field": "Documents Required",
            },
        )
        self.irrigation_facts = graph_retriever.ExtractedFacts(
            entities=[
                {"name": "Micro Irrigation Support", "entity_type": "scheme"},
                {"name": "drip irrigation", "entity_type": "practice"},
                {"name": "subsidy", "entity_type": "benefit"},
                {"name": "farmers", "entity_type": "beneficiary"},
            ],
            relations=[
                {
                    "subject": "Micro Irrigation Support",
                    "predicate": "provides",
                    "object": "subsidy",
                },
                {
                    "subject": "subsidy",
                    "predicate": "supports",
                    "object": "drip irrigation",
                },
            ],
        )
        self.seed_facts = graph_retriever.ExtractedFacts(
            entities=[
                {"name": "Seed Support", "entity_type": "scheme"},
                {"name": "seed assistance", "entity_type": "benefit"},
                {"name": "documents", "entity_type": "document"},
            ],
            relations=[
                {
                    "subject": "Seed Support",
                    "predicate": "requires",
                    "object": "documents",
                }
            ],
        )
        self.build_progress = []
        self.graph, self.build_usage = graph_retriever.build_graph(
            [self.irrigation, self.documents],
            "source-version",
            facts_loader=self.facts_for_document,
            progress_callback=lambda completed, total: self.build_progress.append(
                (completed, total)
            ),
        )

    def facts_for_document(self, document):
        facts = (
            self.irrigation_facts
            if document.metadata["scheme_id"] == "101"
            else self.seed_facts
        )
        return facts, {"prompt_tokens": 5, "completion_tokens": 2}

    def test_graph_contains_typed_entities_and_supported_relations(self):
        entity_nodes = [
            attributes for _, attributes in self.graph.nodes(data=True)
            if attributes.get("kind") == "entity"
        ]
        fact_edges = [
            attributes for _, _, attributes in self.graph.edges(data=True)
            if attributes.get("relation") == "FACT"
        ]

        self.assertTrue(any(node["name"] == "drip irrigation" for node in entity_nodes))
        self.assertTrue(any(edge["predicate"] == "supports" for edge in fact_edges))
        self.assertEqual(self.build_usage, {"prompt_tokens": 10, "completion_tokens": 4})
        self.assertEqual(self.build_progress, [(1, 2), (2, 2)])

    def test_query_entity_traversal_returns_graph_stored_source_text(self):
        query_facts = graph_retriever.ExtractedFacts(
            entities=[
                {"name": "drip irrigation", "entity_type": "practice"},
                {"name": "subsidy", "entity_type": "benefit"},
            ],
            relations=[
                {
                    "subject": "subsidy",
                    "predicate": "supports",
                    "object": "drip irrigation",
                }
            ],
        )
        documents, diagnostics, usage = graph_retriever.retrieve_documents(
            self.graph,
            "What subsidy supports drip irrigation?",
            _FakeLLM(query_facts),
            k=3,
        )

        self.assertEqual(documents[0].page_content, self.irrigation.page_content)
        self.assertEqual(diagnostics["strategy"], "knowledge_graph_entities_relations")
        self.assertEqual(diagnostics["matched_entities"], ["drip irrigation", "subsidy"])
        self.assertEqual(usage, {"prompt_tokens": 7, "completion_tokens": 3})

    def test_scheme_and_field_filter_returns_all_full_graph_records(self):
        documents, diagnostics, usage = graph_retriever.retrieve_documents(
            self.graph,
            "Which documents are required?",
            _FakeLLM(graph_retriever.ExtractedFacts()),
            scheme_id="202",
            content_type="required_documents",
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, self.documents.page_content)
        self.assertEqual(diagnostics["strategy"], "knowledge_graph_exact_field")
        self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})

    def test_scheme_filter_prevents_cross_scheme_evidence(self):
        query_facts = graph_retriever.ExtractedFacts(
            entities=[
                {"name": "drip irrigation", "entity_type": "practice"},
                {"name": "subsidy", "entity_type": "benefit"},
            ]
        )
        documents, _, _ = graph_retriever.retrieve_documents(
            self.graph,
            "drip irrigation subsidy",
            _FakeLLM(query_facts),
            k=5,
            scheme_id="202",
        )

        self.assertEqual(documents, [])

    def test_resolved_scheme_falls_back_when_query_entities_do_not_match(self):
        query_facts = graph_retriever.ExtractedFacts(
            entities=[
                {"name": "Uzhavar Sandhai Scheme", "entity_type": "scheme"},
            ]
        )
        documents, diagnostics, _ = graph_retriever.retrieve_documents(
            self.graph,
            "Explain more about Uzhavar Sandhai Scheme",
            _FakeLLM(query_facts),
            k=5,
            scheme_id="202",
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, self.documents.page_content)
        self.assertTrue(diagnostics["scheme_fallback"])
        self.assertEqual(diagnostics["strategy"], "knowledge_graph_scheme_fallback")

    def test_scheme_name_search_finds_all_matching_schemes(self):
        documents = [
            Document(
                page_content=f"Scheme name: {name}\nScheme description: {name}",
                metadata={
                    "scheme_id": scheme_id,
                    "scheme_name": name,
                    "content_type": "scheme_description",
                },
            )
            for scheme_id, name in (
                ("74", "Chief Minister's Scheme of Solar Powered Pump-set"),
                ("153", "Chief Minister\ufffds Mannuyir Kaathu Mannuyir Kaappom Scheme"),
                ("159", "Kalaignar All Village Integrated Development Program 2024-25"),
                ("60", "Kalaignar\ufffds All Village Integrated Agriculture Development Programme (KAVIADP) 2022-2023"),
                ("65", "Uzhavar sandhai"),
            )
        ]
        graph, _ = graph_retriever.build_graph(
            documents,
            "source-version",
            facts_loader=lambda _: (
                graph_retriever.ExtractedFacts(),
                {"prompt_tokens": 0, "completion_tokens": 0},
            ),
        )

        results, names = graph_retriever._scheme_name_search(
            graph,
            "Any Scheme in the name of Chief Minister's ?",
            k=5,
        )

        self.assertEqual(
            {document.metadata["scheme_id"] for document in results},
            {"74", "153"},
        )
        self.assertEqual(set(names), {
            "Chief Minister's Scheme of Solar Powered Pump-set",
            "Chief Minister\ufffds Mannuyir Kaathu Mannuyir Kaappom Scheme",
        })

        kalaignar_results, kalaignar_names = graph_retriever._scheme_name_search(
            graph,
            "Explain scheme related to Kalaignar",
            k=5,
        )
        self.assertEqual(
            {document.metadata["scheme_id"] for document in kalaignar_results},
            {"60", "159"},
        )
        self.assertEqual(len(kalaignar_names), 2)

        listed_results, listed_names = graph_retriever._scheme_name_search(
            graph,
            "List schemes related to Kalaignar",
            k=5,
        )
        self.assertEqual(
            {document.metadata["scheme_id"] for document in listed_results},
            {"60", "159"},
        )
        self.assertEqual(set(kalaignar_names), set(listed_names))

    def test_graph_serialization_enforces_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.json"
            graph_retriever.save_graph(self.graph, graph_path)
            loaded = graph_retriever.load_graph(
                graph_path, expected_digest="source-version"
            )

            self.assertEqual(loaded.number_of_nodes(), self.graph.number_of_nodes())
            self.assertEqual(graph_retriever.scheme_catalog(loaded)["101"], "Micro Irrigation Support")
            with self.assertRaisesRegex(ValueError, "stale"):
                graph_retriever.load_graph(graph_path, expected_digest="wrong-version")


if __name__ == "__main__":
    unittest.main()
