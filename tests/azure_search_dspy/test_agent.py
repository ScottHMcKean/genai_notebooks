"""Unit tests for Information Extraction Agent"""

import pytest
import dspy
from unittest.mock import Mock, patch, MagicMock
from src.azure_search_dspy import (
    AzureSearchRM,
    InformationExtractionAgent,
    evaluate_quality,
    evaluate_agent,
)


class TestAzureSearchRM:
    """Tests for Azure Search Retrieval Module"""

    @patch("src.azure_search_dspy.retriever.SearchClient")
    def test_init(self, mock_search_client):
        """Test retriever initialization"""
        retriever = AzureSearchRM(
            search_endpoint="https://test.search.windows.net",
            search_key="test-key",
            index_name="test-index",
            k=5,
        )

        assert retriever.k == 5
        assert retriever.content_field == "content"
        assert retriever.title_field == "title"
        assert retriever.use_semantic_search is True

    @patch("src.azure_search_dspy.retriever.SearchClient")
    def test_forward(self, mock_search_client):
        """Test retrieval forward pass"""
        # Setup mock search results
        mock_result = Mock()
        mock_result.get = Mock(
            side_effect=lambda x, default=None: {
                "content": "Test content",
                "title": "Test Title",
                "@search.score": 0.9,
            }.get(x, default)
        )

        mock_client_instance = Mock()
        mock_client_instance.search = Mock(return_value=[mock_result])
        mock_search_client.return_value = mock_client_instance

        retriever = AzureSearchRM(
            search_endpoint="https://test.search.windows.net",
            search_key="test-key",
            index_name="test-index",
            k=1,
        )

        result = retriever("test query")

        assert isinstance(result, dspy.Prediction)
        assert len(result.passages) == 1
        assert "Test Title" in result.passages[0]
        assert "Test content" in result.passages[0]


class TestInformationExtractionAgent:
    """Tests for Information Extraction Agent"""

    def test_init(self):
        """Test agent initialization"""
        mock_retriever = Mock()
        agent = InformationExtractionAgent(retriever=mock_retriever)

        assert agent.retriever == mock_retriever
        assert hasattr(agent, "query_rewriter")
        assert hasattr(agent, "summarizer")
        assert hasattr(agent, "key_points_extractor")
        assert hasattr(agent, "entity_extractor")

    @patch.object(dspy.ChainOfThought, "__call__")
    def test_forward_pipeline(self, mock_cot):
        """Test full agent pipeline"""
        # Setup mock retriever
        mock_retriever = Mock()
        mock_retriever.return_value = dspy.Prediction(
            passages=["Title: Test Doc\n\nTest content"]
        )

        # Setup mock responses for each stage
        mock_cot.side_effect = [
            dspy.Prediction(rewritten_query="enhanced query"),
            dspy.Prediction(summary="Test summary"),
            dspy.Prediction(key_points="1. Point 1\n2. Point 2\n3. Point 3"),
            dspy.Prediction(
                entities_json='[{"name":"test","type":"person","value":"test","confidence":"high"}]'
            ),
        ]

        agent = InformationExtractionAgent(retriever=mock_retriever)
        result = agent("test query")

        assert result.query == "test query"
        assert result.rewritten_query == "enhanced query"
        assert result.summary == "Test summary"
        assert "Point 1" in result.key_points
        assert len(result.sources) > 0


class TestEvaluation:
    """Tests for evaluation functions"""

    def test_evaluate_quality_perfect_score(self):
        """Test quality evaluation with perfect predictions"""
        example = dspy.Example(
            query="test query", expected_topics=["topic1", "topic2"]
        ).with_inputs("query")

        prediction = dspy.Prediction(
            summary="First sentence. Second sentence. Third sentence.",
            key_points="1. Point 1\n2. Point 2\n3. Point 3\n4. Point 4",
            entities_json='[{"name":"test","type":"person","value":"test","confidence":"high"}]',
            sources=["Source 1", "Source 2"],
            query="test query",
            rewritten_query="enhanced query",
            context="Test context with topic1 and topic2",
        )

        score = evaluate_quality(example, prediction)

        # Should score high with good summary, key points, topics, entities, and sources
        assert score > 0.6
        assert score <= 1.0

    def test_evaluate_quality_low_score(self):
        """Test quality evaluation with poor predictions"""
        example = dspy.Example(
            query="test query", expected_topics=["topic1", "topic2"]
        ).with_inputs("query")

        prediction = dspy.Prediction(
            summary="Short.",  # Too short
            key_points="Only one point",  # Too few
            entities_json="invalid json",  # Invalid
            sources=[],  # No sources
            query="test query",
            rewritten_query="enhanced query",
            context="Test context",  # Missing topics
        )

        score = evaluate_quality(example, prediction)

        # Should score low
        assert score < 0.6

    def test_evaluate_agent(self):
        """Test agent evaluation on dataset"""
        mock_agent = Mock()
        mock_agent.return_value = dspy.Prediction(
            summary="Test summary. Another sentence.",
            key_points="1. Point 1\n2. Point 2\n3. Point 3",
            entities_json='[{"name":"test","type":"person","value":"test","confidence":"high"}]',
            sources=["Source 1", "Source 2"],
            query="test query",
            rewritten_query="enhanced query",
            context="Test context",
        )

        dataset = [
            dspy.Example(query="query 1", expected_topics=["topic"]).with_inputs(
                "query"
            ),
            dspy.Example(query="query 2", expected_topics=["topic"]).with_inputs(
                "query"
            ),
        ]

        results = evaluate_agent(mock_agent, dataset)

        assert "average_score" in results
        assert "scores" in results
        assert "predictions" in results
        assert "num_examples" in results
        assert results["num_examples"] == 2
        assert len(results["scores"]) == 2
        assert len(results["predictions"]) == 2


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring real Azure services"""

    @pytest.mark.skip(reason="Requires Azure credentials")
    def test_real_azure_search(self):
        """Test with real Azure Search (requires credentials)"""
        import os

        retriever = AzureSearchRM(
            search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX"),
            k=3,
        )

        result = retriever("test query")
        assert len(result.passages) <= 3

