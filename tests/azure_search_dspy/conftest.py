"""Pytest configuration and fixtures for azure_search_dspy tests"""

import pytest
import dspy
from unittest.mock import Mock


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing"""
    retriever = Mock()
    retriever.return_value = dspy.Prediction(
        passages=["Title: Test Document\n\nTest content here"]
    )
    return retriever


@pytest.fixture
def mock_lm():
    """Mock language model for testing"""
    lm = Mock()
    lm.return_value = "Test response"
    return lm


@pytest.fixture
def sample_dataset():
    """Sample evaluation dataset"""
    return [
        dspy.Example(
            query="What is Azure AI Search?",
            expected_topics=["search", "azure", "cognitive"],
        ).with_inputs("query"),
        dspy.Example(
            query="How does semantic search work?",
            expected_topics=["semantic", "ranking"],
        ).with_inputs("query"),
    ]


@pytest.fixture
def sample_prediction():
    """Sample prediction for testing"""
    return dspy.Prediction(
        query="test query",
        rewritten_query="enhanced test query",
        summary="This is a test summary. It has multiple sentences.",
        key_points="1. First point\n2. Second point\n3. Third point",
        entities_json='[{"name":"Azure","type":"organization","value":"Microsoft Azure","confidence":"high"}]',
        context="Sample context from retrieved documents",
        sources=["Document 1", "Document 2"],
    )

