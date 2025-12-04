"""DsPy Azure AI Search Integration for Information Extraction"""

from .retriever import AzureSearchRM
from .agent import InformationExtractionAgent
from .evaluation import evaluate_quality, evaluate_agent

__all__ = [
    "AzureSearchRM",
    "InformationExtractionAgent",
    "evaluate_quality",
    "evaluate_agent",
]

