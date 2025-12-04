"""DsPy Agent for Information Extraction and Summarization"""

import dspy
from .signatures import QueryRewrite, GenerateSummary, ExtractKeyPoints, ExtractEntities


class InformationExtractionAgent(dspy.Module):
    """Complete RAG agent for information extraction and summarization"""

    def __init__(self, retriever):
        """
        Initialize the information extraction agent.

        Args:
            retriever: DsPy retrieval module (e.g., AzureSearchRM)
        """
        super().__init__()
        self.retriever = retriever
        self.query_rewriter = dspy.ChainOfThought(QueryRewrite)
        self.summarizer = dspy.ChainOfThought(GenerateSummary)
        self.key_points_extractor = dspy.ChainOfThought(ExtractKeyPoints)
        self.entity_extractor = dspy.ChainOfThought(ExtractEntities)

    def forward(self, query: str) -> dspy.Prediction:
        """
        Process a query through the full extraction pipeline.

        Args:
            query: User's information request

        Returns:
            DsPy Prediction with summary, key_points, entities, and sources
        """
        # Step 1: Rewrite query for better retrieval
        rewritten = self.query_rewriter(query=query)

        # Step 2: Retrieve relevant documents
        retrieved = self.retriever(rewritten.rewritten_query)
        context = "\n\n---\n\n".join(retrieved.passages)

        # Step 3: Generate summary
        summary_result = self.summarizer(query=query, context=context)

        # Step 4: Extract key points
        key_points_result = self.key_points_extractor(query=query, context=context)

        # Step 5: Extract entities
        entities_result = self.entity_extractor(context=context)

        # Extract source titles from passages
        sources = []
        for p in retrieved.passages[:3]:
            if "Title: " in p:
                title = p.split("Title: ")[1].split("\n")[0]
                sources.append(title)
            else:
                sources.append("Unknown")

        return dspy.Prediction(
            query=query,
            rewritten_query=rewritten.rewritten_query,
            summary=summary_result.summary,
            key_points=key_points_result.key_points,
            entities_json=entities_result.entities_json,
            context=context,
            sources=sources,
        )

