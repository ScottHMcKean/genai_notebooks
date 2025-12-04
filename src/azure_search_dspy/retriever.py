"""Azure AI Search Retrieval Module for DsPy"""

import dspy
from typing import Optional
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


class AzureSearchRM(dspy.Retrieve):
    """Azure AI Search Retrieval Module for DsPy"""

    def __init__(
        self,
        search_endpoint: str,
        search_key: str,
        index_name: str,
        k: int = 5,
        content_field: str = "content",
        title_field: str = "title",
        use_semantic_search: bool = True,
    ):
        """
        Initialize Azure Search retrieval module.

        Args:
            search_endpoint: Azure Search service endpoint URL
            search_key: Azure Search API key
            index_name: Name of the search index
            k: Number of results to retrieve
            content_field: Name of the content field in the index
            title_field: Name of the title field in the index
            use_semantic_search: Whether to use semantic search capabilities
        """
        super().__init__(k=k)
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key),
        )
        self.content_field = content_field
        self.title_field = title_field
        self.use_semantic_search = use_semantic_search

    def forward(self, query: str, k: Optional[int] = None) -> dspy.Prediction:
        """
        Search Azure AI Search and return results.

        Args:
            query: Search query string
            k: Number of results to retrieve (overrides default)

        Returns:
            DsPy Prediction with passages list
        """
        k = k or self.k

        # Configure search parameters
        search_params = {"search_text": query, "top": k, "include_total_count": True}

        # Use semantic search if available
        if self.use_semantic_search:
            search_params["query_type"] = "semantic"
            search_params["semantic_configuration_name"] = "default"

        # Execute search
        results = self.search_client.search(**search_params)

        # Format results for DsPy
        passages = []
        for result in results:
            content = result.get(self.content_field, "")
            title = result.get(self.title_field, "")
            score = result.get("@search.score", 0)

            # Combine title and content
            passage = f"Title: {title}\n\n{content}"
            passages.append(passage)

        return dspy.Prediction(passages=passages)

