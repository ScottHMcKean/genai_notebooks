import mlflow
import uuid
from typing import Dict, List, Optional, Generator
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

class EDCAgentModel(mlflow.pyfunc.ResponsesAgent):
    """
    MLflow ResponsesAgent wrapper for EDC AI Summarization Agent.
    Uses the same prompt template as simple_knowledge_extraction notebook.
    """
    
    # EDC Summarization Prompt Template (exact same as original notebook)
    PROMPT_TEMPLATE = """[SUMMARIZATION RULES]

[SYSTEM]
You are an AI summarizer for Export Development Canada (EDC). Answer the query using ONLY the provided search results.
If insufficient info, reply: 'INSUFFICIENT_INFORMATION'.

WRITING STYLE (EDC):
- Plain language, short sentences (15–20 words max).
- Inverted pyramid: begin with the most important point.
- Professional but friendly tone.
- Headings: Capitalize first word only (unless proper names/titles).
- Write in second person ("you" not "companies").
- EDC products: Capitalize all EDC products (start with "EDC").
- Canadian spelling (e.g., "favour", "centre").
- No jargon, idioms, or unnecessary adjectives.
- Use consistent formatting: headings (##, ###), bullets, bold for emphasis.

CONTENT FOCUS:
- Emphasize Canadian exporters and EDC services: financing, credit insurance, bonding, market intelligence, Trading.
- Synthesize information across results.
- Maximum length: {max_words} words.

GUARDRAILS:
- Do not invent or assume facts.
- Use only the provided search results.
- Every factual statement must include an inline numbered citation [1], [2], [3].
- No URLs. Do not add a citation list at the end.

[CONTEXT]
USER QUERY: {query}
SEARCH RESULTS: {contents}

[INSTRUCTIONS]
1. Start with a direct answer under a level-2 heading (##).
2. Add a Key Points section for supporting details or recommendations.
3. Use bullets for clarity and bold for emphasis.
4. Prioritize higher-scoring results first.
5. Keep sentences short, tone neutral, and spelling Canadian

[RESPONSE FORMAT]
## [Main Answer]
[Concise synthesis with inline citations like [1]]

### Key points
- [Insight with citation like [2]]
- [Recommendation with citations like [1], [3]]
"""
    
    def __init__(self):
        """Initialize the agent with Azure AI Search and Databricks clients."""
        super().__init__()
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        import os
        
        # Load configuration
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        index = os.getenv("AZURE_SEARCH_INDEX")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.model_endpoint = os.getenv("DATABRICKS_MODEL_ENDPOINT")
        
        # Initialize search client
        credential = AzureKeyCredential(api_key)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index,
            credential=credential
        )
        
        # Initialize Databricks client if model endpoint is configured
        if self.model_endpoint:
            from databricks.sdk import WorkspaceClient
            self.workspace_client = WorkspaceClient()
        
        print("✅ EDC Agent initialized successfully")
    
    def load_context(self, context):
        """Load context (called by MLflow when loading logged model)."""
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        import os
        
        # Load configuration
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        index = os.getenv("AZURE_SEARCH_INDEX")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.model_endpoint = os.getenv("DATABRICKS_MODEL_ENDPOINT")
        
        # Initialize search client
        credential = AzureKeyCredential(api_key)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index,
            credential=credential
        )
        
        # Initialize Databricks client if model endpoint is configured
        if self.model_endpoint:
            from databricks.sdk import WorkspaceClient
            self.workspace_client = WorkspaceClient()
        
        print("✅ EDC Agent initialized successfully")
    
    def _extract_knowledge(self, query: str, top: int = 6) -> Dict:
        """Extract knowledge from Azure AI Search with keyword search."""
        from azure.search.documents.models import QueryType
        
        # Build search parameters for keyword search
        search_params = {
            'search_text': query,
            'top': top,
            'include_total_count': True,
            'select': ['id', 'content'],
            'query_type': QueryType.FULL,
        }
        
        # Execute search
        results = self.search_client.search(**search_params)
        
        # Extract documents with all available information
        documents = []
        raw_results = []
        for result in results:
            # Store the complete result for reference
            raw_result = dict(result)
            raw_results.append(raw_result)
            
            # Extract key fields for processing
            doc = {
                'id': result.get('id', ''),
                'content': result.get('content', ''),
            }
            documents.append(doc)
        
        return {
            'query': query,
            'total_results': results.get_count(),
            'documents': documents,
            'raw_results': raw_results,
            'ranking_mode': 'keyword search'
        }
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results for the prompt with full content."""
        formatted = []
        
        for i, doc in enumerate(search_results, 1):
            id = doc.get('id', '')
            content = doc.get('content', '')
            
            formatted.append(
                f"[{i}] Id: {id}\n"
                f"Content: {content}\n"
            )
        
        return "\n---\n".join(formatted)
    
    def _generate_summary(self, query: str, search_results: List[Dict], max_words: int) -> str:
        """Generate summary using the EDC prompt template."""
        
        if not search_results:
            return "INSUFFICIENT_INFORMATION"
        
        # Format search results
        formatted_contents = self._format_search_results(search_results)
        
        # Build the prompt using the EDC template
        prompt = self.PROMPT_TEMPLATE.format(
            max_words=max_words,
            query=query,
            contents=formatted_contents
        )
        
        # Generate summary
        if self.model_endpoint:
            return self._call_databricks_model(prompt)
        else:
            return "NO_LLM_CONFIGURED: Please set DATABRICKS_MODEL_ENDPOINT to use LLM-based summarization"
    
    def _call_databricks_model(self, prompt: str) -> str:
        """Call Databricks model endpoint for summarization."""
        try:
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
            
            response = self.workspace_client.serving_endpoints.query(
                name=self.model_endpoint,
                messages=[
                    ChatMessage(
                        role=ChatMessageRole.USER,
                        content=prompt
                    )
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            if response and response.choices:
                return response.choices[0].message.content
            else:
                return "Error: No response from model"
                
        except Exception as e:
            return f"Error calling model: {str(e)}"
    
    def predict(self, context=None, request: ResponsesAgentRequest=None) -> ResponsesAgentResponse:
        """Handle agent queries using ResponsesAgent interface."""
        try:
            # Handle different calling conventions
            if request is None and context is not None:
                if isinstance(context, dict):
                    request = context
                    context = None
            
            if request is None:
                raise ValueError("No request data provided")
            
            # Extract the user message
            messages = request.get('input', [])
            if not messages:
                raise ValueError("No messages provided in request")
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg
                    break
            
            if not user_message:
                raise ValueError("No user message found in request")
            
            # Extract query from message content
            content = user_message.get('content', '')
            if isinstance(content, list):
                query = ' '.join([c.get('text', '') for c in content if c.get('type') == 'text'])
            else:
                query = content
            
            # Extract optional parameters
            top_k = 6
            max_words = 800
            
            # Extract knowledge from Azure Search
            search_result = self._extract_knowledge(query=query, top=top_k)
            
            # Generate summary using EDC prompt
            summary = self._generate_summary(
                query=query,
                search_results=search_result['documents'],
                max_words=max_words
            )
            
            # Create output with references in custom_outputs
            return ResponsesAgentResponse(
                output=[
                    {
                        "type": "message",
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": summary
                            }
                        ],
                        "status": "completed"
                    }
                ],
                custom_outputs={
                    "references": search_result['raw_results'],
                    "total_results": search_result['total_results'],
                    "ranking_mode": search_result['ranking_mode'],
                    "query": query
                }
            )
            
        except Exception as e:
            return ResponsesAgentResponse(
                output=[
                    {
                        "type": "message",
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"Error: {str(e)}"
                            }
                        ],
                        "status": "completed"
                    }
                ],
                custom_outputs={"error": str(e)}
            )
    
    def predict_stream(
        self, 
        context=None, 
        request: ResponsesAgentRequest = None
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Handle streaming agent queries using ResponsesAgent interface."""
        try:
            # Handle different calling conventions
            if request is None and context is not None:
                if isinstance(context, dict):
                    request = context
                    context = None
            
            if request is None:
                raise ValueError("No request data provided")
            
            # Extract the user message
            messages = request.get('input', [])
            if not messages:
                raise ValueError("No messages provided in request")
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg
                    break
            
            if not user_message:
                raise ValueError("No user message found in request")
            
            # Extract query from message content
            content = user_message.get('content', '')
            if isinstance(content, list):
                query = ' '.join([c.get('text', '') for c in content if c.get('type') == 'text'])
            else:
                query = content
            
            # Extract optional parameters
            top_k = 6
            max_words = 800
            
            # Extract knowledge from Azure Search
            search_result = self._extract_knowledge(query=query, top=top_k)
            
            # Generate summary
            summary = self._generate_summary(
                query=query,
                search_results=search_result['documents'],
                max_words=max_words
            )
            
            # Stream the response in chunks
            item_id = str(uuid.uuid4())
            chunk_size = 50
            
            # Manual streaming format
            for i in range(0, len(summary), chunk_size):
                chunk = summary[i:i + chunk_size]
                yield ResponsesAgentStreamEvent(
                    event="response.output_text.delta",
                    type="message",
                    delta={"text": chunk},
                    item_id=item_id
                )
            
            # Send final done event with complete text and references
            yield ResponsesAgentStreamEvent(
                event="response.output_item.done",
                type="message",
                item={
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": summary
                        }
                    ],
                    "status": "completed"
                },
                custom_outputs={
                    "references": search_result['raw_results'],
                    "total_results": search_result['total_results'],
                    "ranking_mode": search_result['ranking_mode'],
                    "query": query
                }
            )
            
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_msg = f"Error: {str(e)}"
            
            yield ResponsesAgentStreamEvent(
                event="response.output_text.delta",
                type="message",
                delta={"text": error_msg},
                item_id=error_id
            )
            
            yield ResponsesAgentStreamEvent(
                event="response.output_item.done",
                type="message",
                item={
                    "type": "message",
                    "id": error_id,
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": error_msg}
                    ],
                    "status": "completed"
                },
                custom_outputs={"error": str(e)}
            )

mlflow.openai.autolog()
AGENT = EDCAgentModel()
mlflow.models.set_model(AGENT)
