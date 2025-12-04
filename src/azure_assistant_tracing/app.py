"""Main application interface for Azure Assistant Tracing."""

from typing import Dict, Any, Optional

from databricks.sdk import WorkspaceClient

from .config import TracingConfig
from .azure_client import AzureOpenAIClient
from .mlflow_tracing import setup_mlflow_tracing, get_trace_by_thread_id
from .tracing_service import AzureAssistantTracingService
from .feedback import (
    log_user_satisfaction,
    log_relevance_score,
    log_llm_evaluation,
    log_customer_service_expectations,
)


class AzureAssistantTracingApp:
    """Main application class for Azure Assistant tracing."""

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        azure_endpoint: str,
        experiment_name: str,
        assistant_id: Optional[str] = None,
    ):
        """Initialize the application."""
        self.config = TracingConfig.from_workspace_client(
            workspace_client, azure_endpoint, experiment_name
        )

        # Setup MLflow tracing
        setup_mlflow_tracing(self.config.mlflow)

        # Initialize services
        self.azure_client = AzureOpenAIClient(self.config.azure)
        self.tracing_service = AzureAssistantTracingService(self.config)

        # Store assistant if provided
        self.assistant = None
        if assistant_id:
            self.assistant = self.azure_client.get_assistant(assistant_id)

    def set_assistant(self, assistant_id: str) -> None:
        """Set the assistant to use for queries."""
        self.assistant = self.azure_client.get_assistant(assistant_id)

    def create_assistant(
        self,
        model: str,
        instructions: str,
        tools: Optional[list] = None,
        tool_resources: Optional[dict] = None,
        temperature: float = 0.17,
        top_p: float = 0.1,
    ) -> None:
        """Create a new assistant and set it as the current one."""
        self.assistant = self.azure_client.create_assistant(
            model=model,
            instructions=instructions,
            tools=tools,
            tool_resources=tool_resources,
            temperature=temperature,
            top_p=top_p,
        )

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question to the current assistant."""
        if not self.assistant:
            raise ValueError(
                "No assistant set. Use set_assistant() or create_assistant() first."
            )

        return self.tracing_service.run_assistant_query(self.assistant, question)

    def log_feedback(
        self,
        question: str,
        satisfied: bool,
        relevance_score: Optional[float] = None,
        rationale: Optional[str] = None,
    ) -> None:
        """Log feedback for the most recent trace."""
        try:
            trace_id = get_trace_by_thread_id(
                self.config.mlflow.experiment_id, self.config.user_name
            )

            # Log user satisfaction
            log_user_satisfaction(
                trace_id=trace_id,
                user_name=self.config.user_name,
                satisfied=satisfied,
                rationale=rationale,
            )

            # Log relevance score if provided
            if relevance_score is not None:
                log_relevance_score(
                    trace_id=trace_id,
                    user_name=self.config.user_name,
                    score=relevance_score,
                    rationale=rationale,
                )

        except ValueError as e:
            print(f"Could not log feedback: {e}")

    def log_expectations(
        self,
        should_escalate: bool = True,
        required_elements: Optional[list] = None,
        max_response_length: int = 150,
        tone: str = "professional_friendly",
    ) -> None:
        """Log expectations for the most recent trace."""
        try:
            trace_id = get_trace_by_thread_id(
                self.config.mlflow.experiment_id, self.config.user_name
            )

            log_customer_service_expectations(
                trace_id=trace_id,
                user_name=self.config.user_name,
                should_escalate=should_escalate,
                required_elements=required_elements,
                max_response_length=max_response_length,
                tone=tone,
            )

        except ValueError as e:
            print(f"Could not log expectations: {e}")

    def get_user_traces(self, max_results: int = 10) -> Any:
        """Get recent traces for the current user."""
        from .mlflow_tracing import search_user_traces

        return search_user_traces(
            self.config.mlflow.experiment_id, self.config.user_name, max_results
        )
