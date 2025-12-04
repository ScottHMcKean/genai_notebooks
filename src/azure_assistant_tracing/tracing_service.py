"""Main tracing service for Azure Assistant interactions."""

import time
from typing import Dict, List, Any

import mlflow
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads.run import Run

from .azure_client import AzureOpenAIClient
from .config import TracingConfig
from .mlflow_tracing import update_trace_metadata, update_trace_state


class AzureAssistantTracingService:
    """Service for running Azure Assistant queries with MLflow tracing."""

    def __init__(self, config: TracingConfig):
        """Initialize the tracing service."""
        self.config = config
        self.azure_client = AzureOpenAIClient(config.azure)

    def run_assistant_query(
        self, assistant: Assistant, question: str
    ) -> Dict[str, Any]:
        """Run an assistant query with full tracing."""
        with mlflow.start_span(name="azure_assistant_query", span_type="LLM") as span:
            # Create thread and set inputs
            thread = self.azure_client.create_thread()

            span.set_inputs(
                {
                    "messages": [{"role": "user", "content": question}],
                    "user_id": self.config.user_id,
                    "assistant_id": assistant.id,
                    "thread_id": thread.id,
                }
            )

            # Update trace metadata
            update_trace_metadata(
                client_request_id=thread.id,
                user_name=self.config.user_name,
                session_id=thread.id,
            )

            # Add user question to thread
            self.azure_client.add_message(thread.id, "user", question)

            # Run the thread
            run = self.azure_client.run_thread(thread.id, assistant.id)

            # Poll until completion
            run = self._wait_for_completion(thread.id, run.id)

            # Update trace state and set attributes
            update_trace_state("OK")
            span.set_attributes(run.model_dump())

            # Extract messages and set outputs
            messages = self.azure_client.get_thread_messages(thread.id)
            output_messages = self._format_messages(messages)
            all_outputs = self._extract_assistant_outputs(messages)

            span.set_outputs({"messages": output_messages, "output": all_outputs})

            return {
                "messages": output_messages,
                "output": all_outputs,
                "attributes": run.model_dump(),
            }

    def _wait_for_completion(self, thread_id: str, run_id: str) -> Run:
        """Wait for a run to complete."""
        while True:
            run = self.azure_client.get_run_status(thread_id, run_id)
            if run.status not in ["queued", "in_progress", "cancelling"]:
                break
            time.sleep(1)
        return run

    def _format_messages(self, messages: List) -> List[Dict[str, str]]:
        """Format messages for output."""
        return [
            {"role": x.role, "content": x.content[0].text.value if x.content else ""}
            for x in messages
        ]

    def _extract_assistant_outputs(self, messages: List) -> List[Dict]:
        """Extract assistant message outputs."""
        return [x.model_dump() for x in messages if x.role == "assistant"]
