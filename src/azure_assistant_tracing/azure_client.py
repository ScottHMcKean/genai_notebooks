"""Azure OpenAI client management."""

from typing import Optional

from openai import AzureOpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads.run import Run

from .config import AzureConfig


class AzureOpenAIClient:
    """Client for Azure OpenAI with assistant management."""

    def __init__(self, config: AzureConfig):
        """Initialize the Azure OpenAI client."""
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
        )

    def get_assistant(self, assistant_id: str) -> Assistant:
        """Retrieve an existing assistant."""
        return self.client.beta.assistants.retrieve(assistant_id=assistant_id)

    def create_assistant(
        self,
        model: str,
        instructions: str,
        tools: Optional[list] = None,
        tool_resources: Optional[dict] = None,
        temperature: float = 0.17,
        top_p: float = 0.1,
    ) -> Assistant:
        """Create a new assistant."""
        return self.client.beta.assistants.create(
            model=model,
            instructions=instructions,
            tools=tools or [],
            tool_resources=tool_resources or {},
            temperature=temperature,
            top_p=top_p,
        )

    def create_thread(self) -> Thread:
        """Create a new thread."""
        return self.client.beta.threads.create()

    def add_message(self, thread_id: str, role: str, content: str) -> None:
        """Add a message to a thread."""
        self.client.beta.threads.messages.create(
            thread_id=thread_id, role=role, content=content
        )

    def run_thread(self, thread_id: str, assistant_id: str) -> Run:
        """Run a thread with an assistant."""
        return self.client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )

    def get_run_status(self, thread_id: str, run_id: str) -> Run:
        """Get the current status of a run."""
        return self.client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )

    def get_thread_messages(self, thread_id: str) -> list:
        """Get all messages from a thread."""
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data
