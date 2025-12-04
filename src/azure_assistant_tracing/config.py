"""Configuration management for Azure Assistant Tracing."""

import os
from dataclasses import dataclass
from typing import Optional

from databricks.sdk import WorkspaceClient


@dataclass
class AzureConfig:
    """Configuration for Azure OpenAI."""

    endpoint: str
    api_key: str
    api_version: str = "2024-05-01-preview"

    @classmethod
    def from_workspace_client(
        cls, workspace_client: WorkspaceClient, endpoint: str
    ) -> "AzureConfig":
        """Create config from Databricks workspace client."""
        api_key = workspace_client.dbutils.secrets.get(
            scope="shm", key="azure_agent_key"
        )
        return cls(endpoint=endpoint, api_key=api_key)

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Create config from environment variables."""
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
        )


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracing."""

    experiment_name: str
    experiment_id: Optional[str] = None

    @classmethod
    def from_experiment_name(
        cls, workspace_client: WorkspaceClient, experiment_name: str
    ) -> "MLflowConfig":
        """Create config from experiment name."""
        experiment = workspace_client.experiments.get_by_name(experiment_name)
        return cls(
            experiment_name=experiment_name, experiment_id=experiment.experiment_id
        )

    @classmethod
    def from_experiment_id(cls, experiment_id: str) -> "MLflowConfig":
        """Create config from experiment ID."""
        return cls(experiment_name="", experiment_id=experiment_id)


@dataclass
class TracingConfig:
    """Complete configuration for tracing."""

    azure: AzureConfig
    mlflow: MLflowConfig
    user_name: str
    user_id: str

    @classmethod
    def from_workspace_client(
        cls,
        workspace_client: WorkspaceClient,
        azure_endpoint: str,
        experiment_name: str,
    ) -> "TracingConfig":
        """Create complete config from workspace client."""
        azure_config = AzureConfig.from_workspace_client(
            workspace_client, azure_endpoint
        )
        mlflow_config = MLflowConfig.from_experiment_name(
            workspace_client, experiment_name
        )

        user_info = workspace_client.current_user.me()

        return cls(
            azure=azure_config,
            mlflow=mlflow_config,
            user_name=user_info.user_name,
            user_id=str(user_info.id),
        )
