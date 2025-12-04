"""Pytest fixtures for Azure Assistant Tracing tests."""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from src.azure_assistant_tracing.config import AzureConfig, MLflowConfig, TracingConfig


@pytest.fixture
def mock_workspace_client():
    """Mock Databricks workspace client."""
    client = Mock()

    # Mock dbutils.secrets
    mock_dbutils = Mock()
    mock_dbutils.secrets.get.return_value = "mock_api_key"
    client.dbutils = mock_dbutils

    # Mock current_user.me()
    mock_user = Mock()
    mock_user.user_name = "test_user"
    mock_user.id = "test_user_id"
    client.current_user.me.return_value = mock_user

    # Mock experiments
    mock_experiment = Mock()
    mock_experiment.experiment_id = "test_experiment_id"
    client.experiments.get_by_name.return_value.experiment = mock_experiment

    return client


@pytest.fixture
def azure_config():
    """Sample Azure configuration."""
    return AzureConfig(
        endpoint="https://test.openai.azure.com/",
        api_key="test_api_key",
        api_version="2024-05-01-preview",
    )


@pytest.fixture
def mlflow_config():
    """Sample MLflow configuration."""
    return MLflowConfig(
        experiment_name="test_experiment", experiment_id="test_experiment_id"
    )


@pytest.fixture
def tracing_config(azure_config, mlflow_config):
    """Sample tracing configuration."""
    return TracingConfig(
        azure=azure_config,
        mlflow=mlflow_config,
        user_name="test_user",
        user_id="test_user_id",
    )


@pytest.fixture
def mock_assistant():
    """Mock Azure OpenAI assistant."""
    assistant = Mock()
    assistant.id = "test_assistant_id"
    return assistant


@pytest.fixture
def mock_thread():
    """Mock Azure OpenAI thread."""
    thread = Mock()
    thread.id = "test_thread_id"
    return thread


@pytest.fixture
def mock_run():
    """Mock Azure OpenAI run."""
    run = Mock()
    run.id = "test_run_id"
    run.status = "completed"
    run.model_dump.return_value = {"status": "completed", "id": "test_run_id"}
    return run


@pytest.fixture
def mock_message():
    """Mock Azure OpenAI message."""
    message = Mock()
    message.role = "assistant"
    message.content = [Mock()]
    message.content[0].text.value = "Test response"
    message.model_dump.return_value = {"role": "assistant", "content": "Test response"}
    return message


@pytest.fixture
def mock_azure_client(azure_config):
    """Mock Azure OpenAI client."""
    from src.azure_assistant_tracing.azure_client import AzureOpenAIClient

    client = AzureOpenAIClient(azure_config)
    client.client = Mock()
    return client
