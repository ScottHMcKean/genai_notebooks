"""MLflow tracing setup and management."""

import mlflow
from mlflow.tracing.destination import Databricks

from .config import MLflowConfig


def setup_mlflow_tracing(config: MLflowConfig) -> None:
    """Setup MLflow tracing with Databricks destination."""
    mlflow.set_tracking_uri("databricks")
    mlflow.tracing.set_destination(Databricks(experiment_id=config.experiment_id))


def update_trace_metadata(
    client_request_id: str, user_name: str, session_id: str
) -> None:
    """Update the current trace with metadata."""
    mlflow.update_current_trace(
        client_request_id=client_request_id,
        metadata={
            "mlflow.trace.user": user_name,
            "mlflow.trace.session": session_id,
        },
    )


def update_trace_state(state: str = "OK") -> None:
    """Update the current trace state."""
    mlflow.update_current_trace(state=state)


def search_user_traces(
    experiment_id: str, user_name: str, max_results: int = 10
) -> mlflow.tracking.entities.TraceSearchResult:
    """Search for traces by user."""
    return mlflow.search_traces(
        experiment_ids=[experiment_id],
        order_by=["timestamp DESC"],
        filter_string=f"metadata.mlflow.trace.user = '{user_name}'",
        max_results=max_results,
    )


def get_trace_by_thread_id(
    experiment_id: str, user_name: str, max_results: int = 1
) -> str:
    """Get the most recent trace ID for a user."""
    traces = search_user_traces(experiment_id, user_name, max_results)
    if traces.empty:
        raise ValueError(f"No traces found for user {user_name}")
    return traces.iloc[0].trace_id
