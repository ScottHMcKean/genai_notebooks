"""Feedback and expectations logging for MLflow traces."""

from typing import Any, Dict, Optional

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType


def log_user_satisfaction(
    trace_id: str, user_name: str, satisfied: bool, rationale: Optional[str] = None
) -> None:
    """Log user satisfaction feedback (thumbs up/down)."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_satisfaction",
        value=satisfied,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id=user_name
        ),
        rationale=rationale,
    )


def log_relevance_score(
    trace_id: str, user_name: str, score: float, rationale: Optional[str] = None
) -> None:
    """Log relevance score feedback."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("Score must be between 0.0 and 1.0")

    mlflow.log_feedback(
        trace_id=trace_id,
        name="relevance",
        value=score,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id=user_name
        ),
        rationale=rationale,
    )


def log_llm_evaluation(
    trace_id: str, llm_source_id: str, score: float, rationale: Optional[str] = None
) -> None:
    """Log LLM-based evaluation feedback."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("Score must be between 0.0 and 1.0")

    mlflow.log_feedback(
        trace_id=trace_id,
        name="llm_evaluation",
        value=score,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id=llm_source_id
        ),
        rationale=rationale,
    )


def log_expectation(
    trace_id: str,
    user_name: str,
    expectation_name: str,
    expectation_value: Any,
    rationale: Optional[str] = None,
) -> None:
    """Log an expectation for a trace."""
    mlflow.log_expectation(
        trace_id=trace_id,
        name=expectation_name,
        value=expectation_value,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id=user_name
        ),
        rationale=rationale,
    )


def log_customer_service_expectations(
    trace_id: str,
    user_name: str,
    should_escalate: bool = True,
    required_elements: Optional[list] = None,
    max_response_length: int = 150,
    tone: str = "professional_friendly",
) -> None:
    """Log customer service specific expectations."""
    expectation_value = {
        "should_escalate": should_escalate,
        "required_elements": required_elements
        or ["empathy", "solution_offer", "follow_up"],
        "max_response_length": max_response_length,
        "tone": tone,
    }

    log_expectation(
        trace_id=trace_id,
        user_name=user_name,
        expectation_name="expected_behavior",
        expectation_value=expectation_value,
    )
