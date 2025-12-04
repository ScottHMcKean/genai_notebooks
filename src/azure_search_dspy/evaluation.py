"""Evaluation metrics for information extraction quality"""

import dspy
from typing import List, Dict, Any


def evaluate_quality(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> float:
    """
    Custom metric to evaluate information extraction quality.

    Args:
        example: DsPy Example with query and expected topics
        prediction: DsPy Prediction from the agent
        trace: Optional trace for training mode

    Returns:
        Normalized quality score between 0 and 1
    """
    score = 0.0
    max_score = 0.0

    # 1. Summary length check (should be concise, 2-3 sentences)
    max_score += 1.0
    summary_sentences = (
        prediction.summary.count(".")
        + prediction.summary.count("!")
        + prediction.summary.count("?")
    )
    if 2 <= summary_sentences <= 4:
        score += 1.0
    elif summary_sentences == 1 or summary_sentences == 5:
        score += 0.5

    # 2. Key points check (should have 3-5 points)
    max_score += 1.0
    key_points_count = prediction.key_points.count("\n") + 1
    if 3 <= key_points_count <= 5:
        score += 1.0
    elif key_points_count in [2, 6]:
        score += 0.5

    # 3. Topic coverage (check if expected topics are mentioned)
    if hasattr(example, "expected_topics"):
        max_score += 1.0
        combined_text = (prediction.summary + " " + prediction.key_points).lower()
        topics_found = sum(
            1 for topic in example.expected_topics if topic.lower() in combined_text
        )
        score += topics_found / len(example.expected_topics)

    # 4. Entity extraction check
    max_score += 1.0
    try:
        import json

        entities = json.loads(prediction.entities_json)
        if isinstance(entities, list) and len(entities) > 0:
            score += 1.0
        elif isinstance(entities, dict):
            score += 0.5
    except:
        pass

    # 5. Sources provided
    max_score += 1.0
    if len(prediction.sources) >= 2:
        score += 1.0
    elif len(prediction.sources) == 1:
        score += 0.5

    normalized_score = score / max_score if max_score > 0 else 0

    # During training, return boolean for positive signal
    if trace is not None:
        return normalized_score > 0.6

    return normalized_score


def evaluate_agent(
    agent, dataset: List[dspy.Example], name: str = "evaluation"
) -> Dict[str, Any]:
    """
    Evaluate agent on dataset and return metrics.

    Args:
        agent: DsPy agent to evaluate
        dataset: List of DsPy Examples
        name: Name for this evaluation run

    Returns:
        Dictionary with scores, predictions, and metrics
    """
    scores = []
    predictions = []

    for example in dataset:
        try:
            pred = agent(query=example.query)
            score = evaluate_quality(example, pred)
            scores.append(score)
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing query '{example.query}': {e}")
            scores.append(0.0)
            predictions.append(None)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "average_score": avg_score,
        "scores": scores,
        "predictions": predictions,
        "num_examples": len(dataset),
    }

