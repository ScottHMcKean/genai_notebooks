from pyspark import sql
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.rag_eval.evaluation import traces

_REQUEST_ID = "request_id"
_TIMESTAMP = "timestamp"
_ROW_NUMBER = "row_number"
_SOURCE = "source"
_SOURCE_ID = "source.id"
_STEP_ID = "step_id"
_TEXT_ASSESSMENT = "text_assessment"
_RETRIEVAL_ASSESSMENT = "retrieval_assessment"

def get_endpoint_config(uc_model_name):
  w = WorkspaceClient()

  active_deployments = agents.list_deployments()
  active_deployment = next(
      (item for item in active_deployments if item.model_name == uc_model_name), None
  )

  endpoint = w.serving_endpoints.get(active_deployment.endpoint_name)

  try:
      endpoint_config = endpoint.config.auto_capture_config
  except AttributeError as e:
      endpoint_config = endpoint.pending_config.auto_capture_config

  return endpoint_config

def _dedup_by_assessment_window(
    assessment_log_df: sql.DataFrame, window: Window
) -> sql.DataFrame:
    """
    Remove duplicates from the assessment logs by taking the first row from each group, defined by the window
    :param assessment_log_df: PySpark DataFrame of the assessment logs
    :param window: PySpark window to group assessments by
    :return: PySpark DataFrame of the assessment logs with duplicates removed
    """
    return (
        assessment_log_df.withColumn(_ROW_NUMBER, F.row_number().over(window))
        .filter(F.col(_ROW_NUMBER) == 1)
        .drop(_ROW_NUMBER)
    )


def _dedup_assessment_log(assessment_log_df: sql.DataFrame) -> sql.DataFrame:
    """
    Remove duplicates from the assessment logs to get the latest assessments.
    :param assessment_log_df: PySpark DataFrame of the assessment logs
    :return: PySpark DataFrame of the deduped assessment logs
    """
    # Dedup the text assessments
    text_assessment_window = Window.partitionBy(_REQUEST_ID, _SOURCE_ID).orderBy(
        F.col(_TIMESTAMP).desc()
    )
    deduped_text_assessment_df = _dedup_by_assessment_window(
        # Filter rows with null text assessments
        assessment_log_df.filter(F.col(_TEXT_ASSESSMENT).isNotNull()),
        text_assessment_window,
    )

    # Remove duplicates from the retrieval assessments
    retrieval_assessment_window = Window.partitionBy(
        _REQUEST_ID,
        _SOURCE_ID,
        f"{_RETRIEVAL_ASSESSMENT}.position",
        f"{_RETRIEVAL_ASSESSMENT}.{_STEP_ID}",
    ).orderBy(F.col(_TIMESTAMP).desc())
    deduped_retrieval_assessment_df = _dedup_by_assessment_window(
        # Filter rows with null retrieval assessments
        assessment_log_df.filter(F.col(_RETRIEVAL_ASSESSMENT).isNotNull()),
        retrieval_assessment_window,
    )

    # Collect retrieval assessments from the same request/step/source into a single list
    nested_retrieval_assessment_df = (
        deduped_retrieval_assessment_df.groupBy(_REQUEST_ID, _SOURCE_ID, _STEP_ID).agg(
            F.any_value(_TIMESTAMP).alias(_TIMESTAMP),
            F.any_value(_SOURCE).alias(_SOURCE),
            F.collect_list(_RETRIEVAL_ASSESSMENT).alias("retrieval_assessments"),
        )
        # Drop the old retrieval assessment, source id, and text assessment columns
        .drop(_RETRIEVAL_ASSESSMENT, "id", _TEXT_ASSESSMENT)
    )

    # Join the deduplicated text assessments with the nested deduplicated retrieval assessments
    deduped_assessment_log_df = deduped_text_assessment_df.alias("a").join(
        nested_retrieval_assessment_df.alias("b"),
        (F.col(f"a.{_REQUEST_ID}") == F.col(f"b.{_REQUEST_ID}"))
        & (F.col(f"a.{_SOURCE_ID}") == F.col(f"b.{_SOURCE_ID}")),
        "full_outer",
    )

    # Coalesce columns from both DataFrames in case a request does not have either assessment
    return deduped_assessment_log_df.select(
        F.coalesce(F.col(f"a.{_REQUEST_ID}"), F.col(f"b.{_REQUEST_ID}")).alias(
            _REQUEST_ID
        ),
        F.coalesce(F.col(f"a.{_STEP_ID}"), F.col(f"b.{_STEP_ID}")).alias(_STEP_ID),
        F.coalesce(F.col(f"a.{_TIMESTAMP}"), F.col(f"b.{_TIMESTAMP}")).alias(
            _TIMESTAMP
        ),
        F.coalesce(F.col(f"a.{_SOURCE}"), F.col(f"b.{_SOURCE}")).alias(_SOURCE),
        F.col(f"a.{_TEXT_ASSESSMENT}").alias(_TEXT_ASSESSMENT),
        F.col("b.retrieval_assessments").alias(_RETRIEVAL_ASSESSMENT),
    )