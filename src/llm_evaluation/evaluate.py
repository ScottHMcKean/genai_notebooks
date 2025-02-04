# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluating LLMs with Databricks
# MAGIC This notebook goes through the Databricks evaluation framework for language moodels. The agenda is as follows:
# MAGIC
# MAGIC - A quick review of the RAG application
# MAGIC - A whirlwind tour of the model setup and deployment
# MAGIC - An interactive tour through the review application

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 rouge_score
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from pyspark.sql import functions as F
from databricks import agents
from eval_func import _dedup_assessment_log, get_endpoint_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Endpoint and Logs
# MAGIC We are going to use a prebaked evaluation set here, but you should generate evaluation sets using the Review App.
# MAGIC
# MAGIC Review App:
# MAGIC https://adb-984752964297111.11.azuredatabricks.net/ml/review/shm.dbdemos_llm_rag.llama_70b_w_rag/1?o=984752964297111
# MAGIC
# MAGIC Endpoint: 
# MAGIC https://adb-984752964297111.11.azuredatabricks.net/ml/endpoints/agents_shm-dbdemos_llm_rag-llama_70b_w_rag

# COMMAND ----------

uc_model_name = 'shm.dbdemos_llm_rag.llama_70b_w_rag'
endpoint_config = get_endpoint_config(uc_model_name)

# COMMAND ----------

# Inference
inference_table_name = endpoint_config.state.payload_table.name
inference_table_catalog = endpoint_config.catalog_name
inference_table_schema = endpoint_config.schema_name
inference_table_uri = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}`"
print(f"Inference table: {inference_table_uri}")
inference_table_df = spark.table(inference_table_uri)

# Request
request_log_table_name = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_request_logs`"
print(f"Request logs: {request_log_table_name}")
request_log_df = spark.table(request_log_table_name)

# Assessment
assessment_log_table_name = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_assessment_logs`"
assessment_log_df = spark.table(assessment_log_table_name)
print(f"Assessment logs: {assessment_log_table_name}")

# COMMAND ----------

display(inference_table_df.limit(10))

# COMMAND ----------

from pyspark.sql.window import Window

window_spec = (
  Window
  .partitionBy("request_id")
  .orderBy(F.col("timestamp").desc())
)

result_df = (
  assessment_log_df
  .withColumn("row_number", F.row_number().over(window_spec))
  .filter(F.col("row_number") == 1)
  .drop("row_number")
  .orderBy(F.col("timestamp").desc())
)

display(result_df)

# COMMAND ----------

display(request_log_df.limit(10))

# COMMAND ----------

eval_set = spark.table(
  "shm.dbdemos_llm_rag.eval_set_databricks_documentation"
  )
display(eval_set)

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
llm = ChatDatabricks(endpoint="databricks-dbrx-instruct")
llm.invoke('test')

# COMMAND ----------

# Generate some responses
import pyspark.sql.types as T

@F.udf(returnType=T.StringType())
def llm_udf(prompt):
  from langchain_community.chat_models import ChatDatabricks
  llm = ChatDatabricks(endpoint="databricks-dbrx-instruct")
  return llm.invoke(prompt).content

test_set = (
  eval_set
  .limit(15)
  .withColumn('response', llm_udf(F.col('request')))
)

display(test_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Metrics
# MAGIC There are several statistical metrics we can use. We will start with classical categorization metrics (precision, recall, F1) and follow with ROUGE and BLEU.
# MAGIC
# MAGIC Let's start with easier evaluation problems - sentiment analysis and named entity recognition. In sentiment analysis, the problem is a classy classic multiclass classification problem (how many times can I say class?). In named entity recognition - the classes increase and the sets become unordered, so it starts to require a bit more handling.
# MAGIC
# MAGIC ### Precision
# MAGIC Precision measures the accuracy of positive predictions:
# MAGIC
# MAGIC $$Precision = \frac{TP}{TP + FP}$$
# MAGIC
# MAGIC ### Recall
# MAGIC
# MAGIC Recall measures the ability to find all positive instances:
# MAGIC
# MAGIC $$Recall = \frac{TP}{TP + FN}$$
# MAGIC
# MAGIC ### F1 Score
# MAGIC
# MAGIC F1 score is the harmonic mean of precision and recall:
# MAGIC
# MAGIC $$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

# COMMAND ----------

# Sentiment analysis is just classification
# Could also use mlflow.metrics.precision_score(), etc.
from sklearn.metrics import precision_score, recall_score, f1_score

y = [
    "Positive", "Neutral", "Positive", "Negative", "Positive",
    "Neutral", "Negative", "Positive", "Neutral", "Positive"
]

y_pred = [
    "Positive", "Neutral", "Neutral", "Negative", "Positive",
    "Positive", "Negative", "Positive", "Negative", "Neutral"
]

# Calculate precision, recall, and F1 score for each class
results = {}
for sentiment in ['Negative', 'Neutral', 'Positive']:
    precision = precision_score(y, y_pred, labels=[sentiment], average='micro')
    recall = recall_score(y, y_pred, labels=[sentiment], average='micro')
    f1 = f1_score(y, y_pred, labels=[sentiment], average='micro')
    results[sentiment] = {
        'precision': np.round(precision,2),
        'recall': np.round(recall,2),
        'f1_score': np.round(f1,2)
    }

# Calculate overall metrics
overall_precision = precision_score(y, y_pred, average='weighted')
overall_recall = recall_score(y, y_pred, average='weighted')
overall_f1 = f1_score(y, y_pred, average='weighted')

results['Overall'] = {
    'precision': np.round(overall_precision,2),
    'recall': np.round(overall_recall,2),
    'f1_score': np.round(overall_f1,2)
}

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Natural Language Processing (NLP) Metrics
# MAGIC NLP existed long before transformers. It has a wide range of accuracy metrics that are generally based on n-grams. Let's look at ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
# MAGIC
# MAGIC Other ones to check out include BLEU and METEOR.
# MAGIC
# MAGIC ![](n-gram.png)

# COMMAND ----------

results = mlflow.evaluate(
    data=test_set.toPandas(),
    targets='expected_response',
    predictions='response',
    extra_metrics=[
      mlflow.metrics.token_count(),
      mlflow.metrics.rouge1(),
      mlflow.metrics.rouge2(),
      mlflow.metrics.rougeL()
      ]
)

# COMMAND ----------

display(results.tables['eval_results_table'])

# COMMAND ----------

results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM As Judge
# MAGIC Because statistical measures can be problematic due to semantics, we often turn to LLMs to judge LLMs.

# COMMAND ----------

from pyspark.sql.types import *

@udf(StringType())
def compare_correctness_llm(actual, expected):
    from langchain import PromptTemplate
    from langchain_community.chat_models import ChatDatabricks
    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
    # Replace with your actual LLM API endpoint
    prompt_template = PromptTemplate(
    input_variables=["response", "expected_response"],
    template="""
    Evaluate the correctness of the actual response compared to the expected response.
    Provide a score from 1 to 5, where:
    1 = Completely incorrect
    2 = Mostly incorrect
    3 = Partially correct
    4 = Mostly correct
    5 = Completely correct

    Return the numeric score (1, 2, 3, 4, or 5) only.

    Examples:
    Actual response: The brown fox
    Expected response: The brown quick red fox
    Answer: 3

    Actual response: Turtle
    Expected response: Turtle
    Answer: 5

    Actual response: The Turtle
    Expected response: Turtle
    Answer: 4

    Compare the following two texts for correctness:

    Actual response: {response}
    Expected response: {expected_response}
    Answer: 
    """
    )
    prompt = prompt_template.format(response=actual, expected_response=expected)
    try:
        response = llm.invoke([{'role': 'user', 'content': prompt}]).content
    except:
        response = '1'

    return response

# COMMAND ----------

test_set = test_set.withColumn(
  "correctness_score", compare_correctness_llm(F.col("response"), F.col("expected_response"))
  )
display(test_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFLow Evaluation Framework
# MAGIC MLFlow can be used to bring these frameworks together in a reproducible way. Databricks also offers an abstract agent evaluation framework that takes care of all of the above results and saves the evaluation to a table.

# COMMAND ----------

# Get latest model
def get_latest_model(model_name):
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = None
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if not latest_version or version_int > int(latest_version.version):
            latest_version = mv
    return latest_version
  
model = get_latest_model(uc_model_name)

# COMMAND ----------

results = mlflow.evaluate(
    data=eval_set.limit(25),
    model=f'runs:/{model.run_id}/chain',
    model_type="databricks-agent"
)
