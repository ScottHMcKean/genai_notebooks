# Databricks notebook source
# MAGIC %md
# MAGIC ## LLMS & User Defined Functions
# MAGIC
# MAGIC This notebook provides some simple examples of calling LLMs via user defined functions (UDFs). There are a couple key considerations here:
# MAGIC
# MAGIC 1. Parallelizing calls to LLM endpoints can quickly overwhelm token limits, so it is best to control parallelism by both choking back your cluster size (limiting the number of workers) and repartitioning the dataframe.
# MAGIC 2. UDFs need to be self contained. This is especially important when declaring API connections and models.
# MAGIC 3. You need to cache or write UDF results, or else Spark will rerun the UDF call every time - this can be expensive so be careful! 

# COMMAND ----------

# MAGIC %md
# MAGIC This is a canonical example - we do imports and establish the ChatDatabricks object within the UDF and then use it over a Spark Dataframe

# COMMAND ----------

import pyspark.sql.types as T

@F.udf(returnType=T.StringType())
def llm_udf(prompt):
  from databricks_langchain import ChatDatabricks
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
# MAGIC Here is an embedding example

# COMMAND ----------

@F.udf(T.ArrayType(T.FloatType()))
def compute_embeddings_udf(description):
  from databricks_langchain import DatabricksEmbeddings
  embeddings = DatabricksEmbeddings(endpoint=config.get("embedding_endpoint"))
  return embeddings.embed_query(description)

# COMMAND ----------

# MAGIC %md
# MAGIC Here is an example of a UDF to do a custom evaluation of correctness. This can be done more easily in MLFLow Evaluate or using the Agent Evaluation framework, but the idea is the same.

# COMMAND ----------

from pyspark.sql.types import *

@udf(StringType())
def compare_correctness_llm(actual, expected):
    from langchain import PromptTemplate
    from databricks_langchain import ChatDatabricks
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
