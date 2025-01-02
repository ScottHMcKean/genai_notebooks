# Databricks notebook source
# MAGIC %md
# MAGIC ## Driver Notebook
# MAGIC This notebook registers our custom chain / agent

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-langchain langchain-community langchain mlflow pydantic databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksTable,
)

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow
from mlflow.models import infer_signature

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is DB SQL?"
        }
    ]
}

output_example = "DB SQL is a SQL dialect for Databricks SQL."

signature = infer_signature(
  input_example,
  output_example
)

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'sequence_chain.py',
        ),
        pip_requirements=[
            "langchain==0.3.13",
            "langchain-community==0.3.13",
            "pydantic==2.10.4",
            "databricks-langchain==0.1.1", # used for the retriever tool
        ],
        model_config="config.yml",
        artifact_path='agent',
        signature=signature,
        resources=[
            DatabricksVectorSearchIndex(index_name="shm.dbdemos_llm_rag.databricks_documentation_shared_index"),
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct")
        ]
    )

# COMMAND ----------

# Register the model to unity
mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "shm"
schema = "dbdemos_llm_rag"
model_name = "sequential_chain"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# Deploy the agent
from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(
  UC_MODEL_NAME, 
  uc_registered_model_info.version, 
  tags = {"endpointSource": "playground"},
  version = 9
  )

# COMMAND ----------


