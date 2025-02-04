# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import time
import yaml
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
w = WorkspaceClient()

# COMMAND ----------

config = mlflow.models.ModelConfig(
    development_config="rag_chain_config.yml"
    )

# COMMAND ----------

from mlflow.models import infer_signature

signature = infer_signature(
  config.get('input_example'),
  config.get('output_example')
)

with mlflow.start_run(run_name='shm_reg_debug'):
    # tags, parameters, etc.
    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model='./modelcode.py',
        model_config='./rag_chain_config.yml',  
        artifact_path="chain", 
        input_example=config.get("input_example"),
        signature=signature,
        extra_pip_requirements=["databricks-agents"]
    )

# COMMAND ----------

print(logged_chain_info.model_uri)

# COMMAND ----------

rag_model_path = "shm.dbdemos_llm_rag.llama_70b_w_rag"
mlflow.set_registry_uri('databricks-uc')

# Register to UC
uc_registered_model_info = mlflow.register_model(
  model_uri=logged_chain_info.model_uri,
  name=rag_model_path
  )

# COMMAND ----------

new_model = mlflow.pyfunc.load_model(
  "models:/shm.dbdemos_llm_rag.llama_70b_w_rag/2"
  )

# COMMAND ----------

new_model.predict({"messages": [
  {"content": "Explain how mlflow can be used with LLMs", "role":"user"}
  ]})

# COMMAND ----------

print(rag_model_path)

# COMMAND ----------

deployment_info = agents.deploy(
  model_name=rag_model_path,
  model_version=1,
  scale_to_zero=True
  )

# COMMAND ----------

deployment_info.endpoint_url

# COMMAND ----------

instructions_to_reviewer = f"""
## Testing Instructions
This is the RAG Review App.
"""
 
# Add the user-facing instructions to the Review App
agents.set_review_instructions(rag_model_path, instructions_to_reviewer)

# COMMAND ----------


