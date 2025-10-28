# Databricks notebook source
# MAGIC %md
# MAGIC This notebook covers how to send chats to a review app  

# COMMAND ----------

# MAGIC %pip install databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks import agents

agents.enable_trace_reviews(
  model_name='shm.dbdemos_llm_rag.llama_70b_w_rag',
  request_ids=[
    "3c413350-9fab-4796-9756-3c8bbd997283"
  ],
)
