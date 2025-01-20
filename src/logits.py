# Databricks notebook source
# MAGIC %md
# MAGIC This notebook provides an example of getting logits out of a model response.
# MAGIC
# MAGIC https://cookbook.openai.com/examples/using_logprobs

# COMMAND ----------

# MAGIC %pip install databricks-langchain langchain-core openai --upgrade
# MAGIC %restart_python

# COMMAND ----------

import openai
from openai import OpenAI
from databricks_langchain import ChatDatabricks

# COMMAND ----------

openai_llm = OpenAI(api_key=dbutils.secrets.get('shm','gpt-4o-mini'))

completion = openai_llm.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  logprobs=True,
  top_logprobs=5
)

print(completion.choices[0].message)

# COMMAND ----------

databricks_llm = ChatDatabricks(
    endpoint="shm-gpt-4o-mini",
    temperature=0,
    extra_params={
        'logprobs':True,
        'top_logprobs':5
    }
)

completion = databricks_llm.invoke(
  [
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

completion

# COMMAND ----------


