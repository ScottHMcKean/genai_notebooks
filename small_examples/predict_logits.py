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

# MAGIC %md
# MAGIC Here is a default use of OpenAI's interface giving log probs of the top 5 choices

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

# MAGIC %md
# MAGIC This section uses Chat Databricks - token probabilities don't pass through

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

# MAGIC %md
# MAGIC This section uses the OpenAI interface but with external model serving

# COMMAND ----------

from openai import OpenAI
import os

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "Tell me about Large Language Models"
  }
  ],
  model="shm-gpt-4o-mini",
  max_tokens=256,
  logprobs=True,
  top_logprobs=5
)

print(chat_completion.choices[0].message.content)
