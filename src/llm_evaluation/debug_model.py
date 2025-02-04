# Databricks notebook source
# MAGIC %pip install -qqqq -U pypdf==4.1.0 databricks-vectorsearch transformers==4.41.1 torch==2.3.1 tiktoken==0.7.0 langchain-text-splitters==0.2.2 langchain_community==0.2.10
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pseudocode
# MAGIC - read config
# MAGIC - setup vector search retreiver
# MAGIC - prompt
# MAGIC - model
# MAGIC - chain

# COMMAND ----------

import os
import yaml
from databricks.vector_search.client import VectorSearchClient
from langchain.pydantic_v1 import BaseModel, Field
import mlflow
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.chat_models import ChatDatabricks

# COMMAND ----------

# Flexible configuration reader
config = mlflow.models.ModelConfig(
    development_config="rag_chain_config.yml"
    )

vector_search_config = config.get('vector_search_config')
resources_config = config.get('databricks_resources')
retriever_config = config.get("retriever_config")
llm_config = config.get("llm_config")

# COMMAND ----------

# Functions
def extract_user_query_string(chat_messages_array):
    """
    Return the string contents of the most recent message from the user
    """
    return chat_messages_array[-1]["content"]

def extract_chat_history(chat_messages_array):
    """
    Return the chat history, which is is everything before the last question
    """
    return chat_messages_array[:-1]

def extract_previous_messages(chat_messages_array):
    messages = "\n"
    for msg in chat_messages_array[:-1]:
        messages += (msg["role"] + ": " + msg["content"] + "\n")
    return messages
 
def combine_all_messages_for_vector_search(chat_messages_array):
    return extract_previous_messages(chat_messages_array) + extract_user_query_string(chat_messages_array)
  
def format_context(docs):
    chunk_template = retriever_config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
        )
        for d in docs
    ]
    return "".join(chunk_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC - I was seeing inconsistent pattern for configuration retrieval so set everything under the ModelConfig().get() pattern

# COMMAND ----------

# Test model
model = ChatDatabricks(
    endpoint=resources_config.get("llm_endpoint_name"),
    extra_params=llm_config.get("llm_parameters"),
)

model.invoke("Is Databricks hard to use?")

# COMMAND ----------

# Prompt
prompt = PromptTemplate(
    template=llm_config.get("prompt_template"),
    input_variables=llm_config.get("prompt_template_variables"),
)


chain = (
  # Chat Extraction
  {
    "question": itemgetter("messages")| RunnableLambda(extract_user_query_string),
    "context": itemgetter("messages"),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages)
  }
  # Rest of Chain
  | prompt 
  | model 
  | StrOutputParser()
  )

chain.invoke({"messages": [
  {"content": "What the heck is lakehouse apps?", "role":"user"}
  ]})

# COMMAND ----------

# MAGIC %md
# MAGIC - I'd recommend cleaning up the yaml a bit - there is duplication in the retriever_config and vector_search config that will add some technical suffering
# MAGIC - I also recommend putting the k=1, or n in the `as_retriever` call, it doesn't work during `invoke`
# MAGIC - I am slightly concerned about namespaces in the get_retriever function and would prefer declaring in the main .py body

# COMMAND ----------

# Client (Endpoint)
vsc = VectorSearchClient(disable_notice=True)

# Index (Table)
vs_index = vsc.get_index(
    endpoint_name=vector_search_config['vector_search_endpoint_name'], 
    index_name=vector_search_config['vector_index_name']
    )

# LangChain Setup
vector_search_as_retriever = DatabricksVectorSearch(vs_index).as_retriever(search_kwargs={'k': 3})

# COMMAND ----------

vector_search_as_retriever.invoke("What is databricks lakehouse?")

# COMMAND ----------

# MAGIC %md
# MAGIC Now that I verified the chain and components I'll ship via the drive and a simplified .py file

# COMMAND ----------

# Test full chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(combine_all_messages_for_vector_search)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages)
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke({"messages": [
  {"content": "What the heck is lakehouse apps?", "role":"user"}
  ]})
