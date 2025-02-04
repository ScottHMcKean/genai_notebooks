# Imports
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

# Read config
config = mlflow.models.ModelConfig(
    development_config="rag_chain_config.yml"
    )

vector_search_config = config.get('vector_search_config')
resources_config = config.get('databricks_resources')
retriever_config = config.get("retriever_config")
llm_config = config.get("llm_config")

# Functions
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
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

model = ChatDatabricks(
    endpoint=resources_config.get("llm_endpoint_name"),
    extra_params=llm_config.get("llm_parameters"),
)

prompt = PromptTemplate(
    template=llm_config.get("prompt_template"),
    input_variables=llm_config.get("prompt_template_variables"),
)

vector_search_endpoint_name = vector_search_config['vector_search_endpoint_name']
vector_index_name = vector_search_config['vector_index_name']
vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(
    endpoint_name=vector_search_endpoint_name, 
    index_name=vector_index_name
    )
vector_search_as_retriever = DatabricksVectorSearch(vs_index).as_retriever(search_kwargs={'k': 3})

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

mlflow.models.set_model(chain)