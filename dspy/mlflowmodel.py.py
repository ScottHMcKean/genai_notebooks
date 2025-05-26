# Databricks notebook source
# MAGIC %pip install -qqqq -U pypdf==4.1.0 databricks-vectorsearch transformers==4.41.1 torch==2.3.1 tiktoken==0.7.0 langchain-text-splitters==0.2.2 langchain_community==0.2.10
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Read config
import os
import yaml

# Read yaml confg file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

tables_config = config['tables_config']
data_pipeline_config = config['pipeline_config']
embedding_config = config['embedding_config']
vector_search_config = config['vector_search_config']
genai_config = config['genai_config']

poc_location_base = tables_config['poc_location_base']
# Print the value
print(poc_location_base)

# COMMAND ----------

# DBTITLE 1,Set variables from config
import os

rag_poc_catalog = tables_config['rag_poc_catalog']
rag_poc_schema = tables_config['rag_poc_schema']
volume_name  = tables_config['volume_name']
raw_delta_name  = tables_config['raw_delta_name']
parsed_delta_name  = tables_config['parsed_delta_name']
chunked_docs_delta_name   = tables_config['delta_name']
source_volume_path = "/Volumes/"+rag_poc_catalog+"/"+rag_poc_schema+"/"+volume_name

raw_files_table_name=f"{rag_poc_catalog}.{rag_poc_schema}.{raw_delta_name}" 
parsed_docs_table_name = f"{rag_poc_catalog}.{rag_poc_schema}.{parsed_delta_name}"
chunked_docs_table_name= f"{rag_poc_catalog}.{rag_poc_schema}.{chunked_docs_delta_name}"

volume_location = poc_location_base+"/"+volume_name
raw_delta_location = poc_location_base+"/"+raw_delta_name
parsed_delta_location = poc_location_base+"/"+parsed_delta_name
chunked_delta_location = poc_location_base+"/"+chunked_docs_delta_name

vector_index_name = f"{rag_poc_catalog}.{rag_poc_schema}.{vector_search_config['vector_index_name']}"
vector_search_endpoint_name  = vector_search_config['vector_search_endpoint_name']
pipeline_type = vector_search_config['pipeline_type']
embedding_endpoint_name = embedding_config['embedding_endpoint_name']
embedding_service_endpoint =  embedding_config['embedding_service_endpoint']

model_name  = genai_config['model_name']

print (f"Volume location '{volume_location}' ")
print (f"Raw delta table location '{raw_delta_location}' ")
print (f"Parsed delta table location '{parsed_delta_location}' ")
print (f"Chunked delta table location '{chunked_delta_location}' ")
print (f"model name '{model_name}' ")
print (f"Source volume path '{source_volume_path}' ")
print (f"vector_index_name '{vector_index_name}' ")


# COMMAND ----------

# DBTITLE 1,Test vector search
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

import os

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    #vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=databricks_pat)
    vsc = VectorSearchClient(disable_notice=True)
    #vsc = VectorSearchClient(workspace_url=databricks_host, service_principal_client_id=client_id, service_principal_client_secret=client_secret, azure_tenant_id=tenant_id, azure_login_id=login_id)
#    vector_index_name = "rag_poc_vector_search_index"
    vs_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vector_index_name)

    # Create the retriever
    return DatabricksVectorSearch(vs_index).as_retriever()

# test our retriever
# vectorstore = get_retriever()
# similar_documents = vectorstore.get_relevant_documents("Responsibilities Director HR?")
# print(f"Relevant documents: {similar_documents[0]}")



# COMMAND ----------

# DBTITLE 1,test LLM endpoint
from langchain_community.chat_models import ChatDatabricks
from langchain.pydantic_v1 import BaseModel, Field
chat_model = ChatDatabricks(endpoint=model_name, max_tokens = 200)



# COMMAND ----------

import mlflow
model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yml")

# databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# COMMAND ----------

############
# Helper functions
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = retriever_config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata[vector_search_schema.get("document_uri")],
        )
        for d in docs
    ]
    return "".join(chunk_contents)

# COMMAND ----------

# DBTITLE 1,Test single request chain
import mlflow
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

retriever=get_retriever()

template = """
Assistant helps the company employees with their questions on company policies, roles. 
Always include the source metadata for each fact you use in the response. Use square brakets to reference the source, e.g. [role_library_pdf-10]. 
Properly format the output for human readability with new lines.
Answer the question based only on the following context:
{context}

Question: {question}
"""
#prompt = ChatPromptTemplate.from_template(template)
model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yml")

# databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

############
# Prompt Template for generation
############
prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            llm_config.get("llm_system_prompt_template"),
        ),
        # User's most current question
        ("user", "{question}"),
    ]
)

#    {"context": retriever   , "question": RunnablePassthrough()}

chain = (
    {
        "context": retriever   ,
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
    }
    | RunnablePassthrough()
    | {
        "context": itemgetter("question")
        | retriever
        | RunnableLambda(format_context),
        "question": itemgetter("question"),
    }        
    | prompt
    | chat_model
    | StrOutputParser()
)



# COMMAND ----------

# result = chain.invoke("Responsibilities of Director of Operations?")
# print (result)

# COMMAND ----------

import mlflow
mlflow.models.set_model(chain)
