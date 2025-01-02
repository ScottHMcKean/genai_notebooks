# Databricks notebook source
# MAGIC %md
# MAGIC We are going to upgrade to the Langchain databricks package, which is quite a bit newer and should take care of authentication issues (and is better supported!)

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-langchain langchain-community mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Define the tokens, defaults, and host and set environment variables. These can be seriealized and should be safe. There was a portion of the code that tried resetting the Databricks Token which could be problematic for serving!!!

# COMMAND ----------

import mlflow
import os
# This is very handy!
from mlflow.models import ModelConfig

# COMMAND ----------

# worried about this, avoid at all costs
# os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")

# COMMAND ----------

mlflow.langchain.autolog()
config = ModelConfig(development_config='config.yml')

# COMMAND ----------

  # move these to a config file
# vs_endpoint = 'one-env-shared-endpoint-1'
vs_endpoint = config.get("vs_endpoint")
# vs_index_name = 'shm.dbdemos_llm_rag.databricks_documentation_shared_index'
vs_index_name = config.get("vs_index_name")
vs_index_name

# If we can avoid this it will simplify deployments a lot, let the packages handle the authentication
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
# DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
# os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST

# COMMAND ----------

# MAGIC %md
# MAGIC Define the retriever - this is a default retriever and should work? In my testing both filtering and hybrid search seem to be working fine.

# COMMAND ----------

from databricks_langchain.vectorstores import DatabricksVectorSearch

score_threshold = config.get('score_threshold')
num_results = config.get('num_results')

def get_retriever(persist_dir: str = None):
    vector_search = DatabricksVectorSearch(
        index_name=vs_index_name,
        columns=['id', 'url', 'content']
        )
    
    return vector_search.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={
            'k': num_results, 
            "score_threshold": score_threshold,
            'query_type': 'hybrid'
            }
    )

# test our retriever
vs_retriever = get_retriever()
similar_documents = vs_retriever.invoke("What is DBSQL?")

# # Specify the return type schema of our retriever, so that evaluation and UIs can
# # automatically display retrieved chunks
mlflow.models.set_retriever_schema(
    primary_key='id',
    text_column='content',
    doc_uri='url',
    name="vs_index",
)

# COMMAND ----------

doc_df = spark.table("shm.dbdemos_llm_rag.databricks_documentation")
unique_urls = doc_df.select("url").distinct().rdd.flatMap(lambda x: x).collect()
url_picklist = unique_urls[0:10]

# Or we can just embed the list in the config
url_picklist = config.get('url_picklist')
url_picklist

# COMMAND ----------

# MAGIC %md
# MAGIC May want to discuss OpenAI vs. ChatDatabricks

# COMMAND ----------

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from databricks_langchain.chat_models import ChatDatabricks

llm_max_tokens = config.get("llm_max_tokens")
llm_temperature = config.get("llm_temperature")

# I personally prefer OpenAI for generalizability but let's go with ChatDatabricks for now, it works very well with our ppt endpoints
chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct", 
    max_tokens=llm_max_tokens, 
    temperature=llm_temperature
    )

chat_model.invoke("What is DBSQL in AI/BI?")

# COMMAND ----------

# MAGIC %md
# MAGIC URL Extraction Chain 

# COMMAND ----------

config.get("url_extraction_prompt")

# COMMAND ----------

url_extraction_prompt = PromptTemplate(
    template=config.get("url_extraction_prompt").replace("{url_picklist}", str(url_picklist)),
    input_variables=["question"]
)

url_extraction_chain = LLMChain(
    llm=chat_model,
    prompt=url_extraction_prompt,
    output_key="url_to_filt"  # Specify the output key
)

url_extraction_chain.invoke({"question": "Where are my release notes?"})

# COMMAND ----------

# MAGIC %md
# MAGIC Define the retrieval chain via TransformChain

# COMMAND ----------

from langchain.chains import TransformChain

retrieval_template = config.get("retrieval_prompt")

prompt = PromptTemplate(
    template=retrieval_template, 
    input_variables=["context", "question"]
    )

def get_filtered_retriever(url_to_filt):
    retriever = get_retriever()
    retriever.search_kwargs["filters"] = {"url": url_to_filt}
    return retriever

def retrieval_qa_chain(inputs):
    question = inputs["question"]
    url_to_filt = inputs["url_to_filt"]
    # Get the filtered retriever
    filtered_retriever = get_filtered_retriever(url_to_filt)
    # Define the RetrievalQA chain with the filtered retriever
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=filtered_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    # Run the retrieval chain
    output = retrieval_chain({"query": question})
    # Return the outputs
    return {"result": output["result"], "source_documents": output["source_documents"]}
  
from langchain_core.output_parsers import StrOutputParser

retrieval_transform_chain = TransformChain(
    input_variables=["question", "url_to_filt"],
    output_variables=["result", "source_documents"],
    transform=retrieval_qa_chain
)

# COMMAND ----------

# MAGIC %md
# MAGIC This works with the prompt and filter. For example, ask What is DBSQL while filtered to runtime release notes.

# COMMAND ----------

retrieval_transform_chain.invoke({
  "question": "What is DBSQL?",
  "url_to_filt": "https://docs.databricks.com/en/archive/runtime-release-notes/7.4.html"}
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Now we create the sequential chain. This works but we need to be very careful about the keys with the two chains, had to debug the keys quite a bit and make sure they were consistent.
# MAGIC

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain.chains import SequentialChain

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

seq_chain = SequentialChain(
    chains=[url_extraction_chain, retrieval_transform_chain],
    input_variables=["question"],
    output_variables=["result", "source_documents"],
    verbose=True
)

# Define the SequentialChain
# I added a very basic chat parser
# Note how we add a string output parser to make the serving work - lots of options here too
final_chain = (
    {
        "question": itemgetter("messages") 
        | RunnableLambda(extract_user_query_string), 
    }
    | seq_chain 
    | (lambda x: x["result"] if isinstance(x, dict) and "result" in x else x) 
    | StrOutputParser()
)

chain_output = final_chain.invoke({"messages":[
  {"role": "user", "content": "What is DBSQL?"}
]})

# COMMAND ----------

import pandas as pd

# Extract the outputs
answer = chain_output["result"]
source_documents = chain_output["source_documents"]

# Create the DataFrame with source documents
source_df = pd.DataFrame([
    {
        'content': doc.to_json()['kwargs']['page_content'],
        'id_name': doc.to_json()['kwargs']['metadata']['url'].split('/')[-2]
          if len(doc.to_json()['kwargs']['metadata']['url'].split('/')) > 1 
          else "Unknown",
        'url': doc.to_json()['kwargs']['metadata'].get('url', "No Source")
    }
    for doc in source_documents
])

display(source_df)

# COMMAND ----------


