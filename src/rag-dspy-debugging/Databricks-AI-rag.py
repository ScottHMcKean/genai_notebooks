# Databricks notebook source
# MAGIC %md
# MAGIC # RAG AI patterns in Azure Databricks using langchain. 
# MAGIC Notebook Demo use RAG AI patterns in Azure Databricks using langchain. Can use Databicks DBRX or external models like Azure AI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install required libraries

# COMMAND ----------

# MAGIC %pip install -qqqq -U pypdf==4.1.0 databricks-vectorsearch transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.2 mlflow mlflow-skinny langchain_community==0.2.10
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize context and variables

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

# DBTITLE 1,Set workspace url and token
databricks_host = "https://"+spark.conf.get("spark.databricks.workspaceUrl")
databricks_pat = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()



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

# MAGIC %md
# MAGIC ## 4. Create volumes, tables and load data. 

# COMMAND ----------

# Create external volume that contains PDF
query = f"CREATE EXTERNAL VOLUME IF NOT EXISTS {rag_poc_catalog}.{rag_poc_schema}.{volume_name} LOCATION '{volume_location}'"
spark.sql(query)


# COMMAND ----------

# DBTITLE 1,Uncomment this if re-run the notebook from scratch
# spark.sql(f"DROP TABLE {raw_files_table_name};")
# spark.sql(f"DROP TABLE {parsed_docs_table_name};")
# spark.sql(f"DROP TABLE {chunked_docs_table_name};")


# COMMAND ----------

# DBTITLE 1,create external tables
# Run the query
spark.sql(f"CREATE TABLE IF NOT EXISTS {raw_files_table_name} LOCATION '{raw_delta_location}'")
spark.sql(f"CREATE TABLE IF NOT EXISTS {parsed_docs_table_name} LOCATION '{parsed_delta_location}'")
spark.sql(f"CREATE TABLE IF NOT EXISTS {chunked_docs_table_name} LOCATION '{chunked_delta_location}'")

# COMMAND ----------

# DBTITLE 1,load raw files into a table
# Load the raw riles
raw_files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.pdf")
    .load(source_volume_path)
)

# Save to a table
raw_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(raw_files_table_name)

# reload to get correct lineage in UC
raw_files_df = spark.read.table(raw_files_table_name)

# For debugging, show the list of files, but hide the binary content
display(raw_files_df.drop("content"))

# Check that files were present and loaded
if raw_files_df.count() == 0:
    display(
        f"`{source_volume_path}` does not contain any files.  Open the volume and upload at least file."
    )
    raise Exception(f"`{source_volume_path}` does not contain any files.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Chunk data,  load into base table and create vector index

# COMMAND ----------

# DBTITLE 1,import
from pypdf import PdfReader
from typing import TypedDict, Dict
import warnings
import io 
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from functools import partial
import tiktoken
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml


# COMMAND ----------

# DBTITLE 1,define parsed content classes and functions
class ParserReturnValue(TypedDict):
    doc_parsed_contents: Dict[str, str]
    parser_status: str


def parse_bytes_pypdf(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)

        parsed_content = [page_content.extract_text() for page_content in reader.pages]
        output = {
            "num_pages": str(len(parsed_content)),
            "parsed_content": "\n".join(parsed_content),
        }

        return {
            "doc_parsed_contents": output,
            "parser_status": "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            "doc_parsed_contents": {"num_pages": "", "parsed_content": ""},
            "parser_status": f"ERROR: {e}",
        }

# COMMAND ----------

# DBTITLE 1,create parse UDF
parser_udf = func.udf(
    parse_bytes_pypdf,
    returnType=StructType(
        [
            StructField(
                "doc_parsed_contents",
                MapType(StringType(), StringType()),
                nullable=True,
            ),
            StructField("parser_status", StringType(), nullable=True),
        ]
    ),
)

# COMMAND ----------

# DBTITLE 1,display raw table
raw_files_df = spark.read.table(raw_files_table_name)
display(raw_files_df)

# COMMAND ----------

print(parsed_delta_name)

# COMMAND ----------

# DBTITLE 1,save parsed content table
parsed_files_staging_df = raw_files_df.withColumn("parsing", parser_udf("content")).drop("content")


# Check and warn on any errors
errors_df = parsed_files_staging_df.filter(
    func.col(f"parsing.parser_status")
    != "SUCCESS"
)

num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} documents had parse errors.  Please review.")
    display(errors_df)

# Filter for successfully parsed files
parsed_files_df = parsed_files_staging_df.filter(parsed_files_staging_df.parsing.parser_status == "SUCCESS").withColumn("doc_parsed_contents", func.col("parsing.doc_parsed_contents")).drop("parsing")

# Write to Delta Table
parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(parsed_docs_table_name)

# reload to get correct lineage in UC
parsed_files_df = spark.table(parsed_docs_table_name)

# Display for debugging
print(f"Parsed {parsed_files_df.count()} documents.")

display(parsed_files_df)

# COMMAND ----------

# DBTITLE 1,Text chunk type and function
class ChunkerReturnValue(TypedDict):
    chunked_text: str
    chunker_status: str

def chunk_parsed_content_langrecchar(
    doc_parsed_contents: str, chunk_size: int, chunk_overlap: int, embedding_config
) -> ChunkerReturnValue:
    try:
        # Select the correct tokenizer based on the embedding model configuration
        if (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "hugging_face"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "tiktoken"
        ):
            tokenizer = tiktoken.encoding_for_model(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        chunks = text_splitter.split_text(doc_parsed_contents)
        return {
            "chunked_text": [doc for doc in chunks],
            "chunker_status": "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            "chunked_text": [],
            "chunker_status": f"ERROR: {e}",
        }

# COMMAND ----------

# DBTITLE 1,Text chunk UDF
chunker_conf = data_pipeline_config.get("chunker")

chunker_udf = func.udf(
    partial(
        chunk_parsed_content_langrecchar,
        chunk_size=chunker_conf.get("config").get("chunk_size_tokens"),
        chunk_overlap=chunker_conf.get("config").get("chunk_overlap_tokens"),
        embedding_config=embedding_config,
    ),
    returnType=StructType(
        [
            StructField("chunked_text", ArrayType(StringType()), nullable=True),
            StructField("chunker_status", StringType(), nullable=True),
        ]
    ),
)

# COMMAND ----------

# DBTITLE 1,Chunk data and save into table
# Run the chunker
chunked_files_df = parsed_files_df.withColumn(
    "chunked",
    chunker_udf("doc_parsed_contents.parsed_content"),
)

# Check and warn on any errors
errors_df = chunked_files_df.filter(chunked_files_df.chunked.chunker_status != "SUCCESS")

num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} chunks had parse errors.  Please review.")
    display(errors_df)

# Filter for successful chunks
chunked_files_df = chunked_files_df.filter(chunked_files_df.chunked.chunker_status == "SUCCESS").select(
    "path",
    func.explode("chunked.chunked_text").alias("chunked_text"),
    func.md5(func.col("chunked_text")).alias("chunk_id")
)


# Write to Delta Table
chunked_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(chunked_docs_table_name)


# Enable CDC for Vector Search Delta Sync
spark.sql(
    f"ALTER TABLE {chunked_docs_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(f"Produced a total of {chunked_files_df.count()} chunks.")

# Display without the parent document text - this is saved to the Delta Table
display(chunked_files_df)


# COMMAND ----------

# DBTITLE 1,Create vector search endpoint
from databricks.vector_search.client import VectorSearchClient
import os
import json

#vsc = VectorSearchClient(workspace_url=databricks_host, service_principal_client_id=client_id, service_principal_client_secret=client_secret, azure_tenant_id=tenant_id, azure_login_id=login_id)
vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=databricks_pat)

endpoints_response = vsc.list_endpoints()
endpoints = endpoints_response.get('endpoints', [])
endpoint_exists = any(endpoint['name'] == vector_search_endpoint_name for endpoint in endpoints)

if endpoint_exists:
    print(f"Endpoint '{vector_search_endpoint_name}' exists.")
else:
    vsc.create_endpoint(
        name=vector_search_endpoint_name,
        endpoint_type="STANDARD"
    )

# COMMAND ----------

# DBTITLE 1,create vector index
from databricks.vector_search.client import VectorSearchClient
import os
import json

vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=databricks_pat)

# List all indexes for the specified endpoint
indexes_response = vsc.list_indexes(name=vector_search_endpoint_name)
indexes = indexes_response.get('vector_indexes', [])
# Check if your index exists
index_exists = any(index['name'] == f"{vector_index_name}" for index in indexes)

if index_exists:
    print(f"Index '{vector_index_name}' exists.")
else:
  index = vsc.create_delta_sync_index(
    endpoint_name=vector_search_endpoint_name,
    source_table_name=chunked_docs_table_name,
    index_name=vector_index_name,
    pipeline_type='TRIGGERED',
    primary_key="chunk_id",
    embedding_source_column="chunked_text",
    embedding_model_endpoint_name=embedding_service_endpoint
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test search and AI model

# COMMAND ----------

# DBTITLE 1,Test vector search
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

import os

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=databricks_pat)
    #vsc = VectorSearchClient(workspace_url=databricks_host, service_principal_client_id=client_id, service_principal_client_secret=client_secret, azure_tenant_id=tenant_id, azure_login_id=login_id)
#    vector_index_name = "rag_poc_vector_search_index"
    vs_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vector_index_name)

    # Create the retriever
    return DatabricksVectorSearch(vs_index).as_retriever()

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("Responsibilities Director HR?")
print(f"Relevant documents: {similar_documents[0]}")



# COMMAND ----------

# DBTITLE 1,test LLM endpoint
from langchain_community.chat_models import ChatDatabricks
from langchain.pydantic_v1 import BaseModel, Field
chat_model = ChatDatabricks(endpoint=model_name, max_tokens = 200)
print(f"Test chat model: {chat_model.invoke('What is Apache Spark')}")


# COMMAND ----------

# DBTITLE 1,Test single request chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever=get_retriever()

template = """
Assistant helps the company employees with their questions on company policies, roles. 
Always include the source metadata for each fact you use in the response. Use square brakets to reference the source, e.g. [role_library_pdf-10]. 
Properly format the output for human readability with new lines.
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever   , "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

#chain.invoke("where did harrison work?")
result = chain.invoke("Responsibilities of Director of Operations?")
print (result)

# COMMAND ----------

with mlflow.start_run(run_name='inital_rag'):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), 'rag_chain.py'
        ),
        model_config=rag_chain_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=rag_chain_config[
            "input_example"
        ],  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        extra_pip_requirements=["databricks-agents"] # TODO: Remove this
    )

# COMMAND ----------

# DBTITLE 1,Prepare GenAI questions
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql import Row


# Define a list of sample questions for an LLM
questions = [
    "Responsibilities of Director of Operations?",
    "Responsibilities of Senior Manager Human resources?",
    "Responsibilities of Directory of Chief Financial Officer?"
]

questions_df = spark.createDataFrame(
    [Row(question= q) for q in questions]
)
display(questions_df)

# COMMAND ----------

# DBTITLE 1,UDF for langchain


# Define the UDF to get chat completions
def call_chain(question):
    template = """
    Assistant helps the company employees with their questions on company policies, roles. 
    Always include the source metadata for each fact you use in the response. Use square brakets to reference the source, e.g. [role_library_pdf-10]. 
    Properly format the output for human readability with new lines.
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever   , "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    result = chain.invoke(question)
    return result

# Register the UDF
call_chain_udf = udf(call_chain, StringType())

# COMMAND ----------

df = questions_df.repartition(3).withColumn("answer", call_chain_udf(questions_df["question"]))
display(df)
