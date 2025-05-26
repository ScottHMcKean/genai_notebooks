# Databricks notebook source
# MAGIC %pip install openai mlflow azure-identity azure-search-documents wget --upgrade
# MAGIC %restart_python

# COMMAND ----------

import json  
import wget
import pandas as pd
import zipfile
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

# COMMAND ----------

from mlflow.models import ModelConfig
config = ModelConfig(development_config='config.yaml')

# COMMAND ----------

api_base: str = config.get("api_base")
api_key: str = dbutils.secrets.get('shm',config.get("api_secret_key"))
api_version: str = config.get("api_version")
deployment_name = config.get("deployment_id")
embedding_name = config.get("embedding_name")
embedding_api_version = config.get("embedding_api_version")

# COMMAND ----------

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base,
)

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Azure OpenAI?"}
    ],
    max_tokens=100
)

response

# COMMAND ----------

search_service_endpoint: str = config.get("azure_search_endpoint")
search_service_api_key: str = dbutils.secrets.get('shm', config.get('search_secret_key'))
index_name: str = config.get("azure_search_index")

credential = AzureKeyCredential(search_service_api_key)

search_client = SearchClient(
    endpoint=search_service_endpoint, 
    index_name=index_name, 
    credential=credential
)

# COMMAND ----------

# Example function to generate document embedding
def generate_embeddings(text, model):
    # Generate embeddings for the provided text using the specified model
    embeddings_response = client.embeddings.create(
        model=embedding_name, 
        input=text)
    # Extract the embedding data from the response
    embedding = embeddings_response.data[0].embedding
    return embedding

data = spark.table('shm.dbdemos_llm_rag.databricks_documentation').limit(1800).toPandas()

data_dict = data.to_dict(orient='records')

first_document_content = data_dict[0]["content"]
print(f"Content: {first_document_content[:100]}")

content_vector = generate_embeddings(first_document_content, deployment_name)
print("Content vector generated")

# use this to generate embeddings prior to using the index
# for i, x in enumerate(data_dict)[0]:
#     vector = generate_embeddings(x['content'], deployment_name)
#     data_dict[i]['vector'] = vector
#     data_dict[i]['id'] = str(data_dict[i]['id'])

# COMMAND ----------

# Initialize the SearchIndexClient
index_client = SearchIndexClient(
    endpoint=search_service_endpoint, credential=credential
)

# Define the fields for the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="url", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
]

# Configure the vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="my-hnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="my-vector-config",
            algorithm_configuration_name="my-hnsw",
        )
    ],
)

# Configure the semantic search configuration
semantic_search = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                keywords_fields=[SemanticField(field_name="url")],
                content_fields=[SemanticField(field_name="content")],
            ),
        )
    ]
)

# Create the search index with the vector search and semantic search configurations
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search,
)

# Create or update the index
result = index_client.create_or_update_index(index)
print(f"{result.name} created")

# COMMAND ----------

# only used for initial loading
# search_client.upload_documents(documents=data_dict)

# COMMAND ----------

with open("./test_data.json", "w") as json_file:
    json.dump(data_dict, json_file, indent=4)

# COMMAND ----------

batch_client = SearchIndexingBufferedSender(
    search_service_endpoint, index_name, credential
)

# COMMAND ----------

query = "modern art in Europe"
  
search_client = SearchClient(
    search_service_endpoint, 
    index_name, 
    credential
    )  

vector_query = VectorizedQuery(
    vector=generate_embeddings(query, deployment_name), 
    k_nearest_neighbors=3, 
    fields="vector"
    )
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query], 
    select=["id", "content", "url"] 
)
  
for result in results:  
    print(f"ID: {result['id']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"URL: {result['url']}\n")  
