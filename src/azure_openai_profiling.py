# Databricks notebook source
# MAGIC %pip install openai mlflow azure-identity azure-search-documents wget
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

endpoint: str = "https://dbmma.openai.azure.com/"
api_key: str = dbutils.secrets.get('shm','azure-gpt4-key')
api_version: str = "2024-08-01-preview"
deployment_name = "gpt-4o-mini"
embedding_name = "text-embedding-3-small"
embedding_api_version = "2023-05-15"

# COMMAND ----------

embed_client = AzureOpenAI(
    api_key=api_key,
    api_version=embedding_api_version,
    azure_endpoint=endpoint,
)

# COMMAND ----------

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
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

search_service_endpoint: str = "https://fieldengeast-ai-search.search.windows.net"
search_service_api_key: str = dbutils.secrets.get('shm', 'azure-ai-search')
index_name: str = "dbmma-manufacturing"

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
    embeddings_response = embed_client.embeddings.create(model=embedding_name, input=text)
    # Extract the embedding data from the response
    embedding = embeddings_response.data[0].embedding
    return embedding

first_document_content = data_dict[0]["content"]
print(f"Content: {first_document_content[:100]}")

content_vector = generate_embeddings(first_document_content, deployment_name)
print("Content vector generated")

# COMMAND ----------

data = spark.table('shm.dbdemos_llm_rag.databricks_documentation').limit(1800).toPandas()

data_dict = data.to_dict(orient='records')

for i, x in enumerate(data_dict):
    vector = generate_embeddings(x['content'], deployment_name)
    data_dict[i]['vector'] = vector
    data_dict[i]['id'] = str(data_dict[i]['id'])

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

len(data_dict)

# COMMAND ----------

search_client.upload_documents(documents=data_dict)

# COMMAND ----------

with open("./test_data.json", "w") as json_file:
    json.dump(data_dict, json_file, indent=4)

# COMMAND ----------

batch_client = SearchIndexingBufferedSender(
    search_service_endpoint, index_name, credential
)

# COMMAND ----------

data_dict[1700]

# COMMAND ----------

query = "modern art in Europe"
  
search_client = SearchClient(search_service_endpoint, index_name, credential)  
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query], 
    select=["title", "text", "url"] 
)
  
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"URL: {result['url']}\n")  

# COMMAND ----------

from azure.core.exceptions import HttpResponseError

try:
    # Add upload actions for all documents in a single call
    batch_client.upload_documents(documents=data_dict)

    # Manually flush to send any remaining documents in the buffer
    batch_client.flush()
except HttpResponseError as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up resources
    batch_client.close()

# COMMAND ----------

data_dict

# COMMAND ----------

batch_client.upload_documents(
  documents=data_dict
  )

# COMMAND ----------

batch_client.flush()

# COMMAND ----------

batch_client.close()

# COMMAND ----------


