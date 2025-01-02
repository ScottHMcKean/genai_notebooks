import os
import mlflow
from mlflow.models import ModelConfig

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import TransformChain
from langchain.chains import SequentialChain

from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

mlflow.langchain.autolog()
config = ModelConfig(development_config='config.yml')

## RETRIEVER

vs_endpoint = config.get("vs_endpoint")
vs_index_name = config.get("vs_index_name")
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

vs_retriever = get_retriever()
mlflow.models.set_retriever_schema(
    primary_key='id',
    text_column='content',
    doc_uri='url',
    name="vs_index",
)

### URL EXTRACTOR

url_picklist = config.get('url_picklist')
llm_max_tokens = config.get("llm_max_tokens")
llm_temperature = config.get("llm_temperature")

chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct", 
    max_tokens=llm_max_tokens, 
    temperature=llm_temperature
    )

url_extraction_prompt = PromptTemplate(
    template=config.get("url_extraction_prompt").replace("{url_picklist}", str(url_picklist)),
    input_variables=["question"]
)

url_extraction_chain = LLMChain(
    llm=chat_model,
    prompt=url_extraction_prompt,
    output_key="url_to_filt"  # Specify the output key
)

## RETRIEVAL CHAIN

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
  
retrieval_transform_chain = TransformChain(
    input_variables=["question", "url_to_filt"],
    output_variables=["result", "source_documents"],
    transform=retrieval_qa_chain
)

## SEQUENTIAL CHAIN

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

mlflow.models.set_model(final_chain)