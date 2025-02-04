import mlflow
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]

# Enable MLflow tracing & logging
mlflow.langchain.autolog()

# Load the chain configuration
model_config = mlflow.models.ModelConfig(
    development_config="./fm_prompt_config.yaml")

# Define a prompt template
prompt = PromptTemplate(
    template=model_config.get("fm_prompt_template"),
    input_variables=["user_query"]
)

# Setup a foundation model
model = ChatDatabricks(
    endpoint=model_config.get("fm_serving_endpoint_name"),
    extra_params={
      "temperature": model_config.get("temperature"), 
      "max_tokens": model_config.get("max_tokens")
    }
)

# define a custom output parsing function
def split_string_output(output):
    """
    Split the string output into tokens and categories.
    """
    return {
        'tokens': re.search(r'Tokens:(.*?)\n', output, re.DOTALL).group(1),
        'categories': re.search(r'Categories:(.*)', output, re.DOTALL).group(1)
    }

# Define the basic chain
chain = (
    # This sections sets up the input extraction of the latest message
    # It will be useful if we want to incorporate RAG with NER examples
    {
        "user_query": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages")
        | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | StrOutputParser()
)

mlflow.models.set_model(chain)