# Databricks notebook source
# MAGIC %md
# MAGIC #Databricks and DSPy
# MAGIC
# MAGIC In the last section, you saw how DSPy has some cutting-edge techniques to help you go to production. Once you have you AI Agents in good shape after developing modular components and optimizing the prompts, you can use Databricks to deploy your agents and use them in production. 
# MAGIC
# MAGIC In this notebook, we will cover some other capabilities that you can use DSPy with so that you have it as reference. These capabilities are not necessary for production but important to keep in mind when you eventually need it

# COMMAND ----------

# MAGIC %md
# MAGIC #Databricks AI Bridge and DSPy 
# MAGIC
# MAGIC This is an ongoing effort at Databricks to create a library that better integrates with Databricks products, particularly with authentication. AI Bridge helps pass your workspace client credentials throughout DSPy! 
# MAGIC
# MAGIC All you need to do is use databricks_dspy to set the LLM and your remaining DSPy code will be compatible! 
# MAGIC
# MAGIC Databricks AI Bridge also provides a way to interact with Genie Spaces and Vector Search, examples of which are provided below

# COMMAND ----------

# MAGIC %pip install --upgrade dspy mlflow databricks-agents databricks-sdk databricks-mcp databricks-dspy uv unitycatalog-ai[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dspy
import mlflow
import databricks_dspy

mlflow.dspy.autolog()

databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False)
dspy.configure(lm=databricksLM)

predict = dspy.Predict("question->answer")

print(predict(question="why did a chicken cross the kitchen?"))



# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search Example

# COMMAND ----------

from databricks_dspy import DatabricksRM

def vector_search_look_up(question):
  """Used to query a vector search endpoint"""
  rm = DatabricksRM(
      databricks_index_name="catalog.schema.your_index_name", #change this to your index
      databricks_endpoint="https://your-workspace.cloud.databricks.com", #change this to your workspace URL. Optional if working in a databricks notebook
      databricks_token="your_databricks_token", #change this to your token, Optional if working in a databricks notebook
      columns=["id", "text", "metadata", "text_vector"], #change these to columns you would like to query and retrieve 
      text_column_name="text", #change this to the text column being retrieved 
      docs_id_column_name="id", #change to the ID 
      k=3 #change this to how many rows you would like to retrieve
  )

  result = rm(query=question) #adjust result to return what you're looking for 

  return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Genie Example

# COMMAND ----------

from databricks_ai_bridge.genie import Genie
from databricks.sdk import WorkspaceClient

customer_product_lookup = Genie(
    space_id="", #change to your Genie Space ID
    client=WorkspaceClient()  
)

response = customer_product_lookup.ask_question("what is the latest vehicle Dawn RADANOVITZ-MINNITI purchased")
print(response.result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##UC Functions

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def system_python_execution(query):
    """used to execute python code"""

    client = DatabricksFunctionClient(execution_mode="local")
    result = client.execute_function(
        "system.ai.python_exec",
        parameters={"query": query}
    )
    return result

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION genai_in_production_demo_catalog.agents.search_web(query STRING COMMENT 'The search query string to send to the web search API')
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "Use this function to search the web for information when you cannot answer a question"
# MAGIC AS $$
# MAGIC     import requests
# MAGIC
# MAGIC     API_KEY = "" 
# MAGIC
# MAGIC     # Define the search query and other parameters
# MAGIC     #query = "Python programming"
# MAGIC
# MAGIC     # Construct the API endpoint URL
# MAGIC     url = "https://google.serper.dev/search"
# MAGIC
# MAGIC     # Define the headers, including your API key
# MAGIC     headers = {
# MAGIC         "X-API-KEY": API_KEY,
# MAGIC         "Content-Type": "application/json"
# MAGIC     }
# MAGIC
# MAGIC     # Define the parameters for the GET request
# MAGIC     params = {
# MAGIC         "q": query
# MAGIC     }
# MAGIC
# MAGIC     return requests.get(url, headers=headers, params=params).json()
# MAGIC $$

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

CATALOG = "genai_in_production_demo_catalog"
SCHEMA = "agents"

def search_function(code: str) -> str:
    """
    Stores Python code as a string.
    Args:
        code (str): The Python code to store.
    Returns:
        str: The stored code.
    """

    code = """    import requests

    API_KEY = "" 

    # Define the search query and other parameters
    #query = "Python programming"

    # Construct the API endpoint URL
    url = "https://google.serper.dev/search"

    # Define the headers, including your API key
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }

    # Define the parameters for the GET request
    params = {
        "q": query
    }

    return requests.get(url, headers=headers, params=params).json()"""
    return code

function_info = client.create_python_function(
    func=search_function,
    catalog=CATALOG,
    schema=SCHEMA,
    replace=True
)
print(function_info)

# COMMAND ----------

# MAGIC %md
# MAGIC #Databricks MCP and DSPy
# MAGIC
# MAGIC DSPy has native capabilities in interacting with any MCP server. Here's an example of how you would do this with a Databricks MCP server

# COMMAND ----------

from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from databricks.sdk import WorkspaceClient
import dspy

# Initialize the Databricks workspace client
workspace_client = WorkspaceClient()

host = workspace_client.config.host
MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/functions/system/ai",
]

dspy.configure(lm=dspy.LM("databricks/databricks-gpt-oss-120b"))

async with streamablehttp_client(
    url=f"{host}/api/2.0/mcp/functions/system/ai",
    auth=DatabricksOAuthClientProvider(workspace_client),
) as (read_stream, write_stream, _):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(tools)
        # Convert MCP tools to DSPy tools
        dspy_tools = []
        for tool in tools.tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

        # Create the agent
        react = dspy.ReAct("question -> answer", tools=dspy_tools)

        result = await react.acall(question="What is 5**5?")
        print(result)


# COMMAND ----------

# MAGIC %md
# MAGIC #Databricks ai_query and DSPy
# MAGIC
# MAGIC

# COMMAND ----------

import os
import mlflow
import databricks_dspy
from mlflow.types.type_hints import TypeFromExample
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
)
from pkg_resources import get_distribution

resources = [
    DatabricksServingEndpoint(endpoint_name="databricks-gpt-oss-120b"),
]

mlflow.dspy.autolog()

databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False)

dspy.settings.configure(lm=databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False))

class MyModel(mlflow.pyfunc.PythonModel):
  def __init__(self):
      self.program = dspy.Predict("question -> answer")
  
  def predict(self, model_input: TypeFromExample):
    dspy.configure(lm=databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False))
    if hasattr(model_input, "to_dict"):
        model_input = model_input.to_dict('records')
    result = self.program(question=model_input[0]['question']).answer
    return result

logged_agent_info = mlflow.pyfunc.log_model(python_model=MyModel(), name="dspy-ai-query-test", input_example=[{"question": "What is MLflow?"}], resources=resources)

# COMMAND ----------

uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name="austin_choi_demo_catalog.agents.dspy-ai-query-test")

# COMMAND ----------

# Load version 1 (or whichever version you registered)
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/9")

# Or use the version from your registration
loaded_model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/{uc_registered_model_info.version}"
)

result = loaded_model.predict([{"question": "What is MLflow?"}])
print(result)

# COMMAND ----------

import mlflow.deployments

# Initialize the deployment client
client = mlflow.deployments.get_deploy_client("databricks")

# Create the serving endpoint
endpoint = client.create_endpoint(
    name="dspy-ai-query-endpoint",  # Choose your endpoint name
    config={
        "served_entities": [
            {
                "name": "dspy-entity",
                "entity_name": "austin_choi_demo_catalog.agents.dspy-ai-query-test",
                "entity_version": "1",  # Specify the model version
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": "dspy-entity",
                    "traffic_percentage": 100
                }
            ]
        }
    }
)


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_query(
# MAGIC     endpoint => 'dspy-ai-query-endpoint',
# MAGIC     request => named_struct(
# MAGIC       "question", "what is MLflow?"
# MAGIC     )
# MAGIC );

# COMMAND ----------


