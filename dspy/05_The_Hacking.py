# Databricks notebook source
# MAGIC %pip install --upgrade dspy mlflow databricks-agents databricks-sdk databricks-mcp databricks-dspy uv unitycatalog-ai[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Making an Agent, or multi-agent? 
# MAGIC
# MAGIC Now that we've seen how to make a tool calling agent in dspy, we will use a combination of tools to create a multi-tool calling agent. 
# MAGIC
# MAGIC For this, we have prepared the following resources for you to use: 
# MAGIC 1. Genie Space called patient lookup with Fake Patient Data 
# MAGIC 2. Vector Search Index with sample insurance documents (genai_in_production_demo_catalog.agents.sbc_details_index). These sample insurance documents generally don't have the big company insurance documents in here so don't expect it to retrieve precise details on this
# MAGIC 3. UC Function that hits you.com 
# MAGIC
# MAGIC ### Try to recreate the Agent Bricks MAS you made earlier in code
# MAGIC
# MAGIC or
# MAGIC
# MAGIC ### Combine these tools to do the following:
# MAGIC
# MAGIC 1. Use a Genie Space to look up the details of a patient and their associated insurance documents 
# MAGIC 2. Use the UC function you.com to look up more information about the patient 
# MAGIC 3. Optional: Use the new DBSQL Managed MCP Server to write this information to a table
# MAGIC
# MAGIC Feel free to use the templates from notebook 4 and below to add more tools of your choice

# COMMAND ----------

import dspy
import mlflow
import databricks_dspy

mlflow.dspy.autolog()

# databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False)
databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-claude-sonnet-4-5', cache=False)
dspy.configure(lm=databricksLM)

predict = dspy.Predict("question->answer")

print(predict(question="why did a chicken cross the kitchen?"))



# COMMAND ----------

# MAGIC %md
# MAGIC #Make a UC Function
# MAGIC
# MAGIC Although the SQL code is provided below, it's mostly if you want to make your own UC function. The python code to hit the UC function is provided in Cell 6

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION genai_in_production_demo_catalog.agents.<TODO: update to your own name>(<TODO: update the input to what your function needs> COMMENT <TODO: add a description of what the input should be/look like>)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT <TODO: Add a description of what this function does to help the agent pick the function>
# MAGIC AS $$
# MAGIC <TODO: python code goes here. it's like defining a python function>
# MAGIC $$

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def web_search(query):
    """Use to research around a specific date. The query should be designed to search for news articles based on key information
    like dates, events, names of people and so forth"""

    client = DatabricksFunctionClient(execution_mode="local")
    result = client.execute_function(
        "genai_in_production_demo_catalog.agents.<TODO: update to your own name>",
        # "genai_in_production_demo_catalog.agents.search_web", #default you.com search function you can use
        parameters={"query": query}
    )
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC #Make a Genie Tool
# MAGIC
# MAGIC The Genie space has already been created for you. Here are the details: 
# MAGIC
# MAGIC Genie ID: 01f0aa277c491ec9bbed549be09984cd
# MAGIC Name: Patient Lookup 
# MAGIC
# MAGIC It has fake patient data and location they visited
# MAGIC
# MAGIC Documentation: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_ai_bridge.html

# COMMAND ----------

from databricks_ai_bridge.genie import Genie
from databricks.sdk import WorkspaceClient

#TODO: create a python function that returns the result of the Genie Space

def genie_tool():
    patient_lookup = Genie(
        space_id="01f0aa277c491ec9bbed549be09984cd", #change to your Genie Space ID
        client=WorkspaceClient()  
    )

    response = TODO
    return TODO

# COMMAND ----------

# MAGIC %md
# MAGIC #Use a Vector Search Index 
# MAGIC
# MAGIC Below is some sample code to interact with a prebuilt Vector Search Index that we made

# COMMAND ----------

from databricks_dspy import DatabricksRM

def health_insurance_look_up(question):
  """Used to query a vector search endpoint"""
  rm = DatabricksRM(
      databricks_index_name="genai_in_production_demo_catalog.agents.sbc_details_index", #change this to your index
      databricks_token="your_databricks_token", #change this to your token, Optional if working in a databricks notebook
      columns=["chunk_id", "content"], #change these to columns you would like to query and retrieve 
      text_column_name="content", #change this to the text column being retrieved 
      docs_id_column_name="chunk_id", #change to the ID 
      k=3 #change this to how many rows you would like to retrieve
  )

  result = rm(query=question) #adjust result to return what you're looking for 

  return result

# COMMAND ----------

# MAGIC %md
# MAGIC #MCP Interaction
# MAGIC
# MAGIC The DBSQL MCP just came out! Here's how to use it in case you want to use MCP

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
    f"{host}/api/2.0/mcp/sql" #this is the new MCP server 
]

databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-claude-sonnet-4-5', cache=False)
dspy.configure(lm=databricksLM)

async with streamablehttp_client(
    url=f"{host}/api/2.0/mcp/sql", #this is the new MCP server
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

        result = await react.acall(question="find information of patients in genai_in_production_demo_catalog.agents.patient_vists")
        print(result.answer)


# COMMAND ----------

# MAGIC %md
# MAGIC #Put your tools together! 
# MAGIC Combine these tools to do the following:
# MAGIC
# MAGIC 1. Look up the details of a patient and their associated insurance documents
# MAGIC 2. Use the UC function you.com to look up more information about the patient
# MAGIC 3. Compile all the information and write it back to a delta table using DBSQL MCP server

# COMMAND ----------

# MAGIC %md
# MAGIC #Easy way
# MAGIC
# MAGIC You can use dspy.ReAct for an OOTB method to quickly stitch tools together
# MAGIC
# MAGIC Task: 
# MAGIC Combine these tools to do the following:
# MAGIC
# MAGIC 1. Use a Genie Space to look up the details of a patient and their associated insurance documents 
# MAGIC 2. Use the UC function you.com to look up more information about the patient 
# MAGIC 3. Optional: Use the new DBSQL Managed MCP Server to write this information to a table
# MAGIC
# MAGIC Remember, the tool description is very important to help the Agent identify what tool to call. Here is the Anthropic guide again: https://www.anthropic.com/engineering/writing-tools-for-agents 
# MAGIC

# COMMAND ----------

class patient_lookup_websearch(dspy.Signature):
  """TODO"""
  TODO = dspy.InputField()
  TODO = dspy.OutputField()

# COMMAND ----------

patient_lookup = dspy.ReAct(patient_lookup_websearch, tools=[TODO, TODO]) 

# COMMAND ----------

#Test
question = "Identify patients who have the most severe symptoms based on doctor notes"
dsa_result = patient_lookup(query=question)

# COMMAND ----------

from IPython.display import Markdown
Markdown(dsa_result.TODO)

# COMMAND ----------

# MAGIC %md
# MAGIC #Below is a template of defining your own module 
# MAGIC
# MAGIC This is typically used if you wish to add more python code inbetween LLM calls or want more control over what happens under the hood. You can define multiple python functions and other logic directly in a custom module. You do not need to use this method

# COMMAND ----------

class BasicModule(dspy.Module):
    """
    <TODO: Describe what this module does> 
    """
    
    def __init__(self):
        super().__init__()
        <TODO: add your 
        self.predictor = dspy.ChainOfThought(BasicSignature)
        
    def forward(self, input_field: str) -> dspy.Prediction:
        """
        Forward pass of the module.
        
        Args:
            input_field: Input string for processing
            
        Returns:
            dspy.Prediction object with output_field
        """
        prediction = self.predictor(input_field=input_field)
        return prediction

# COMMAND ----------


