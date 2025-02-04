# Databricks notebook source
# MAGIC %md
# MAGIC This notebook evaluates performance of an OpenAI model

# COMMAND ----------

# MAGIC %pip install openai==1.6.1
# MAGIC %pip install mlflow
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from openai import AzureOpenAI

class AIDAModel(mlflow.pyfunc.PythonModel):
    def generate_response(self, question):
        """
        Generate a response for a given question using predefined responses or Azure OpenAI.

        Args:
        - question (str): The question to generate a response for.
        - config (ConfigParser): Configuration settings.


        Returns:
        - tuple: A tuple containing the answer text and references.
        """

        azure_search_endpoint = ""
        azure_search_index = ""
        oi_api_key = ""
        search_index_key = ""
        
        deployment_id = "gpt-4"
        api_version = "2024-02-15-preview"
        api_base = "https://oi-insight-bis-ce-601.openai.azure.com/"

        temperature = 0
        top_p = 0.95
        max_tokens = 800
        in_scope = True
        top_n_documents =  10
        query_type = "vectorSemanticHybrid"
        semantic_configuration = "default"
        role_information = "You are an AI assistant that helps people find information. Your name is AIDA"
        deployment_name = "text-embedding-ada-002"
        strictness = "3"

        lower_question = question.lower().strip()
       

        system_content_ = '''
        You are an AI assistant and expert in analyzing and synthesizing information from the documents provided to you.\ 
        Do not respond to the questions out of your knowledge base.\
        If you are asked questions similar to the questions listed below, provide the answer given for each question:\
        ---- Question and Answers ----
        User: "what can you do?"
        AI: "I am an AI assistant that helps people find information based on the {} dataset provided to me."

        User: "what can you help me with?"
        AI: "I can read the {} dataset I have and give you the best answer to your question."

        User: "what is AIDA?"
        AI: "AIDA stands for AI Digital Assistant."

        User: "what is the name of the dataset you are answering question based on?
        AI: The dataset is called {}.
        '''
        data_name = "BAPS, BAMS, and BAERD "
        system_content = system_content_.format(data_name, data_name, data_name)

        message_text = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        try:

            client = AzureOpenAI(
                api_key=oi_api_key,
                api_version=api_version,
                azure_endpoint=api_base,
            )
            response = client.chat.completions.create(
                model=deployment_id,
                messages=message_text,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                extra_body={
                    "data_sources": [
                        {
                            "type": "azure_search",
                            "parameters": {
                                "endpoint": azure_search_endpoint,
                                "authentication": {
                                    "type": "api_key",
                                    "api_key": search_index_key,
                                },
                                # "filter": "group_ids/any(g:search.in(g, '744ed6bc-79e5-4b04-b472-67b2ae87273f'))",
                                "fieldsMapping": {
                                    "content_fields": ["content"],
                                    "title_field": "metadata_storage_name",
                                    "url_field": "metadata_storage_path",
                                    "filepath_field": "",
                                    # "vector_fields": ["vectorfield"],
                                },
                                "index_name": azure_search_index,
                                "in_scope": in_scope,
                                "top_n_documents": int(
                                top_n_documents
                                ),
                                "query_type": query_type,
                                "semantic_configuration": semantic_configuration
                                or "",
                                "role_information": role_information,
                                # "filter": config.get("OpenAIParams", "FILTER") or None,
                                "embedding_dependency": {
                                    "type": "deployment_name",
                                    "deployment_name": deployment_name,
                                },
                                "strictness": int(strictness),
                            },
                        }
                    ]
                },
            )

            if response and response.choices:
                combined_response = response.choices[
                    0].message.content

                return combined_response
            else:
                return ["No response generated."]
        except Exception as e:
            print("An error occurred while generating response: ")
            return ["Error processing your request."]
    
    def load_context(self, context):
        from openai import AzureOpenAI
        pass

    def predict(self, context, model_input: str):
        return self.generate_response(str(model_input["question"]))

# COMMAND ----------

import shutil
import os

model_path = "aida_model"

if os.path.exists(model_path):
    shutil.rmtree(model_path)

# Save the model
mlflow.pyfunc.save_model(
    path="aida_model",
    python_model=AIDAModel(),
    extra_pip_requirements=["openai==1.6.1"],
)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model("aida_model")

# Example input
model_input = [{"question":"Tell me about the docs that you can retrieve"}]

# COMMAND ----------

# Log the model
with mlflow.start_run() as run:
    run_id = run.info.run_id
    mlflow.pyfunc.log_model(
      "aida_model",
      python_model=AIDAModel(),
      extra_pip_requirements=["openai==1.6.1"]
    )

# COMMAND ----------

model_uri = f"runs:/{run_id}/aida_model"
model_name = "aida_model"
mlflow.register_model(model_uri, model_name)
