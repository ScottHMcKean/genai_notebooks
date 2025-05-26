import os
import mlflow
from typing import List
from mlflow.pyfunc import PythonModel
from openai import AzureOpenAI

from mlflow.models import ModelConfig
config = ModelConfig(development_config='config.yaml')

llm_api_key = os.getenv('API_SECRET_KEY')
search_api_key = os.getenv('SEARCH_SECRET_KEY')

class AzureCompletionRetriever(PythonModel):
    
    def predict(self, context, model_input: list[str]) -> list[str]:    
        """
        Generate a response for a given question using predefined responses or Azure OpenAI.

        Args:
        - question (str): The question to generate a response for.
        - config (ConfigParser): Configuration settings.

        Returns:
        - tuple: A tuple containing the answer text and references.
        """
        if isinstance(model_input, list):
            model_input = model_input[-1]

        question = model_input.lower().strip()
       
        system_prompt_ = config.get("system_prompt")
        system_content = system_prompt_.format("DATA1", "DATA2")

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user","content": question}
        ]

        try:
            client = AzureOpenAI(
                api_key=llm_api_key,
                api_version=config.get("api_version"),
                azure_endpoint=config.get("api_base"),
            )
            response = client.chat.completions.create(
                model=config.get("deployment_id"),
                messages=messages,
                temperature=config.get("temperature"),
                top_p=config.get("top_p"),
                max_tokens=config.get("max_tokens"),
                extra_body={
                    "data_sources": [
                        {
                            "type": "azure_search",
                            "parameters": {
                                "endpoint": config.get("azure_search_endpoint"),
                                "authentication": {
                                    "type": "api_key",
                                    "api_key": search_api_key,
                                },
                                "fieldsMapping": {
                                    "content_fields": ["content"],
                                    "title_field": "metadata_storage_name",
                                    "url_field": "metadata_storage_path",
                                    "filepath_field": "",
                                },
                                "index_name": config.get("azure_search_index"),
                                "in_scope": config.get("in_scope"),
                                "top_n_documents": config.get("top_n_documents"),
                                "query_type": config.get("query_type"),
                                "semantic_configuration": config.get("semantic_configuration")
                                or "",
                                "role_information": config.get("role_information"),
                                "embedding_dependency": {
                                    "type": "deployment_name",
                                    "deployment_name": config.get("embedding_name"),
                                },
                                "strictness": config.get("strictness"),
                            },
                        }
                    ]
                },
            )

            if response and response.choices:
                combined_response = response.choices[
                    0].message.content

                return [combined_response]
            else:
                return ["No response generated."]
        except Exception as e:
            return([f"An error occurred while generating response: {e}"])
    
    def load_context(self, context):
        from openai import AzureOpenAI
        pass
  
completion_model = AzureCompletionRetriever()
mlflow.models.set_model(completion_model)
