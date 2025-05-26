import os
import mlflow
from typing import List, Optional
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatMessage, ChatParams, ChatCompletionResponse
from openai import AzureOpenAI

from mlflow.models import ModelConfig
config = ModelConfig(development_config='config.yaml')

llm_api_key = os.getenv('API_SECRET_KEY')
search_api_key = os.getenv('SEARCH_SECRET_KEY')

class AzureChatRetriever(ChatModel):
    
    def predict(
        self, 
        context, 
        messages: List[ChatMessage], 
        params: Optional[ChatParams] = None
    ) -> ChatCompletionResponse:    
        
        client = AzureOpenAI(
            api_key=llm_api_key,
            api_version=config.get("api_version"),
            azure_endpoint=config.get("api_base"),
        )

        # system message
        # pass as dict
        system_prompt_ = config.get("system_prompt")
        system_content = system_prompt_.format("DATA1", "DATA2")
        system_message = {"role": "system", "content": system_content}

        user_message = messages[-1]
        if isinstance(user_message, ChatMessage):
            user_message = user_message.to_dict()
        message_list = [system_message, user_message]

        response = client.chat.completions.create(
            model=config.get("deployment_id"),
            messages=message_list,
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
                                "api_key":search_api_key,
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

        return ChatCompletionResponse.from_dict(response.model_dump())
        
chat_model = AzureChatRetriever()
mlflow.models.set_model(chat_model)
