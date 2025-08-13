from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from databricks_langchain import ChatDatabricks
from databricks.vector_search.client import VectorSearchClient
from mlflow.entities import SpanType
import mlflow
from langchain_core.language_models.llms import create_base_retry_decorator

from fastapi import APIRouter
from typing import Dict
import sys

router = APIRouter()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPERATURE = 0.1
MAX_TOKENS = 2048

retry_error_types = (Exception,)
llm_retry_strategy = create_base_retry_decorator(
    error_types=retry_error_types,
    max_retries=5,
)


def _create_vector_search_client() -> VectorSearchClient:
    """Create a VectorSearchClient using either PAT or service principal env vars.

    Supported env variables:
      - DATABRICKS_HOST (required)
      - DATABRICKS_TOKEN (PAT)
    """
    try:
        host = os.getenv("DATABRICKS_HOST")
        assert host is not None, "DATABRICKS_HOST is not set"
        token = os.getenv("PAT")
        assert token is not None, "PAT is not set"
    except Exception as e:
        logger.error(
            f"Error getting DATABRICKS_HOST, trying SDK: {str(e)}",
            exc_info=True,
        )
        from databricks.sdk.core import Config

        config = Config(profile="DEFAULT")
        token = config.oauth_token().access_token
        host = config.host

    return VectorSearchClient(
        workspace_url=host,
        personal_access_token=token,
    )


class LLMClient:
    def __init__(self) -> None:
        self.dbr_llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        self.dbr_llm_mini = ChatDatabricks(
            endpoint="databricks-meta-llama-3-1-8b-instruct",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        try:
            self.vsc = _create_vector_search_client()
            self.vs_index = self.vsc.get_index(
                endpoint_name="one-env-shared-endpoint-3",
                index_name="shm.multimodal.index",
            )
        except Exception as e:
            logger.error(
                f"Error initializing vector search client: {str(e)}",
                exc_info=True,
            )
            self.vsc = None
            self.vs_index = None

        logger.info("LLMClient initialized")

    @llm_retry_strategy
    @mlflow.trace(
        span_type=SpanType.CHAT_MODEL, name="evaluate_context_sufficiency_llm"
    )
    async def evaluate_context_sufficiency(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Evaluate if existing context is sufficient to answer user's question."""
        logger.info("Evaluating context sufficiency for follow-up conversation")

        try:
            initial_retrieved_context = self.vs_index.similarity_search(
                query_text=user_message,
                columns=["enriched_text", "headings"],
            )
        except Exception as e:
            logger.error(
                f"Error retrieving context: {str(e)}",
                exc_info=True,
            )
            initial_retrieved_context = []

        context_text = "\n\n".join(
            f"Headings: {doc[1]}, Content: {doc[0]}"
            for doc in initial_retrieved_context["result"]["data_array"]
        )

        prompt = PromptTemplate.from_template(
            """You are a specialized routing node. Your purpose is to determine if a `User's Current Question` requires fetching new documents, given the `Chat History` and any `Previously Retrieved Documents`.

            ### Decision Logic:
            Set `need_retrieval` to `true` if:
            * The `User's Current Question` asks for new facts, details, or topics not present in the `Chat History` or `Previously Retrieved Documents`.

            Set `need_retrieval` to `false` if:
            * The `User's Current Question` is a rephrasing or clarification that can be answered using only the information already in the `Chat History` or `Previously Retrieved Documents`.

            <BEGIN CONTEXT>
            Chat History:
            {chat_history}

            User's Current Question:
            {user_question}

            Previously Retrieved Documents:
            {retrieved_documents}
            <END CONTEXT>

            **Provide your assessment in JSON format.**
            ```json
            {{
            "need_retrieval": true, // boolean: true if additional documents are likely needed, false otherwise
            "reasoning": "A concise explanation for the decision (e.g., 'Follow-up asks for new details on X not in original summary.', or 'Question is a rephrasing of previous info.')"
            }}
            ```"""
        )

        base_chain = prompt | self.dbr_llm | JsonOutputParser()

        input_dict = {
            "chat_history": "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in chat_history]
            ),
            "user_question": user_message,
            "retrieved_documents": context_text,
        }

        try:
            result = await base_chain.ainvoke(input_dict)
            if not isinstance(result, dict) or "need_retrieval" not in result:
                logger.warning(
                    "Invalid result structure during evaluate_context_sufficiency: %s",
                    result,
                )
                result = {
                    "need_retrieval": True,
                    "reasoning": "Could not reliably determine context sufficiency.",
                }
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Unexpected error in evaluate_context_sufficiency: {str(e)}",
                exc_info=True,
            )
            result = {
                "need_retrieval": True,
                "reasoning": f"Error during evaluation: {e}",
            }
        return result


client = LLMClient()


class Message(BaseModel):
    role: str
    content: str


class EvaluateRequest(BaseModel):
    user_message: str
    chat_history: List[Message]


@router.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    return await client.evaluate_context_sufficiency(
        user_message=req.user_message,
        chat_history=[m.model_dump() for m in req.chat_history],
    )
