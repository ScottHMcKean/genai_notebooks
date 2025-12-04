#!/usr/bin/env python3
"""
Simple example of using the Azure Search DsPy Information Extraction system.

This script demonstrates:
1. Loading configuration from environment
2. Initializing the retriever and agent
3. Running a single query
4. Optimizing prompts with MIPROv2
"""

import os
import sys
import dspy
import mlflow

# Add src to path
sys.path.insert(0, os.path.abspath("../src"))

from azure_search_dspy import (
    AzureSearchRM,
    InformationExtractionAgent,
    evaluate_quality,
)


def main():
    """Run example information extraction"""

    # 1. Configuration
    print("=" * 80)
    print("DsPy + Azure AI Search Information Extraction Example")
    print("=" * 80)

    # Load from environment
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

    # Validate configuration
    if not all(
        [
            AZURE_SEARCH_ENDPOINT,
            AZURE_SEARCH_KEY,
            AZURE_SEARCH_INDEX,
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_KEY,
        ]
    ):
        print("ERROR: Missing required environment variables.")
        print(
            "Please set: AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX,"
        )
        print("            AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY")
        return 1

    print(f"\n✓ Configuration loaded")
    print(f"  Search Index: {AZURE_SEARCH_INDEX}")
    print(f"  OpenAI Deployment: {AZURE_OPENAI_DEPLOYMENT}")

    # 2. Initialize components
    print("\n" + "=" * 80)
    print("Initializing Components")
    print("=" * 80)

    # Setup Azure Search retriever
    retriever = AzureSearchRM(
        search_endpoint=AZURE_SEARCH_ENDPOINT,
        search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX,
        k=5,
    )
    print("✓ Azure Search retriever initialized")

    # Setup Azure OpenAI LLM
    lm = dspy.AzureOpenAI(
        api_base=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-15-preview",
        deployment_id=AZURE_OPENAI_DEPLOYMENT,
        model_type="chat",
        max_tokens=2000,
        temperature=0.1,
    )
    dspy.settings.configure(lm=lm)
    print("✓ Azure OpenAI LLM configured")

    # Create agent
    agent = InformationExtractionAgent(retriever=retriever)
    print("✓ Information Extraction Agent created")

    # 3. Run example query
    print("\n" + "=" * 80)
    print("Running Example Query")
    print("=" * 80)

    query = "What are the key features and capabilities of Azure AI Search?"
    print(f"\nQuery: {query}")

    mlflow.set_experiment("/Shared/dspy_azure_search_demo")

    with mlflow.start_run(run_name="example_query"):
        result = agent(query)

        print("\n" + "-" * 80)
        print("SUMMARY:")
        print(result.summary)

        print("\n" + "-" * 80)
        print("KEY POINTS:")
        print(result.key_points)

        print("\n" + "-" * 80)
        print("ENTITIES:")
        print(result.entities_json[:500])  # Truncate for display

        print("\n" + "-" * 80)
        print("SOURCES:")
        for i, source in enumerate(result.sources, 1):
            print(f"  {i}. {source}")

        # Log to MLflow
        mlflow.log_param("query", query)
        mlflow.log_param("model", AZURE_OPENAI_DEPLOYMENT)
        mlflow.log_text(result.summary, "summary.txt")
        mlflow.log_text(result.key_points, "key_points.txt")

        print("\n✓ Results logged to MLflow")

    # 4. Demonstrate optimization (optional)
    print("\n" + "=" * 80)
    print("Prompt Optimization Available")
    print("=" * 80)
    print(
        "\nTo optimize prompts, use the full notebook (azure_search_extraction.ipynb)"
    )
    print("which includes:")
    print("  - Evaluation dataset creation")
    print("  - MIPROv2 prompt optimization")
    print("  - Baseline vs optimized comparison")
    print("  - MLflow GENAI evaluation metrics")

    print("\n" + "=" * 80)
    print("✓ Example Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

