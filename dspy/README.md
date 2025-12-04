# DsPy + Azure AI Search Information Extraction

A modular, production-ready implementation of DsPy for information extraction and summarization using Azure AI Search and MLFlow 3.

## Features

- 🔍 **Azure AI Search Integration**: Native DsPy retrieval module with semantic search
- 🤖 **Information Extraction**: Structured entity extraction, summarization, and key points
- 📊 **MLFlow 3 Tracking**: Comprehensive experiment tracking and GENAI evaluation
- 🎯 **Prompt Optimization**: Automatic prompt improvement with MIPROv2
- 🧪 **Testable Design**: Modular components with unit tests
- 🚀 **Production Ready**: Ready for Databricks deployment

## Architecture

```
dspy/
├── azure_search_extraction.ipynb  # Main notebook
├── azure_config.yaml              # Configuration
└── README.md                      # This file

src/azure_search_dspy/
├── __init__.py                    # Package exports
├── retriever.py                   # Azure Search retrieval module
├── signatures.py                  # DsPy signatures
├── models.py                      # Pydantic data models
├── agent.py                       # Information extraction agent
└── evaluation.py                  # Evaluation metrics

tests/azure_search_dspy/
└── test_agent.py                  # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -U dspy-ai mlflow>=3.1.0 azure-search-documents azure-identity openai pydantic
```

### 2. Set Environment Variables

```bash
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_KEY="your-key"
export AZURE_SEARCH_INDEX="your-index"
export AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

### 3. Run the Notebook

Open `azure_search_extraction.ipynb` and run all cells.

## Components

### AzureSearchRM

Custom DsPy retrieval module for Azure AI Search:

```python
from azure_search_dspy import AzureSearchRM

retriever = AzureSearchRM(
    search_endpoint=AZURE_SEARCH_ENDPOINT,
    search_key=AZURE_SEARCH_KEY,
    index_name=AZURE_SEARCH_INDEX,
    k=5
)
```

### InformationExtractionAgent

Complete RAG agent with query rewriting, retrieval, summarization, and entity extraction:

```python
from azure_search_dspy import InformationExtractionAgent

agent = InformationExtractionAgent(retriever=retriever)
result = agent("What are the key features of Azure AI Search?")
```

### Evaluation

Built-in evaluation metrics for quality assessment:

```python
from azure_search_dspy import evaluate_quality, evaluate_agent

# Evaluate on a dataset
results = evaluate_agent(agent, testset)
print(f"Average Score: {results['average_score']:.2%}")
```

## Optimization

The notebook demonstrates automatic prompt optimization using DsPy's MIPROv2:

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    prompt_model=lm,
    task_model=lm,
    metric=evaluate_quality,
    num_candidates=5
)

optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset
)
```

## MLFlow Integration

All experiments are tracked in MLFlow with:

- Parameters (query, model, optimizer settings)
- Metrics (baseline_score, optimized_score, improvement)
- Artifacts (optimized agent, summaries, entities)
- GENAI evaluation (answer_relevance, answer_correctness)

## Customization

### Custom Evaluation Metrics

Edit `src/azure_search_dspy/evaluation.py` to add domain-specific metrics:

```python
def evaluate_quality(example, prediction, trace=None):
    # Add your custom evaluation logic
    score = 0.0
    # ... compute score
    return score
```

### Custom Signatures

Add new DsPy signatures in `src/azure_search_dspy/signatures.py`:

```python
class CustomSignature(dspy.Signature):
    """Your custom signature"""
    input_field = dspy.InputField(desc="...")
    output_field = dspy.OutputField(desc="...")
```

## Testing

Run unit tests:

```bash
uv run pytest tests/azure_search_dspy/ -v
```

## Deployment

### Databricks Model Serving

The optimized agent can be deployed to Databricks Model Serving:

1. Save the optimized agent: `optimized_agent.save("model.json")`
2. Log with MLFlow: `mlflow.log_artifact("model.json")`
3. Register as MLFlow model
4. Deploy to Model Serving endpoint

### API Endpoint

Wrap the agent in a FastAPI endpoint for production use.

## Best Practices

1. **Evaluation Dataset**: Use a diverse, high-quality evaluation dataset (300+ examples recommended)
2. **Iterative Optimization**: Run multiple optimization rounds with different hyperparameters
3. **Monitor Production**: Continuously evaluate production queries and retrain
4. **Version Control**: Track prompt versions and model configurations in MLFlow
5. **Testing**: Write unit tests for all components before deploying

## Troubleshooting

### Azure Search Connection Issues

- Verify endpoint URL and API key
- Check index exists and has data
- Ensure semantic configuration is set up for semantic search

### DsPy Optimization Takes Too Long

- Reduce `num_candidates` in MIPROv2
- Use a smaller training set
- Increase `num_threads` for parallel processing

### MLFlow Tracking Not Working

- Check MLFlow tracking URI is set
- Ensure experiment name is valid
- Verify permissions for Databricks workspace

## References

- [DsPy Documentation](https://dspy-docs.vercel.app/)
- [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/)
- [MLFlow 3](https://mlflow.org/docs/latest/index.html)
- [MLFlow GENAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

## License

MIT


