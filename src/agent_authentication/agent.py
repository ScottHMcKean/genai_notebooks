from langchain.prompts import PromptTemplate
from databricks_langchain import ChatDatabricks
import mlflow

# Initialize Databricks LLM
databricks_llm = ChatDatabricks(
    endpoint="shm-gpt-4o-mini",
    temperature=0    
)

# Define the prompt template
template_instruction = (
    "Imagine you are a fine dining sous chef. Your task is to meticulously prepare for a dish, focusing on the mise-en-place process."
    "Your goal is to set the stage flawlessly for the chef to execute the cooking seamlessly."
    "The recipe you are given is for: {recipe} for {customer_count} people. "
)

# Create the prompt and chain using RunnableSequence
prompt = PromptTemplate(
    input_variables=["recipe", "customer_count"],
    template=template_instruction,
)
lc_agent = prompt | databricks_llm

# Set the model in MLflow
mlflow.models.set_model(lc_agent)