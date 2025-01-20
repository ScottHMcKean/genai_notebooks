# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC %pip install databricks-agents
# MAGIC %pip install databricks-langchain
# MAGIC %pip install langchain
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import json
import subprocess

# Set the registry URI to Unity Catalog if needed
mlflow.set_registry_uri('databricks-uc')

# Define the model URI
model_uri = "models:/shm.default.mlflow_example/3"

# Fetch the model's environment file
env_file = mlflow.pyfunc.get_model_dependencies(model_uri)

# Install the dependencies using the environment file 
subprocess.run(['pip', 'install', '-r', env_file])

# COMMAND ----------

# Define the input data
input_data = {
    "recipe": "Wiener Schnitzel",
    "customer_count": 2  # Replace with your long integer value
}

# Load the model
model = mlflow.langchain.load_model(model_uri)
predictions = model.invoke(input_data)
print(predictions)

# COMMAND ----------

from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

# Deploy the model to serving
deploy_name = "shm-mlflow-test"
model_name = "shm.default.mlflow_example"
model_version = 3

endpoint = client.create_endpoint(
    name=f"{deploy_name}_{model_version}",
    config={
        "served_entities": [{
            "entity_name": model_name,
            "entity_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    }
)

# COMMAND ----------


