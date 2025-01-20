# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC %pip install langchain
# MAGIC %pip install databricks-agents
# MAGIC %pip install databricks-langchain
# MAGIC %restart_python

# COMMAND ----------

import mlflow
mlflow.end_run()

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksTable,
)

with mlflow.start_run():
    # Set the registry URI to Unity Catalog if needed
    mlflow.set_registry_uri('databricks-uc')

    # Define the list of Databricks resources needed to serve the agent
    list_of_databricks_resources = [
        DatabricksServingEndpoint(endpoint_name="shm-gpt-4o-mini"),
    ]

    # Define the custom signature
    input_schema = Schema([
        ColSpec("string", "recipe"),
        ColSpec("long", "customer_count")
    ])
    output_schema = Schema([ColSpec("string")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Log the model in MLflow with the signature  
    logged_agent_info = mlflow.langchain.log_model(
        lc_model="./agent.py",
        artifact_path="model",
        resources=list_of_databricks_resources, 
        signature=signature)

    uc_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, 
        name="shm.default.mlflow_example")

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

input_data = {
    "recipe": "Wiener Schnitzel",
    "customer_count": 3  # Replace with your long integer value
}
model = mlflow.langchain.load_model('models:/shm.default.mlflow_example/2')
print(model)

result = model.invoke(input_data)
print(result)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "shm.default.mlflow_example"
model_version = "3"

model_uri = client.get_model_version_download_uri(model_name, model_version)
print('Model URI:', model_uri)

# COMMAND ----------

import mlflow

# model_uri = 'models:/shm.default.mlflow_example/3'

# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
)

# COMMAND ----------

from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

# Deploy the model to serving
deploy_name = "shm-mlflow-test"
model_name = "shm.default.mlflow_example"
model_version = 2

endpoint = client.create_endpoint(
    name=f"{deploy_name}_{model_version}",
    config={
        "served_entities": [{
            "entity_name": model_name,
            "entity_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }],
        "traffic_config": {
            "routes": [{
                "served_model_name":  f"{deploy_name}_{model_version}",
                "traffic_percentage": 100
            }]
        }
    }
)

# COMMAND ----------


