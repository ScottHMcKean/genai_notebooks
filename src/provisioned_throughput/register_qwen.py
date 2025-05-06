# Databricks notebook source
# MAGIC %md
# MAGIC ## Register Qwen
# MAGIC This further modifies the Qwen model and registers it in Unity Catalog to prepare for serving.

# COMMAND ----------

# MAGIC %pip install -U mlflow transformers==4.44.2 torch==2.3.0 accelerate==0.29.0 torchvision
# MAGIC %restart_python

# COMMAND ----------

import json, os

import mlflow

# Path to Qwen model converted to Llama architecture
dbutils.widgets.text("model_path", "/dbfs/models/Qwen/Qwen2.5-14B-Instruct-Llama")

# Where to register the model in Unity Catalog
dbutils.widgets.text("catalog", "shm")
dbutils.widgets.text("schema", "default")

# The name of the model in Unity Catalog. If unspecified the name will be generated from the
# input path automatically.
dbutils.widgets.text("model_name", "qwen25_14b")

# COMMAND ----------

model_path = dbutils.widgets.get("model_path")
model_name = dbutils.widgets.get("model_name")

assert model_path

if not model_name:
    model_name = model_path.rstrip("/").split("/")[-1]
    model_name = model_name.replace(".", "").lower()

print(f"Model path: {model_path}")
print(f"Model name: {model_name}")

# COMMAND ----------

# Modify the tokenizer files and model config in the source directory

# Load the tokenizer config
with open(os.path.join(model_path, "tokenizer_config.json")) as f:
    tokenizer_config_obj = json.loads(f.read())

# Remove Qwen’s chat template since it is not recognized by model serving.
if "chat_template" in tokenizer_config_obj:
    del tokenizer_config_obj["chat_template"]

# Update the tokenizer class so that Databricks sees it as "PreTrainedTokenizerFast",
# since Qwen's tokenizer is not recognized, and it derives from PreTrainedTokenizerFast.  
# This also avoids saving additional files during registration that model
# serving would not expect.
tokenizer_config_obj["tokenizer_class"] = "PreTrainedTokenizerFast"

# Model serving expects model_input_names to be specified for Llama models.
tokenizer_config_obj["model_input_names"] = ["input_ids", "attention_mask"]

# Write the updated configs back
with open(os.path.join(model_path, "tokenizer_config.json"), "w") as f:
    f.write(json.dumps(tokenizer_config_obj, indent=2))

config_path = os.path.join(model_path, "config.json")
with open(config_path) as f:
    config_obj = json.loads(f.read())

# Set 'max_position_embeddings' to 16000 for compatibility with certain throughput settings.
config_obj["max_position_embeddings"] = 16000

with open(config_path, "w") as f:
    f.write(json.dumps(config_obj, indent=2))

# don't need the slow tokenizer files
files_to_delete = ["merges.txt", "vocab.json"]
for file_to_delete in files_to_delete:
    if os.path.exists(os.path.join(model_path, file_to_delete)):
        os.remove(os.path.join(model_path, file_to_delete))

# COMMAND ----------

# Register the model as a Llama model

# We'll create metadata that references Llama so Databricks sees "LlamaForCausalLM".
# Some of the specific versions/sizes referenced here don't matter for our purposes.
# The important thing is that model serving considers this a Llama model.
# Note that we register this as a completion model and removed the chat template.
# Chat formatting will need to be performed by the client.
task = "llm/v1/completions"
metadata = {
    "task": task,
    "curation_version": 1,
    "databricks_model_family": "LlamaForCausalLM (llama-3.2)",
    "databricks_model_size_parameters": "7b",
    "databricks_model_source": "genai-fine-tuning",
    "source": "huggingface",
    "source_model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "source_model_revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
}

input_example = {"prompt": "def print_hello_world():", "max_tokens": 20, "temperature": 0.05, "stop": ["\n\n"]}

# COMMAND ----------

# Use mlflow.transformers.log_model to register it to Unity Catalog

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

assert catalog
assert schema
assert model_name

registered_model_name = ".".join([catalog, schema, model_name])

print(f"Registering model as {registered_model_name}")

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=model_path,
        artifact_path="model",
        registered_model_name=registered_model_name,
        input_example=input_example,
        metadata=metadata,
        task=task,
        torch_dtype="bfloat16",
    )
