# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from utils import print_gpu_utilization, clear_gpu
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print_gpu_utilization()

# COMMAND ----------

from datasets import load_from_disk
dataset = load_from_disk('/Volumes/shm/work_order/datasets/workorder_text/')
dataset

# COMMAND ----------

import pandas as pd
out = dataset['train'].map(lambda x: {'len':len(x['text'])})
pd.DataFrame(out).len.max()

# COMMAND ----------

import mlflow
from mlflow.artifacts import download_artifacts
mlflow.set_registry_uri("databricks-uc")
uc_model_path = "models:/system.ai.llama_v3_2_3b_instruct/2"
base_model_local_path = "/local_disk0/llama_v3_2_3b_instruct/"
ft_model_local_path = "/local_disk0/llama_v3_2_3b_ft_activity/"
artifact_path = download_artifacts(
  artifact_uri=uc_model_path, 
  dst_path=base_model_local_path
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are going to reload the fine tuned model in order to make predictions

# COMMAND ----------

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
from peft import PeftModel, PeftConfig

tokenizer_path = os.path.join(artifact_path, "components", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, 
    padding_side='left'
    )
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model_path = os.path.join(artifact_path, "model")
base_model = AutoModelForCausalLM.from_pretrained(
  base_model_path, 
  quantization_config=bnb_config
  )

peft_ft_path = "./test_ft_run/"
model = PeftModel.from_pretrained(base_model, peft_ft_path)
merged_model = model.merge_and_unload()

base_pipeline = pipeline(
    task="text-generation",
    model=base_model,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    max_length=4096,
    temperature=0.1,
    top_p = 0.95,
    repetition_penalty=1.15
)

ft_pipeline = pipeline(
    task="text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    max_length=4096,
    temperature=0.1,
    top_p = 0.95,
    repetition_penalty=1.15
)

# COMMAND ----------

print_gpu_utilization()

# COMMAND ----------

eval_df = pd.DataFrame(dataset['test'][0:10])
eval_df['base_pred'] = [
  x[0]['generated_text'].split("Activity:")[-1] for x in base_out
  ]
eval_df

# COMMAND ----------

ft_out = ft_pipeline(dataset['test'][0:10]['input'])

# COMMAND ----------

eval_df['ft_pred'] = [
  x[0]['generated_text'].split("Activity:")[-1] for x in ft_out
  ]
eval_df['inputs'] = eval_df['input']
eval_df['response'] = eval_df['label']
eval_df['expected_response'] = eval_df['base_pred']

# COMMAND ----------

from mlflow.metrics.genai import answer_correctness, answer_similarity

# COMMAND ----------

# MAGIC %md
# MAGIC We want a capable model for evaluating our results. We can use the PPT model serving in Databricks to accomplish this.

# COMMAND ----------

correctness_metric = answer_correctness(
  model="endpoints:/databricks-meta-llama-3-1-70b-instruct"
  )

similarity_metric = answer_similarity(
  model="endpoints:/databricks-meta-llama-3-1-70b-instruct"
)

# COMMAND ----------

with mlflow.start_run():
    results = mlflow.evaluate(
        data=eval_df,
        targets="label",
        predictions="base_pred",
        extra_metrics=[
          similarity_metric
          ],
    )

# COMMAND ----------

results.tables['eval_results_table']

# COMMAND ----------

# MAGIC %md
# MAGIC This needs a bit of research - QLoRa models aren't natively supported in serving endpoints and will likely need to be wrapped in a custom pyfunc flavour - homework for next time.

# COMMAND ----------

# mlflow.set_registry_uri("databricks-uc")

# signature = mlflow.models.infer_signature(
#     model_input="What are the three primary colors?",
#     model_output="The three primary colors are red, yellow, and blue.",
# )

# with mlflow.start_run():
#     model_info = mlflow.transformers.log_model(
#         transformers_model=base_pipeline,
#         artifact_path='text_generation',
#         task='text-generation',
#         signature=signature,
#         registered_model_name='shm.work_order.ft_test_pipeline'
#     )

# COMMAND ----------

# reloaded_model = mlflow.transformers.load_model("models:/shm.work_order.ft_test_pipeline/2")
