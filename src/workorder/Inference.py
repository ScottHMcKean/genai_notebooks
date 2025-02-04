# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline Evaluation
# MAGIC This notebook uses the test set to evaluate the baseline model performance at classifying the activities using few shot prompting. Few shot prompting is a nice step towards RAG. We will use a local model with a V100 GPU.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pyspark.sql.functions as F
import torch
import gc
import pandas as pd

# COMMAND ----------

from utils import print_gpu_utilization
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print_gpu_utilization()

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
# MAGIC I was having some issues when loading a huggingface pipeline into Langchain - it seems to overload the GPU. 

# COMMAND ----------

import os
from transformers import AutoTokenizer
tokenizer_path = os.path.join(artifact_path, "components", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

from datasets import load_from_disk
dataset = load_from_disk('/Volumes/shm/work_order/datasets/workorder_text/')

# COMMAND ----------

messages = [
    {"role": "user", "content": dataset['train'][0]['input']},
    {"role": "assistant", "content": dataset['train'][0]['label']},
 ]
tokenized_chat = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=False, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    add_system_prompt=False
    )
print(tokenizer.decode(tokenized_chat[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC Setup a pipeline for predictions. The most critical factors to avoid blowing up the GPU are the max length and the batch size. In testing, batch size did very bad things to the GPU so removed it. Setting MAX_LENGTH was important for performance. The maximum length of the dataset was ~1,000 characters, so I quadrupled that to be safe. That should allow plenty of context while keeping memory requirements down.

# COMMAND ----------

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
from peft import PeftModel, PeftConfig

MAX_LENGTH = 4096
TEMPERATURE = 0.1
TOP_P = 0.95
REPETITION_PENALTY = 1.15

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

base_pipeline = pipeline(
    task="text-generation",
    model=base_model,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
)

# COMMAND ----------

print_gpu_utilization()

# COMMAND ----------

from datasets import concatenate_datasets, DatasetDict
combined_dataset = concatenate_datasets([dataset['train'], dataset['test']])

# COMMAND ----------

sampled_entries = combined_dataset.shuffle(seed=42).select(range(100))
sampled_df = pd.DataFrame(sampled_entries)

# COMMAND ----------

base_out = base_pipeline(sampled_entries['input'])
sampled_df['base_pred'] = [
  x[0]['generated_text'].split("Activity:")[-1] for x in base_out
  ]

# COMMAND ----------

base_out[0:3]

# COMMAND ----------

print_gpu_utilization()

# COMMAND ----------

base_model

# COMMAND ----------

base_model_path = os.path.join(artifact_path, "model")
base_model = AutoModelForCausalLM.from_pretrained(
  base_model_path, 
  quantization_config=bnb_config
  )

peft_ft_path = "./test_ft_run/"
ft_a_merged = PeftModel.from_pretrained(base_model, peft_ft_path)
merged_model_a = ft_a_merged.merge_and_unload()

ft_a_pipeline = pipeline(
    task="text-generation",
    model=merged_model_a,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY
)


# COMMAND ----------

ft_a_out = ft_a_pipeline(sampled_entries['input'])
sampled_df['ft_a_out'] = [
  x[0]['generated_text'].split("Activity:")[-1] for x in ft_a_out
  ]

# COMMAND ----------

ft_a_out[0:3]

# COMMAND ----------

print_gpu_utilization()

# COMMAND ----------

base_model_path = os.path.join(artifact_path, "model")
base_model = AutoModelForCausalLM.from_pretrained(
  base_model_path, 
  quantization_config=bnb_config
  )

peft_ft_path = "./test_ft_run_2/"
ft_b_merged = PeftModel.from_pretrained(base_model, peft_ft_path)
merged_model_b = ft_b_merged.merge_and_unload()

ft_b_pipeline = pipeline(
    task="text-generation",
    model=merged_model_b,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY
)

# COMMAND ----------

ft_b_out = ft_b_pipeline(sampled_entries['input'])
sampled_df['ft_b_out'] = [
  x[0]['generated_text'].split("Activity:")[-1] for x in ft_b_out
  ]

# COMMAND ----------

display(sampled_df)
