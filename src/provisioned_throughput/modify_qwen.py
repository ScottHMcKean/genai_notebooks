# Databricks notebook source
# MAGIC %md
# MAGIC ## Modify Qwen
# MAGIC
# MAGIC This modifies a Qwen 2.5 model so that it is consistent with the Llama architecture.

# COMMAND ----------

# MAGIC %pip install -U mlflow transformers==4.44.2 torch==2.3.0 accelerate==0.29.0
# MAGIC %restart_python

# COMMAND ----------

import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import huggingface_hub
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    shard_checkpoint,
)

# The model to download from Hugging Face
dbutils.widgets.text("model_name", "Qwen/Qwen2.5-14B-Instruct")

# Root path where models will be stored
dbutils.widgets.text("dbfs_root_target", "/dbfs/models")

# COMMAND ----------

# Define a function to convert and save Qwen weights


def save_weight(input_dir: str, output_dir: str, shard_size: str = "2GB") -> None:
    """
    Copies a Qwen model in the input directory to a Llama-compatible version of it in the output directory.
    Injects zeroed-out bias vectors in attention layers where needed in order to make it compatible with the Llama
    architecture.  Also updates configuration as needed.
    """
    # Load Qwen state dict from .safetensors files
    qwen_state_dict = OrderedDict()
    for filepath in os.listdir(input_dir):
        if filepath.endswith(".safetensors"):
            full_path = os.path.join(input_dir, filepath)
            with safe_open(full_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    qwen_state_dict[key] = sf.get_tensor(key)

    # Copy tensors and inject bias where needed to match Llama
    llama_state_dict = OrderedDict()
    for key, value in qwen_state_dict.items():
        llama_state_dict[key] = value
        # Qwen omits bias on attn.o_proj; Llama expects it
        if "attn.o_proj.weight" in key:
            # Each attn.o_proj.weight needs an associated bias in order to be
            # compatible with the Llama architecture. Since Qwen doesn't use this we
            # insert zeroed out vectors.
            bias_key = key.replace("attn.o_proj.weight", "attn.o_proj.bias")
            llama_state_dict[bias_key] = torch.zeros_like(value[:, 0]).squeeze()

    # Save weights using safetensors
    shards, index = shard_checkpoint(llama_state_dict, max_shard_size=shard_size, weights_name=SAFE_WEIGHTS_NAME)
    for shard_file, shard_data in shards.items():
        save_path = os.path.join(output_dir, shard_file)
        save_file(shard_data, save_path, metadata={"format": "pt"})

    if index is not None:
        with open(os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)


# COMMAND ----------

# Define a function to update the Qwen config and tokenizer files


def save_configs(input_dir: str, output_dir: str) -> None:
    """
    Copies Qwen config and tokenizer files to output_dir, removing Qwen-specific fields
    and making them compatible with the Llama architecture.
    """
    config_name = "config.json"

    # Load Qwen config.json
    with open(os.path.join(input_dir, config_name), encoding="utf-8") as f:
        qwen_config_dict = json.load(f)

    # Modify the Qwen config to look like a Llama model
    llama_config_dict = {**qwen_config_dict}
    llama_config_dict["architectures"] = ["LlamaForCausalLM"]  # now it's Llama 8-)
    llama_config_dict["model_type"] = "llama"
    llama_config_dict["attention_bias"] = True  # Llama-specific
    llama_config_dict["mlp_bias"] = False
    llama_config_dict["pretraining_tp"] = 0

    # Remove Qwen-specific fields related to sliding window
    for del_key in ["sliding_window", "use_sliding_window", "max_window_layers"]:
        if del_key in llama_config_dict:
            del llama_config_dict[del_key]

    # Write updated config to the new directory
    with open(os.path.join(output_dir, config_name), "w", encoding="utf-8") as f:
        json.dump(llama_config_dict, f, indent=2)

    # Copy other relevant files (tokenizer, merges, vocab, and so on)
    additional_files = [
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for fname in additional_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, dst)


# COMMAND ----------

# Master Function to Orchestrate the Qwen→Llama Conversion


def llamafy_qwen(input_dir: str, output_dir: str) -> None:
    """
    Converts Qwen2.5 into a Llama-like architecture by rewriting weights and configs.
    After this step, the resulting folder can be treated as if it's a Llama model.
    """
    os.makedirs(output_dir, exist_ok=False)

    # Rewrite Qwen weights to add missing biases
    save_weight(input_dir, output_dir)

    # Update config to make it a Llama model and copy other files
    save_configs(input_dir, output_dir)

    print(f"Successfully converted Qwen from '{input_dir}' to Llama format at '{output_dir}'.")


# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
dbfs_root_target = dbutils.widgets.get("dbfs_root_target")

assert model_name
assert dbfs_root_target

target_dbfs_path = os.path.join(dbfs_root_target, model_name)
target_dbfs_modified_path = f"{target_dbfs_path}-Llama"

if not os.path.exists(Path(target_dbfs_path).parent):
    os.makedirs(Path(target_dbfs_path).parent)

if not os.path.exists(target_dbfs_path):
    print(f"Downloading to {target_dbfs_path}")
    huggingface_hub.snapshot_download(model_name, local_dir=target_dbfs_path)
else:
    print(f"Already exists: {target_dbfs_path}")

# COMMAND ----------

if not os.path.exists(target_dbfs_modified_path):
    print(f"Modifying Qwen model and writing to {target_dbfs_modified_path}")
    llamafy_qwen(target_dbfs_path, target_dbfs_modified_path)
else:
    print(f"Already exists: {target_dbfs_modified_path}")
