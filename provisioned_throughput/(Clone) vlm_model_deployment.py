# Databricks notebook source
# for Qwen
# %pip install -U mlflow git+https://github.com/huggingface/transformers.git@8585450 torch torchvision accelerate bitsandbytes

# for Gemma 3
%pip install -U mlflow git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 torch torchvision accelerate bitsandbytes

# for Molmo and others
# %pip install -U mlflow transformers torch torchvision accelerate bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import snapshot_download
import os 

# model_path = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
# model_path = "meta-llama/Llama-3.2-90B-Vision-Instruct"
# model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_path = "allenai/Molmo-7B-D-0924"
model_path = "google/gemma-3-4b-it"
model_cache_path = f"/local_disk0/models_cache/{model_path}"
download_models = False

if download_models:
  os.environ["HF_HOME"] = "/local_disk0/hf"
  os.environ["HF_TOKEN"] = dbutils.secrets.get(
      scope="shj_db_scope", key="hf_secret"
  )

  snapshot_location = snapshot_download(repo_id=model_path, 
                                        local_dir=model_cache_path,
                                        ignore_patterns="*.pth")
else:
  snapshot_location = model_cache_path
  
snapshot_location

# COMMAND ----------

import yaml
import json
import mlflow
import os

model_type = model_path.split("/")[-1].split("-")[0].lower()
print(model_type)
inference_config={
  "model_type": model_type,
  "use_quantization": False,
  "quant_config":{
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_quant_type": "nf4"
  }
}

with open('inference_config.yml', 'w') as f:
    yaml.dump(inference_config, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy code

# COMMAND ----------

from mlflow.models.signature import infer_signature, ModelSignature
import requests
from PIL import Image
import io
import base64
import pandas as pd

def pillow_image_to_base64_string(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

example_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
example_image = Image.open(requests.get(example_image_url, stream=True).raw)
example_image_base64 = pillow_image_to_base64_string(example_image)

input_example = pd.DataFrame().from_records([{"user_prompt": "describe the image and note the objects in the image", "image": example_image_base64}])
params = {"max_new_tokens": 256, "temperature": 0.01, "top_p":0.1}
output_example = pd.DataFrame().from_records([{"output_text": "this is an example output"}])

signature = infer_signature(input_example, output_example, params)
print(signature)

# COMMAND ----------

ds_model_path = os.path.join(os.getcwd(), "vlm_model.py")
config_path = os.path.join(os.getcwd(), "inference_config.yml")

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        "model",
        python_model=ds_model_path,
        model_config=config_path,
        artifacts={"model_path": model_cache_path},
        input_example=input_example,
        signature=signature,
        pip_requirements=["git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3", "torch", "torchvision", "accelerate", "mlflow", "bitsandbytes"]
    )

# COMMAND ----------

model_info.model_uri

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(model_info.model_uri, "uc_sriharsha_jana.test_db.gemma-3b-it-model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and test the model

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import requests
import pandas as pd
from PIL import Image
import mlflow
import io
import os
import base64

def reduce_image_size(img, factor=4):
    width, height = img.size
    new_size = (width // factor, height // factor)
    return img.resize(new_size)

def pillow_image_to_base64_string(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# example_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# example_image = Image.open(requests.get(example_image_url, stream=True).raw)

example_image_url = "/Volumes/uc_sriharsha_jana/test_db/shjdata/test_image.png"
example_image = Image.open(example_image_url).convert("RGB")

# example_image_resized = reduce_image_size(example_image)
example_image_base64 = pillow_image_to_base64_string(example_image)

input_example = pd.DataFrame().from_records([{"user_prompt": "describe the image and note the objects in the image", "image": example_image_base64}])

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# model_uri = 'runs:/ff6e3a64a71f44d9a11974bc05b9e179/model'
model_uri = "models:/uc_sriharsha_jana.test_db.gemma-3b-it-model/1"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

output = loaded_model.predict(input_example, params={"max_new_tokens": 512, "temperature": 0.01, "top_p":0.1})
print(output.iloc[0]["output_text"])

# COMMAND ----------

output = loaded_model.predict(input_example, params={"max_new_tokens": 512, "temperature": 0.01, "top_p":0.1})
print(output.iloc[0]["output_text"])
