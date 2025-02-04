# Databricks notebook source
# MAGIC %md
# MAGIC # Named Entity Recognition
# MAGIC This notebook provides an example of using MLFlow and Mosaic AI for named entity recognition (NER). This notebook sets up the basic framework for evaluation and prompt engineering, but NER benefits a lot from fine-tuning, so we should follow with that.
# MAGIC
# MAGIC This notebook focuses on few-shot prompting for improving NER. Few-shot prompting is discussed here: https://arxiv.org/pdf/2407.08035v1. But fine-tuning is nearly essential for good NER performance - see the [FabNER paper](https://par.nsf.gov/servlets/purl/10290810)

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.28.0 databricks-agents mlflow==2.15.0 databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-sql-connector==3.3.0 langchain_openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datasets import load_dataset
import pandas as pd
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ##FabNER Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC In this demo, we are going to use the [FabNER dataet](https://par.nsf.gov/servlets/purl/10290810). It uses a BIO framework for encoding entities and has over 13,000 sentences or 350,000 tokens for entity recognition. In the future it would be great to do an end-to-end example with the [Utah Forge drilling comment dataset](https://catalog.data.gov/dataset?publisher=Energy%20and%20Geoscience%20Institute%20at%20the%20University%20of%20Utah).

# COMMAND ----------

fabner_ds = load_dataset("DFKI-SLT/fabner", "fabner_bio")

# COMMAND ----------

# MAGIC %md
# MAGIC These are the BIO classes we use to classify the text. These are applied to each token, so the tokenizer matters when applying these models. These BIO classes would be specific to your buisness, but the approach should be the same.

# COMMAND ----------

bio_classes = {"O": 0, "B-MATE": 1, "I-MATE": 2, "B-MANP": 3, "I-MANP": 4, "B-MACEQ": 5, "I-MACEQ": 6, "B-APPL": 7, "I-APPL": 8, "B-FEAT": 9, "I-FEAT": 10, "B-PRO": 11, "I-PRO": 12, "B-CHAR": 13, "I-CHAR": 14, "B-PARA": 15, "I-PARA": 16, "B-ENAT": 17, "I-ENAT": 18, "B-CONPRI": 19, "I-CONPRI": 20, "B-MANS": 21, "I-MANS": 22, "B-BIOP": 23, "I-BIOP": 24}

train = pd.DataFrame(fabner_ds['train'])
train['input'] = train['tokens'].apply(lambda x: ' '.join(x))
train['ner_class'] = train['ner_tags'].apply(
  lambda x: [list(bio_classes.keys())[list(bio_classes.values()).index(tag)] for tag in x]
  )
display(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Engineering
# MAGIC The following section sets up a prompt. It would be worthwhile to either version control the prompt or store it in a volume for future referencing. Also note that we are using a pay-per-token endpoint, but you'll likely want to fine-tune a model and serve that instead (let's get ot that later). Without fine-tuning, you are stuck with few-shot prompting. It works, but I expect the results do be a little lackluster. Fine-tuning is almost an essential for NER.

# COMMAND ----------

fm_chain_config = {
    "fm_serving_endpoint_name": "databricks-mixtral-8x7b-instruct",
    "fm_prompt_template": """
    You are a data labeler labeling data to be used in token/Named Entity Recognition. Your task is to identify categorical classifications using the BIO format, where the tokens are the words in the sentence and the tags are the classifications. The classifications should begin with Beginning (B), Inside (I) of an entity, or Outside (O) of any entity. 
    
    Use the following categories:
    MATE: Material
    MANP: Manufacturing Process
    APPL: Application
    ENGF: Features
    MECHP: Mechanical Properties
    PROC: Characterization
    PROP: Parameters
    MACEQ: Machine/Equipment
    ENAT: Enabling Technology
    CONPRI: Concept/Principles
    BIOP: BioMedical
    MANS: Manufacturing Standards

    Follow these guidelines:
    - Do not add any commentary or repeat the instructions.
    - Extract the categories list above - if the token is not classified as as category, return O.
    - Place each token from the input in a Python list format. Ensure the tokens are enclosed in square brackets and separated by commas.
    - Next, Place the extracted names in a Python list format. Ensure the names are enclosed in square brackets and separated by commas.
    - Do not add any text before or after the two Python lists.
    - Respond without using special or escape characters other than commas (,) and dashes (-)
    - Respond without using newline or tab characters

    Here are some examples:
    <example>
    Revealed the location-specific flow patterns and quantified the speeds of various types of flow .

    Tokens:['Revealed','the','location-specific','flow','patterns','and','quantified','the','speeds','of','various','types','of','flow','.']
    Categories: ['O','O','O','I-FEAT','I-FEAT','O','O','O','O','O','O','O','O','O','O'])

    <example>
    In this work , X-ray tomography was employed to provide insights into pore closure efficiency by HIP for an intentional and artificially-induced cavity as well as for a range of typical process-induced pores ( lack of fusion , keyhole , contour pores , etc .

    Tokens:['In','this','work',',','X-ray','tomography','was','employed','to','provide','insights','into','pore','closure','efficiency','by','HIP','for','an','intentional','and','artificially-induced','cavity','as','well','as','for','a','range','of','typical','process-induced','pores','(','lack','of','fusion',',','keyhole',',','contour','pores',',','etc','.']
    Categories:['O','O','O','O','B-CHAR','I-CHAR','O','O','O','O','O','O','B-PRO','O','O','O','B-MANP','O','O','O','O','O','O','B-MATE','O','B-MATE','O','O','B-PARA','O','O','O','B-PRO','O','O','O','B-CONPRI','O','O','O','B-FEAT','O','O','O','O']

    <example>
    in coupon samples of Ti6Al4V .
    
    Answer: BHP: 58 MPa

    Tokens:['in', 'coupon', 'samples', 'of', 'Ti6Al4V', '.']
    Categories:['O','O','O','B-CONPRI','O','B-MATE','O']

    Now classify the following text:
    {user_query}

    """,
    "temperature": 0.01,
    "max_tokens": 1000
}

# COMMAND ----------

with open('./fm_prompt_config.yaml', 'w') as f:
    yaml.dump(fm_chain_config, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure a Custom Model
# MAGIC
# MAGIC This is an example of deploying a custom model and introduces a couple things:
# MAGIC
# MAGIC - Use MLFlow for model tracing & evaluation
# MAGIC - Load a model configuration from CICD
# MAGIC - Setup a custom `PromptTemplate`
# MAGIC - Use the `ChatDatabricks` inferface to hit an endpoint
# MAGIC - Create a `chain` and invoke it
# MAGIC
# MAGIC Things are moving extremely fast in the generative AI space and all the orchestration packages (DsPy, LangChain, LlamaIndex, etc.) are changing rapidly.
# MAGIC
# MAGIC One key thing that causes friction is the input and output formats of the model. To make your life easier, take a look at the [agent deployment instructions](https://docs.databricks.com/en/generative-ai/create-log-agent.html#example-notebooks) and make sure you are using the OpenAI chat completion schema for the input and a single string for the output. In terms of NER this doesn't make a lot of sense, but it is what the industry has standardized on so far, so don't fight it =).

# COMMAND ----------

import mlflow
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]

# Enable MLflow tracing & logging
mlflow.langchain.autolog()

# Load the chain configuration
model_config = mlflow.models.ModelConfig(development_config=fm_chain_config)

# Define a prompt template
prompt = PromptTemplate(
    template=model_config.get("fm_prompt_template"),
    input_variables=["user_query"]
)

# Setup a foundation model
model = ChatDatabricks(
    endpoint=model_config.get("fm_serving_endpoint_name"),
    extra_params={
      "temperature": model_config.get("temperature"), 
      "max_tokens": model_config.get("max_tokens")
    }
)

# define a custom output parsing function
def split_string_output(output):
    """
    Split the string output into tokens and categories.
    """
    return {
        'tokens': re.search(r'Tokens:(.*?)\n', output, re.DOTALL).group(1),
        'categories': re.search(r'Categories:(.*)', output, re.DOTALL).group(1)
    }

# Define the basic chain
chain = (
    # This sections sets up the input extraction of the latest message
    # It will be useful if we want to incorporate RAG with NER examples
    {
        "user_query": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages")
        | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | StrOutputParser()
)

#Let's try our prompt:
question = {
    "messages": [
               {
            "role": "user",
            "content": 'in coupon samples of Ti6Al4V .',
        },
    ]
}
result = chain.invoke(question)

# COMMAND ----------

result

# COMMAND ----------

# MAGIC %md
# MAGIC This chunk batches a dataframe and runs multiple queries. One optimization you may want to do immediately is to stop MLFlow logging as it creates a lot of overhead.

# COMMAND ----------

def extract_entities(df):
  predictions = (chain
    .with_retry(stop_after_attempt=2)
    .batch(
      [{"messages": [{"role": "user", "content": content}]} for content in sample["input"].tolist()], 
      config={"max_concurrency": 128})
  )
  return predictions

# COMMAND ----------

# Let's test our batch predictions
sample = train.iloc[0:5].copy()
predictions = extract_entities(sample)
sample_w_pred = pd.concat([sample, pd.DataFrame(predictions)], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the Model
# MAGIC
# MAGIC We want to deploy this foundation model with our custom prompt template. In order to do that, we need to generate an [MLFlow signature](https://mlflow.org/docs/latest/model/signatures.html) that tells users how to infer with the model, specifically the expected input and output. We then register the model in Unity Catalog as a LangChain model. Because the models don't have binary files (e.g. scikit-learn pickle files), we need to provide a path to the chain (`fm_chain.py`) and the configuration (`fm_chain_config.yaml`). This also ensures that the model is a `pyfunc` model, which is eseential for both serving and evaluation

# COMMAND ----------

from mlflow.models import infer_signature
import os

input_example = {
    "messages": [
               {
            "role": "user",
            "content": 'in coupon samples of Ti6Al4V .',
        },
    ]
}

output_example = "\nTokens:['in', 'coupon', 'samples', 'of', 'Ti6Al4V', '.']\nCategories:['O','O','O','B-CONPRI','O','B-MATE','O']"

fm_model_name = "ner_few_shot"
fm_model_path = f"scottmckean.ner.{fm_model_name}"

signature = infer_signature(input_example, output_example)

# Log the model to MLflow
with mlflow.start_run(run_name=fm_model_name):
  logged_chain_info = mlflow.langchain.log_model(
          lc_model=os.path.join(os.getcwd(), 'fm_prompt.py'),
          model_config=fm_chain_config, 
          artifact_path="chain",
          signature=signature,
          example_no_conversion=True,
          registered_model_name=fm_model_path
      )

# COMMAND ----------

from databricks import agents
## Deploy the Custom Model as a REST Endpoint ##
deployment_info = agents.deploy(
  fm_model_path,
  model_version=13,
  scale_to_zero=True
  )

# COMMAND ----------

deployment_info

# COMMAND ----------

instructions_to_reviewer = f"""
This is a named entity recognition model. Please test it out and record the proper response. The goal is to identify categorical classifications using the BIO format, where the tokens are the words in the sentence and the tags are the classifications. The classifications should begin with Beginning (B), Inside (I) of an entity, or Outside (O) of any entity.
    
Use the following categories:
MATE: Material
MANP: Manufacturing Process
APPL: Application
ENGF: Features
MECHP: Mechanical Properties
PROC: Characterization
PROP: Parameters
MACEQ: Machine/Equipment
ENAT: Enabling Technology
CONPRI: Concept/Principles
BIOP: BioMedical
MANS: Manufacturing Standards
"""

# Add the user-facing instructions to the Review App
agents.set_review_instructions(fm_model_path, instructions_to_reviewer)

# COMMAND ----------

# MAGIC %md
# MAGIC So far, we created and tested a custome NER model, registered the model with Unity, deployed it as a serving endpoint, and created a review app to test that deploymment. The last step is to take our NER dataset with manual annotations (these can be collected via the review app as well!) and evaluate our first version of the model for accuracy. In NER we can get some quantitative metrics because we are doing multiple classifications for each token. This is worth a bit of thought on HOW we want to do it, but we can iterate together on it prior to fine tuning (because fine tuning needs a loss function).
# MAGIC
# MAGIC Worth noting you don't NEED to deploy the agent to run evaluation, you just need to provide a logged model URI (logged_chain_info.model_uri)

# COMMAND ----------

sample['messages'] = [[{"role": "user", "content": content}] for content in sample["input"].tolist()]

# COMMAND ----------

## Evaluate the Model Based on User Feedback With MLFLow##
with mlflow.start_run(run_id=logged_chain_info.run_id):
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        model=logged_chain_info.model_uri,
        data=sample,
        model_type="question-answering",
        evaluators="default",
        predictions="result",
        evaluator_config={
            "col_mapping": {
                "inputs": "messages",
            }
        },
    )

# COMMAND ----------


