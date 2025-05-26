# Databricks notebook source
import pyspark.sql.functions as F
import datasets

# COMMAND ----------

activity = spark.table("shm.work_order.activity")
display(activity.limit(20))

# COMMAND ----------

# Explode the list of activities - this generates 3,132 unique activities
exploded_activity = (
  activity
  .withColumn("activity", F.explode(F.split(activity["activity"], ",")))
  .withColumn("activity", F.regexp_replace("activity", "[\\[\\]'\"']", ""))
  .withColumn("activity", F.lower("activity"))
  .withColumn("activity", F.trim("activity"))
)

display(exploded_activity.select('activity').distinct().limit(50))

exploded_activity.select('activity').distinct().write.mode("overwrite").saveAsTable("shm.work_order.distinct_activities")

# Combine all unique activities into a single string
combined_activity = exploded_activity.agg(
  F.concat_ws(",", F.collect_list("activity")).alias("combined_activities")
  ).collect()[0][0]
combined_activity[0:500]

# COMMAND ----------

# MAGIC %md
# MAGIC Huggingface uses the datasets library to generate a prompt. We can generate a simple text column for supervised fine tuning using this library and then leverage the huggingface library in downstream tasks. We will save the dataset into volumes.

# COMMAND ----------

dataset = datasets.Dataset.from_spark(activity, cache_dir="/local_disk0/")

def combine_columns(example):
    prompt = f"Use the following work order information and create a comma separated list of activites, max three words per activity with a maximum of five activities. Only return the list.\nEquipment: {example['equipment']}\nShort Description: {example['short_description']}\nLong Description: {example['long_description']}\nActivity: "
    return {
      "input": prompt, 
      "label": f"{example['activity']}",
      "text": prompt + example['activity']
      }

dataset = dataset.map(combine_columns)

train_dataset = dataset.filter(lambda x: x['test_set'] == False)
test_dataset = dataset.filter(lambda x: x['test_set'] == True)
split_dataset = split_dataset = datasets.DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

split_dataset['train'][0]

split_dataset.save_to_disk("/Volumes/shm/work_order/datasets/workorder_text")

# COMMAND ----------

# MAGIC %md
# MAGIC Because of GEOS, we should now be able to use PPT models in Canada Central. Here is a quick example of this.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
llm.invoke('test')

# COMMAND ----------

messages = [
    {"role": "system", "content": "You are a maintenance expert. Take the following list of unique tasks. Group them into 50 overarching categories. Return a list only."},
    {"role": "user", "content": combined_activity}
]
categories = llm.invoke(messages)

# COMMAND ----------


