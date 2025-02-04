# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is a mock for moving Bronze to Silver, creating metadata and time series tables. It is meant to prototype the Silver DLT notebook for the jobs.

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC Worth noting that the Bronze table doesn't really do much other than ingest one file per row. We expect duplicates if the files are overwritten. We use a window function to select the most recent file load - but this causes us to switch from a streaming table with exactly once guarantee to a materialized view. You may want to shift silver into streaming tables by removing this window function, and then rely on Gold for materialized views while doing aggregations.

# COMMAND ----------

bronze = spark.table("shm.dts.bronze")
display(bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC This is a quick example of using splits and explodes to extra time series data from the sensors along with a unique_well_id - you will have a better idea of your requirements here.

# COMMAND ----------

window_spec = (Window
    .partitionBy("source_metadata.file_path")
    .orderBy(F.desc("source_metadata.file_modification_time"))
)

ROOT = "/Volumes/shm/dts/witsml_landing/"
DATA_VALUE_LOC = "wellSet.well.wellboreSet.wellbore.wellLogSet.wellLog.logData.data._VALUE"
TIMESTAMP_LOC = "wellSet.well.wellboreSet.wellbore.wellLogSet.wellLog.creationDate"

dts_readings = (
    bronze
    .withColumn("row_num", F.row_number().over(window_spec))
    .filter("row_num == 1")
    .drop("row_num")
    .withColumn("unique_well_id", F.regexp_replace(F.col("well_path"), ROOT, ""))
     .withColumn("unique_well_id", F.regexp_replace(F.col("unique_well_id"), "Halliburton/", ""))
    .withColumn("unique_well_id", F.regexp_replace(F.col("unique_well_id"), "Petrospec/", ""))
    .select(
        "unique_well_id",
        F.col(TIMESTAMP_LOC).alias("timestamp"),
        F.explode(F.col(DATA_VALUE_LOC)).alias("data_value")
    )
    .select(
        "unique_well_id",
        "timestamp",
        F.split(F.col("data_value"), ",").alias("data_values")
    ).select(
        "unique_well_id",
        "timestamp",
        F.col("data_values").getItem(0).alias("depth"),
        F.col("data_values").getItem(1).alias("temp")
    )
)

display(dts_readings)

# COMMAND ----------

# MAGIC %md
# MAGIC Here is a quick example of a recursive function to flatten nested json structures prior to a) typing the columns, b) dropping data, or c) selecting relevant columns. This avoid hard coding, but could easily result in duplicate columns if the WITSML schema changes. Would suggest discussing schema enforcement here instead of bronze during a next session.

# COMMAND ----------

def flatten_df(df):
  flat_cols = [c[0] for c in df.dtypes if c[1][:6] != 'struct']
  nested_cols = [c[0] for c in df.dtypes if c[1][:6] == 'struct']
  
  # Select flat columns and flatten nested structures
  flat_df = df.select(flat_cols + [
      F.col(nc+'.'+c).alias(nc+'_'+c)
      for nc in nested_cols
      for c in df.select(nc+'.*').columns
  ])
  # Recursively process nested structures
  for nc in nested_cols:
      flat_df = flatten_df(flat_df.drop(nc))
  return flat_df

metadata_out = (
  flatten_df(bronze)
  .drop('wellSet_well_wellboreSet_wellbore_wellLogSet_wellLog_logData_data')
  .drop('_version','_xmlns:witsml', '_xmlns:xsi', '_xsi:noNamespaceSchemaLocation', '_rescued_data')
)

display(metadata_out)
