# Databricks notebook source
import dlt
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

# COMMAND ----------

root_to_remove = spark.conf.get(
  "root_to_remove",
  "/Volumes/shm/dts/witsml_landing/"
  )
  
data_value_loc = spark.conf.get(
  "data_value_loc",
  "wellSet.well.wellboreSet.wellbore.wellLogSet.wellLog.logData.data._VALUE"
  )

timestamp_loc = spark.conf.get(
  "timestamp_loc",
  "wellSet.well.wellboreSet.wellbore.wellLogSet.wellLog.creationDate"
  )


# COMMAND ----------

@dlt.table(
  name="dts_readings",
  table_properties={"pipelines.reset.allowed": "true"},
  partition_cols=["unique_well_id"]
)
def dts_readings():
    window_spec = (Window
        .partitionBy("source_metadata.file_path")
        .orderBy(F.desc("source_metadata.file_modification_time"))
    )

    return (
        dlt.read("bronze")
        .withColumn("row_num", F.row_number().over(window_spec))
        .filter("row_num == 1")
        .drop("row_num")
        .withColumn("unique_well_id", F.regexp_replace(F.col("well_path"), root_to_remove, ""))
        .withColumn("unique_well_id", F.regexp_replace(F.col("unique_well_id"), "Halliburton/", ""))
        .withColumn("unique_well_id", F.regexp_replace(F.col("unique_well_id"), "Petrospec/", ""))
        .select(
            "unique_well_id",
            F.col(timestamp_loc).alias("timestamp"),
            F.explode(F.col(data_value_loc)).alias("data_value")
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

@dlt.table(
  name="dts_metadata",
  table_properties={"pipelines.reset.allowed": "true"},
  partition_cols=["unique_well_id"]
)
def dts_metadata():
    window_spec = (Window
        .partitionBy("source_metadata.file_path")
        .orderBy(F.desc("source_metadata.file_modification_time"))
    )

    return (
        flatten_df(dlt.read("bronze"))
        .withColumn("unique_well_id", F.regexp_replace(F.col("well_path"), root_to_remove, ""))
        .withColumn("unique_well_id", F.regexp_replace(F.col("unique_well_id"), "Halliburton/", ""))
        .withColumn("unique_well_id", F.regexp_replace(F.col("unique_well_id"), "Petrospec/", ""))
        .drop('wellSet_well_wellboreSet_wellbore_wellLogSet_wellLog_logData_data')
        .drop('_version','_xmlns:witsml', '_xmlns:xsi', '_xsi:noNamespaceSchemaLocation', '_rescued_data')
        )
