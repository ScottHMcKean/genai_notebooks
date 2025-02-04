# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is a DLT implementation of the Bronze ingestion using SQL DLT

# COMMAND ----------

import dlt
import pyspark.sql.functions as F
import pyspark.sql.types as T

# COMMAND ----------

ingest_path = spark.conf.get(
  "ingest_path",
  "/Volumes/shm/dts/witsml_landing/"
  )

# COMMAND ----------

@dlt.table(
  name="bronze",
  table_properties={"pipelines.reset.allowed": "false"},
  partition_cols=["well_path"]
)
def bronze():
    return (
      spark.readStream
      .format("cloudFiles")
      .option("cloudFiles.format", "xml")
      .option("inferSchema", "true")
      .option("cloudFiles.inferColumnTypes", "true")
      .option("cloudFiles.allowOverwrites", "true")
      .option("recursiveFileLookup", "true")
      .option("cloudFiles.schemaLocation", "dbfs:/FileStore/schemaLocation/")
      .option("rowTag", "WITSMLComposite")
      .option("rescuedDataColumn", "_rescued_data")
      .load(ingest_path)
      .selectExpr("*", "_metadata as source_metadata")
      .withColumn('well_path', F.regexp_replace(F.col("source_metadata.file_path"), "/[^/]+$", ""))
    )

