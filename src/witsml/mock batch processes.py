# Databricks notebook source
# MAGIC %md
# MAGIC This notebook mocks a daily batch load by moving subfolders from a volume into a 'landing' volume that kicks off a DLT pipeline. It also mocks modifying a file and landing it into the 'landing' zone.

# COMMAND ----------

raw_volume = '/Volumes/shm/dts/witsml'
landing_volume = '/Volumes/shm/dts/witsml_landing'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS shm.dts.witsml_landing

# COMMAND ----------

# MAGIC %md
# MAGIC This chunk migrates our on-prem storage zone (raw_volume) to the WITSML landing zone (landing_volume).

# COMMAND ----------

import shutil
import os

def move_directories(
    src_volume: str, 
    dest_volume: str,
    ):
    for root, dirs, files in os.walk(src_volume):
            dest_dir = os.path.join(dest_volume, os.path.relpath(root, src_volume))
            os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
            shutil.move(root, dest_dir)

def move_directories_with_day_folder(
    src_volume: str, 
    dest_volume: str, 
    day:str = '01'
    ):
    for root, dirs, files in os.walk(src_volume):
        if day in dirs:
            src_dir = os.path.join(root, day)
            dest_dir = os.path.join(dest_volume, os.path.relpath(root, src_volume), day)
            os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
            shutil.move(src_dir, dest_dir)

# move_directories_with_day_folder(raw_volume, landing_volume, '03')

# COMMAND ----------

move_directories(raw_volume, landing_volume)

# COMMAND ----------

# MAGIC %md
# MAGIC Randomly change five files and modify the names to C2X. We can then query the number of changed rows to validate the table change.

# COMMAND ----------

from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType
import xml.etree.ElementTree as ET

def xml_to_dict(element):
    """Convert an XML element and its children to a dictionary."""
    if len(element) == 0:
        return element.text
    return {child.tag: xml_to_dict(child) for child in element}

result = spark.sql("""
  SELECT source_metadata.file_path 
  FROM shm.dts.bronze 
  ORDER BY RAND() LIMIT 5
  """)

for row in result.collect():
    file_to_modify_path = row['file_path']
    print(file_to_modify_path)
    
    # Load the XML file from Unity Catalog
    tree = ET.parse(file_to_modify_path)
    root = tree.getroot()

    orig_name = xml_to_dict(root)['wellSet']['well']['name']

    for well in root.findall('.//wellSet/well'):
        name_element = well.find('name')
        if name_element is not None:
            name_element.text = 'C2X'

    new_name = xml_to_dict(root)['wellSet']['well']['name']

    print(f"Changed name from {orig_name} to {new_name}")
    
    # Writeback the modified file
    tree.write(file_to_modify_path)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import StartUpdate

w = WorkspaceClient()

name = "witsml_ingest"

found_pipelines = (p for p in w.pipelines.list_pipelines() if p.name == name)
pipeline = next(found_pipelines, None)

if pipeline:
    # Trigger the pipeline update
    update = w.pipelines.start_update(
        pipeline_id=pipeline.pipeline_id,
        full_refresh=False  # Set to True if you want a full refresh
    )
    print(f"Pipeline update started. Update ID: {update.update_id}")
else:
    print(f"Pipeline '{name}' not found.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM (
# MAGIC   SELECT *, ROW_NUMBER() OVER (PARTITION BY source_metadata.file_path ORDER BY source_metadata.file_modification_time ASC) as row_num
# MAGIC   FROM shm.dts.bronze
# MAGIC   WHERE source_metadata.file_path IN (
# MAGIC     SELECT source_metadata.file_path 
# MAGIC     FROM shm.dts.bronze 
# MAGIC     WHERE wellSet.well.name = 'C2X'
# MAGIC   )
# MAGIC ) subquery
# MAGIC -- WHERE row_num = 1
# MAGIC ORDER BY source_metadata.file_path

# COMMAND ----------


