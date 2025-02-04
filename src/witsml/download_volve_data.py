# Databricks notebook source
# MAGIC %pip install azure-storage-blob

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os
from pathlib import Path

# Blob storage details
account_url = "https://datavillagesa.blob.core.windows.net"
container_name = "volve"
sas_token = dbutils.secrets.get(scope="shm", key="volve-sas")

# Unity Catalog volume details
target_volume = "/Volumes/my_catalog/my_schema/my_volume/volve_data"

# Create BlobServiceClient
blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)

# Get container client
container_client = blob_service_client.get_container_client(container_name)

# Function to download a blob
def download_blob(blob_client, target_path):
    with open(target_path, "wb") as file:
        data = blob_client.download_blob()
        file.write(data.readall())

volume_path = f"/Volumes/shm/drilling/statoil"
blob_subfolder = "WITSML Realtime drilling data/"
blobs = container_client.list_blobs(name_starts_with=blob_subfolder)

for blob in blobs:
    print(f"Downloading {blob.name}")
    blob_client = container_client.get_blob_client(blob.name)
    
    # Create the directory structure if it doesn't exist
    target_dir = os.path.dirname(os.path.join(volume_path, blob.name.replace(blob_subfolder, "")))
    Path(target_dir).mkdir(exist_ok=True, parents=True)
    
    # Download to a temporary local file
    local_path = f"/tmp/{blob.name}"
    Path(os.path.dirname(local_path)).mkdir(exist_ok=True,parents=True)
    download_blob(blob_client, local_path)
    
    # Move the file to the Unity Catalog volume
    dbutils.fs.mv(f"file:{local_path}", os.path.join(volume_path, blob.name.replace(blob_subfolder, "")))    

print("All files downloaded successfully.")
