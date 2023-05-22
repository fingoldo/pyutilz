# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .python import ensure_installed
ensure_installed("google-cloud boto3")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

# --------------------------------------------------------------------------------------------------------------
# Inits
# --------------------------------------------------------------------------------------------------------------

s3 = None

# --------------------------------------------------------------------------------------------------------------
# Google cloud storage
# --------------------------------------------------------------------------------------------------------------


def gcp_storage_upload_blob(bucket_name, source_file_name, destination_blob_name, acc_file):
    from google.cloud import storage

    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client.from_service_account_json(acc_file)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def gcp_storage_download_blob(bucket_name, source_blob_name, destination_file_name, acc_file):
    from google.cloud import storage

    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.from_service_account_json(acc_file)

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


# --------------------------------------------------------------------------------------------------------------
# Amazon cloud storage
# --------------------------------------------------------------------------------------------------------------

def connect_to_s3(file: str = "settings.ini"):
    global s3
    import boto3
    from .string import read_config_file

    if read_config_file(file=file, object=globals(), section="S3", variables="aws_access_key_id,aws_secret_access_key"):
        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        s3 = boto_session.resource("s3")
        return s3


def s3_file_exists(key: str, bucket: str) -> bool:
    try:
        s3.meta.client.head_object(Bucket=bucket, Key=key)
    except:
        return False
    else:
        return True
        
def get_from_s3_or_cache(local_object_path:str,s3_object_path:str,temp_dir:str):
    """
        Gets an object from s3 or local filesystem (including zipped version).
        If not found even on s3, reports and starts waiting cycle until expected object is uploaded.
    """
    while not exists(local_object_path):
        bDownload=True
        if s3_object_path.endswith('.zip'):
            if exists(local_object_path+'.zip'):
                logger.info(f'Zipped version of {s3_object_path} found in local FS')
                bDownload=False
        
        if bDownload:
            # If file exists in our s3        
            if cloud.s3_file_exists(s3_object_path,S3_BUCKET_NAME):
                #Load model from our s3
                try:
                    if s3_object_path.endswith('.zip'):
                        s3.meta.client.download_file(Bucket=S3_BUCKET_NAME,Key=s3_object_path,Filename=local_object_path+'.zip')                
                    else:
                        s3.meta.client.download_file(Bucket=S3_BUCKET_NAME,Key=s3_object_path,Filename=local_object_path)                                    
                except Exception as e:
                    logger.error(f"Error while downloading model {s3_object_path} from bucket {S3_BUCKET_NAME}: {e}")
                else:
                    logger.info(f"Downloaded model {s3_object_path} from bucket {S3_BUCKET_NAME}")
            else:
                logger.warning(f"Model {s3_object_path} not found in bucket {S3_BUCKET_NAME}")
                sleep(10)
        else:
            if s3_object_path.endswith('.zip'):
                if exists(local_object_path+'.zip'):        
                    # unpack it
                    import shutil
                    try:
                        shutil.unpack_archive(local_object_path+'.zip', temp_dir)
                    except Exception as e:
                        logger.error(f"Error while unpacking model from bucket {local_object_path+'.zip'}: {e}")
                    else:
                        logger.info(f"Unzipped model from archive {local_object_path+'.zip'}")
                        os.remove(local_object_path+'.zip')