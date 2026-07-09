"""Helpers for uploading/downloading files to/from Google Cloud Storage and Amazon S3."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import os
from os.path import exists
from time import sleep
from typing import Any as _Any

# --------------------------------------------------------------------------------------------------------------
# Inits
# --------------------------------------------------------------------------------------------------------------

s3: _Any = None
S3_BUCKET_NAME = None  # To be configured by user

# --------------------------------------------------------------------------------------------------------------
# Google cloud storage
# --------------------------------------------------------------------------------------------------------------


def gcp_storage_upload_blob(bucket_name, source_file_name, destination_blob_name, acc_file):
    """Upload a local file to a GCS bucket using a service-account JSON key file."""
    from google.cloud import storage

    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client.from_service_account_json(acc_file)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info("File %s uploaded to %s.", source_file_name, destination_blob_name)


def gcp_storage_download_blob(bucket_name, source_blob_name, destination_file_name, acc_file):
    """Download a blob from a GCS bucket to a local file using a service-account JSON key file."""
    from google.cloud import storage

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

    logger.info("Blob %s downloaded to %s.", source_blob_name, destination_file_name)


# --------------------------------------------------------------------------------------------------------------
# Amazon cloud storage
# --------------------------------------------------------------------------------------------------------------

def connect_to_s3(file: str = "settings.ini"):
    """Read AWS credentials from the ``[S3]`` section of a config file and open an S3 resource.

    Sets the module-level ``s3`` global to the created boto3 S3 resource and returns it.
    Returns None (leaving ``s3`` unset) if credentials could not be read from the file.
    """
    global s3
    import boto3
    from pyutilz.text.strings import read_config_file

    aws_access_key_id = None
    aws_secret_access_key = None
    if read_config_file(file=file, object=globals(), section="S3", variables="aws_access_key_id,aws_secret_access_key"):
        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        s3 = boto_session.resource("s3")
        return s3


def s3_file_exists(key: str, bucket: str) -> bool:
    """Check whether an object with the given key exists in an S3 bucket (via a HEAD request)."""
    try:
        s3.meta.client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    else:
        return True

def get_from_s3_or_cache(local_object_path:str,s3_object_path:str,temp_dir:str):
    """
    Gets an object from s3 or local filesystem (including zipped version).
    If not found even on s3, reports and starts waiting cycle until expected object is uploaded.
    """
    if S3_BUCKET_NAME is None:
        raise ValueError("pyutilz.cloud.cloud.S3_BUCKET_NAME must be configured before calling get_from_s3_or_cache()")
    while not exists(local_object_path):
        bDownload = True
        if s3_object_path.endswith(".zip"):
            if exists(local_object_path + ".zip"):
                logger.info("Zipped version of %s found in local FS", s3_object_path)
                bDownload = False

        if bDownload:
            # If file exists in our s3
            if s3_file_exists(s3_object_path, S3_BUCKET_NAME):
                # Load model from our s3
                try:
                    if s3_object_path.endswith(".zip"):
                        s3.meta.client.download_file(Bucket=S3_BUCKET_NAME, Key=s3_object_path, Filename=local_object_path + ".zip")
                    else:
                        s3.meta.client.download_file(Bucket=S3_BUCKET_NAME, Key=s3_object_path, Filename=local_object_path)
                except Exception as e:
                    logger.error("Error while downloading model %s from bucket %s: %s", s3_object_path, S3_BUCKET_NAME, e)
                else:
                    logger.info("Downloaded model %s from bucket %s", s3_object_path, S3_BUCKET_NAME)
            else:
                logger.warning("Model %s not found in bucket %s", s3_object_path, S3_BUCKET_NAME)
                sleep(10)
        else:
            if s3_object_path.endswith(".zip"):
                if exists(local_object_path + ".zip"):
                    # unpack it
                    import shutil
                    try:
                        shutil.unpack_archive(local_object_path + ".zip", temp_dir)
                    except Exception as e:
                        logger.error("Error while unpacking model from bucket %s: %s", local_object_path + ".zip", e)
                    else:
                        logger.info("Unzipped model from archive %s", local_object_path + ".zip")
                        os.remove(local_object_path + ".zip")
