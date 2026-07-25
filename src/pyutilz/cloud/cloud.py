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

    # Regression fix: `object=globals()` made read_config_file() write into THIS MODULE's
    # global namespace dict, not the local variables `aws_access_key_id`/`aws_secret_access_key`
    # declared just below -- Python resolves those names as the function's own local bindings
    # (compile-time scoping) throughout the whole function body, which the globals() mutation
    # never touched. boto3.Session() below was therefore ALWAYS called with (None, None),
    # silently falling back to ambient credentials (env vars / ~/.aws/credentials / an IAM
    # role) if any existed, or raising NoCredentialsError on the first real S3 call otherwise --
    # regardless of what settings.ini actually contained.
    creds: dict = {}
    if read_config_file(file=file, object=creds, section="S3", variables="aws_access_key_id,aws_secret_access_key"):
        aws_access_key_id = creds.get("aws_access_key_id")
        aws_secret_access_key = creds.get("aws_secret_access_key")
        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        s3 = boto_session.resource("s3")
        return s3


def s3_file_exists(key: str, bucket: str) -> bool:
    """Check whether an object with the given key exists in an S3 bucket (via a HEAD request)."""
    # Regression fix: a blanket `except Exception: return False` conflated a genuine "object
    # missing" 404 (the case this function is meant to report) with the module-level `s3`
    # global being None (connect_to_s3() never called, or it failed) -- an AttributeError on
    # `s3.meta` -- and with real botocore auth/network errors. All three previously looked
    # identical to a caller, masking the real root cause behind a misleading "not found" log
    # message and (in get_from_s3_or_cache()) an indefinite polling loop for a file that could
    # never appear "uploaded" no matter how long the wait, because the client was never
    # actually configured/reachable.
    if s3 is None:
        raise RuntimeError("pyutilz.cloud.cloud.s3 is not set; call connect_to_s3() first")
    try:
        from botocore.exceptions import ClientError

        s3.meta.client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise
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
                    # A persistent per-object failure (bad IAM perms, always-corrupt remote
                    # object) previously retried with only network-RTT pacing -- no backoff --
                    # unlike this function's two sibling failure branches ("not found in bucket"
                    # and "corrupt zip", both `sleep(10)`). Matching them here closes the same
                    # busy-loop risk class already fixed in those two branches.
                    logger.error("Error while downloading model %s from bucket %s: %s", s3_object_path, S3_BUCKET_NAME, e)
                    sleep(10)
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
                        # Regression fix: previously neither removed the corrupted zip nor
                        # slept -- the `while not exists(local_object_path):` loop immediately
                        # re-entered with the exact same on-disk state (unpack always fails the
                        # same way), an unbounded zero-delay busy loop pinning a CPU core at
                        # 100% forever. Removing the corrupted file lets the next iteration
                        # re-download it instead (mirrors the "not found in bucket" branch's own
                        # sleep(10), which this branch previously lacked entirely).
                        logger.error("Error while unpacking model from bucket %s: %s", local_object_path + ".zip", e)
                        try:
                            os.remove(local_object_path + ".zip")
                        except OSError as remove_err:
                            logger.error("Could not remove corrupted archive %s: %s", local_object_path + ".zip", remove_err)
                        sleep(10)
                    else:
                        logger.info("Unzipped model from archive %s", local_object_path + ".zip")
                        os.remove(local_object_path + ".zip")
