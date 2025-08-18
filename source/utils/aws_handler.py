# utils/aws_handler.py

# global imports
import botocore.exceptions
import boto3
import functools
import io
import os
from typing import Any

# local imports
from source.utils import SingletonMeta

class AWSHandler(metaclass = SingletonMeta):
    """
    Responsible for handling communication with Amazon AWS services.
    """

    # local constants
    __DEFAULT_REGION = "eu-central-1"

    def __renew_s3_client_session(self) -> Any:
        """
        Assumes a role in AWS and renews S3 client session. Functionality is
        put into a separate method to allow for easier session renewal. The
        credentials are expired after 1 hour by default.
        """

        credentials = self.__credential_session.client('sts'). \
            assume_role(RoleArn = self.__role_arn,
                        RoleSessionName = 'S3_bucket_user_session')['Credentials']

        return boto3.client('s3', aws_access_key_id=credentials['AccessKeyId'],
                            aws_secret_access_key=credentials['SecretAccessKey'],
                            aws_session_token=credentials['SessionToken'],
                            region_name=self.__region_name)

    def __init__(self, region_name: str = __DEFAULT_REGION) -> None:
        """
        Class constructor. Before calling it AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        and ACCOUNT_ID should be available as environmental variables.

        Parameters:
            region_name (str): Region name to connect to.

        Raises:
            RuntimeError: If AWS credentials or account ID are not defined.
        """

        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        ACCOUNT_ID = os.getenv('ACCOUNT_ID')
        ROLE_NAME = os.getenv('ROLE_NAME')
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not ACCOUNT_ID \
            or not ROLE_NAME:
            raise RuntimeError("AWS credentials or account ID not found in environment variables!")

        self.__credential_session = boto3.Session(aws_access_key_id = AWS_ACCESS_KEY_ID,
                                                  aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
        self.__role_arn = f'arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}'
        self.__region_name = region_name
        self.__s3_session = self.__renew_s3_client_session()

    def with_session_renewal(method):
        """
        Decorator to handle session renewal on ExpiredToken error.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == 'ExpiredToken':
                    self.__s3_session = self.__renew_s3_client_session()
                    try:
                        return method(self, *args, **kwargs)
                    except Exception as e2:
                        raise RuntimeError(f"Operation failed after session renewal! Original error: {e2}")
                else:
                    raise # Re-raise the exception if it's not an ExpiredToken error
            except Exception as e:
                raise RuntimeError(f"Did not manage to perform S3 operation! Original error: {e}")
        return wrapper

    @with_session_renewal
    def upload_file_to_s3(self, bucket_name: str, file_path: str, desired_name: str = "") -> None:
        """
        Attempts to upload local file specified by path to S3 Amazon bucket.

        Parameters:
            bucket_name (str): String denoting bucket name.
            file_path (str): String representing file to the path that should be uploaded.
            desired_name (str): Desired name to be given to the file after being uploaded.
                If left unspecified, name does not change.

        Raises:
            RuntimeError: If approached problem during file uploading.
        """

        if desired_name == "":
            desired_name = file_path.split('/')[-1]

        self.__s3_session.upload_file(file_path, bucket_name, desired_name)

    @with_session_renewal
    def upload_buffer_to_s3(self, bucket_name: str, buffer: io.StringIO, desired_name: str) -> None:
        """
        Attempts to upload buffer as file body directly to S3 Amazon bucket.

        Parameters:
            bucket_name (str): String denoting bucket name.
            buffer (io.StringIO): Buffer containing data that should be directly
                written to bucket.
            desired_name (str): Desired name to be given to the file after being uploaded.

        Raises:
            RuntimeError: If approached problem during file uploading.
        """

        self.__s3_session.put_object(Bucket = bucket_name, Key = desired_name, Body = buffer.getvalue())

    @with_session_renewal
    def download_file_from_s3(self, bucket_name: str, file_name: str, desired_path: str = "") -> None:
        """
        Downloads a file from an S3 bucket to a local path.

        Parameters:
            bucket_name (str): The name of the S3 bucket.
            file_name (str): The key/path of the file in the S3 bucket.
            desired_path (str, optional): The local path where the file will be saved.
                If not provided, the file will be downloaded to the current working directory
                with the original filename.

        Raises:
            RuntimeError: If the download operation fails.
        """

        if desired_path == "":
            desired_path = os.getcwd() + '/' + file_name.split('/')[-1]

        self.__s3_session.download_file(bucket_name, file_name, desired_path)
