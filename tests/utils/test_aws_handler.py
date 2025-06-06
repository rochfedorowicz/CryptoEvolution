# tests/utils/test_aws_handler.py

# global imports
import boto3
import io
import logging
import os
import tempfile
from moto import mock_aws
from ddt import ddt
from unittest import TestCase
from unittest.mock import patch

# local imports
from source.utils import AWSHandler

@ddt
class AWSHandlerTestCase(TestCase):
    """
    Test case for AWSHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    # local constants
    __TEST_CONTENT = "This is a test file content."
    __TEST_FILE_NAME = "test_file.txt"

    @mock_aws
    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'access_key',
        'AWS_SECRET_ACCESS_KEY': 'secret_key',
        'ACCOUNT_ID': '123456789012',
        'ROLE_NAME': 'mocked_bucket-user-role'
    })
    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: AWSHandler = AWSHandler()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injections of certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                if name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    @mock_aws
    def test_aws_handler_upload_file_to_s3(self) -> None:
        """
        Tests the upload_file_to_s3 method of AWSHandler.

        Verifies that method successfully uploads a file to an S3 bucket.
        It mocks AWS S3 to create a bucket and handle the file upload. Also,
        the open function is mocked to simulate file creation and writing.

        Asserts:
            The content of the uploaded file in the S3 bucket matches the expected content.
        """

        logging.info("Attempting to upload file to S3 bucket.")
        mocked_bucket_name = 'mocked-bucket'
        mocked_region = 'us-east-1'
        mocked_s3_client = boto3.client('s3', region_name = mocked_region)
        mocked_s3_client.create_bucket(Bucket = mocked_bucket_name)

        with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
            tmp_file.write(self.__TEST_CONTENT.encode('utf-8'))
            mocked_file_path = tmp_file.name

        logging.info("Invoking upload_file_to_s3 method.")
        self.__sut.upload_file_to_s3(mocked_bucket_name, mocked_file_path, self.__TEST_FILE_NAME)
        os.remove(mocked_file_path)

        logging.info("Verifying file upload to S3 bucket.")
        mocked_s3_resource = boto3.resource('s3', region_name = mocked_region)
        object = mocked_s3_resource.Object(mocked_bucket_name, self.__TEST_FILE_NAME)
        file_content = object.get()['Body'].read().decode('utf-8')
        self.assertEqual(file_content, self.__TEST_CONTENT)

    @mock_aws
    def test_aws_handler_upload_buffer_to_s3(self) -> None:
        """
        Tests the upload_buffer_to_s3 method of AWSHandler.

        Verifies that method successfully uploads content from a buffer to an S3 bucket.
        It mocks AWS S3 to create a bucket and handle the buffer upload. The test
        creates a StringIO buffer with test content and passes it to the handler.

        Asserts:
            The content of the uploaded file in the S3 bucket matches the content from the buffer.
        """

        logging.info("Attempting to upload buffer to S3 bucket.")
        mocked_bucket_name = 'mocked-bucket'
        mocked_region = 'us-east-1'
        mocked_s3_client = boto3.client('s3', region_name = mocked_region)
        mocked_s3_client.create_bucket(Bucket = mocked_bucket_name)

        logging.info("Invoking upload_buffer_to_s3 method.")
        buffer_content = io.StringIO(self.__TEST_CONTENT)
        self.__sut.upload_buffer_to_s3(mocked_bucket_name, buffer_content, self.__TEST_FILE_NAME)

        logging.info("Verifying buffer upload to S3 bucket.")
        mocked_s3_resource = boto3.resource('s3', region_name = mocked_region)
        object = mocked_s3_resource.Object(mocked_bucket_name, self.__TEST_FILE_NAME)
        file_content = object.get()['Body'].read().decode('utf-8')
        self.assertEqual(file_content, self.__TEST_CONTENT)

    @mock_aws
    def test_aws_handler_download_file_from_s3(self) -> None:
        """
        Tests the download_file_from_s3 method of AWSHandler.

        Verifies that method successfully downloads a file from an S3 bucket.
        It mocks AWS S3 to create a bucket, uploads test content, and then
        downloads the file.

        Asserts:
            The content of the downloaded file matches what was originally uploaded.
        """

        logging.info("Attempting to download file from S3 bucket.")
        mocked_bucket_name = 'mocked-bucket'
        mocked_region = 'us-east-1'
        mocked_s3_client = boto3.client('s3', region_name = mocked_region)
        mocked_s3_client.create_bucket(Bucket = mocked_bucket_name)
        mocked_s3_client.put_object(
            Bucket = mocked_bucket_name,
            Key = self.__TEST_FILE_NAME,
            Body = self.__TEST_CONTENT
        )

        with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
            download_path = tmp_file.name

        logging.info("Invoking download_file_from_s3 method.")
        self.__sut.download_file_from_s3(mocked_bucket_name, self.__TEST_FILE_NAME, download_path)

        logging.info("Verifying file download from S3 bucket.")
        with open(download_path, 'r') as file:
            downloaded_content = file.read()
        os.remove(download_path)
        self.assertEqual(downloaded_content, self.__TEST_CONTENT)