# tests/utils/test_gradient_handler.py

# global imports
import logging
import os
import re
from ddt import ddt
from unittest import TestCase
from unittest.mock import ANY, Mock, patch
from typing import Any

# local imports
from source.utils import GradientHandler

class StringRegexMatcher:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        return bool(re.search(self.pattern, other))

    def __repr__(self) -> str:
        return f"StringRegexMatcher({self.pattern!r})"

@ddt
class GradientHandlerTestCase(TestCase):
    """
    Test case for GradientHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    @patch.dict(os.environ, {
        'GRADIENT_API_KEY': 'api_key',
        'GRADIENT_PROJECT_ID': 'project_id'
    })
    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: GradientHandler = GradientHandler()

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

    @patch('gradient.NotebooksClient.create', new_callable = Mock)
    def test_gradient_handler_created_notebook__with_2_machine_types_1_fail_1_success(self,
            mocked_notebooks_client_create) -> None:
        """
        Tests the create_notebook method of GradientHandler.

        Verifies that the create_notebook method successfully creates a notebook using
        the Paperspace Gradient API. The NotebooksClient's create method is mocked to simulate the creation
        process without making actual API calls.

        Asserts:
            The notebook ID returned by create_notebook matches the mocked notebook ID.
            The NotebooksClient's create method was called once with the correct parameters.
        """

        logging.info("Attempting to create a notebook.")
        mocked_error = Exception("""Failed to create resource: We are currently out of capacity for the
                                    selected VM type. Try again in a few minutes, or select a different instance.""")
        mocked_notebook_id = 'NOTEBOOK_ID'
        mocked_notebooks_client_create.side_effect = [mocked_error, mocked_notebook_id]

        expected_workspace = 'https://github.com/user/repo_name.git'
        expected_machine_types = ['Free-A4000', 'Free-P5000']
        expected_container = 'paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125'
        expected_timeout = 6
        expected_command = 'echo "Testing command"'
        expected_command_full_matcher = StringRegexMatcher(rf'{expected_command}')
        expected_environment = dict()

        logging.info("Invoking create_notebook method.")
        notebook_id = self.__sut.create_notebook(github_repository_url = expected_workspace,
                                                    command_to_invoke = expected_command,
                                                    machine_types = expected_machine_types)

        logging.info("Validating expected results and notebook creation.")
        self.assertEqual(mocked_notebook_id, notebook_id)
        self.assertEqual(mocked_notebooks_client_create.call_count, len(expected_machine_types))
        for expected_machine_type in expected_machine_types:
            mocked_notebooks_client_create.assert_any_call(machine_type = expected_machine_type,
                                                           container = expected_container,
                                                           project_id = self.__sut.project_id,
                                                           shutdown_timeout = expected_timeout,
                                                           workspace = expected_workspace,
                                                           command = expected_command_full_matcher,
                                                           environment = expected_environment,
                                                           name = ANY)