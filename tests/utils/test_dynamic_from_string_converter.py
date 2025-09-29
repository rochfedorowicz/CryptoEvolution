# tests/utils/test_dynamic_from_string_converter.py

# global imports
import logging
import pandas as pd
import sklearn
from ddt import data, ddt, unpack
from unittest import TestCase

# local imports
from source.environment import TradingEnvironment
from source.utils import DynamicFromStringConverter

@ddt
class DynamicFromStringConverterTestCase(TestCase):
    """
    Test case for DynamicFromStringConverter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: DynamicFromStringConverter = DynamicFromStringConverter()

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

    @data(
        ({'builtins': False}, 'str', str),
        ({'pandas': False}, 'DataFrame', pd.DataFrame),
        ({'sklearn': False}, 'RandomForestClassifier', sklearn.ensemble.RandomForestClassifier),
        ({'source': False}, 'TradingEnvironment', TradingEnvironment)
    )
    @unpack
    def test_dynamic_from_string_converter_convert_from_string__success(self, packages_to_be_registered: dict[str, bool],
            expected_class_name: str, expected_class_handle: type) -> None:
        """
        Tests DynamicFromStringConverter's convert_from_string functionality.

        Validates that the converter can successfully convert a string
        representation of a class into the actual class handle.

        Parameters:
            packages_to_be_registered: Dictionary of package names to be registered in the converter.
            expected_class_handle: The expected class handle that should be returned by the conversion.

        Asserts:
            The result of the conversion matches the expected class handle.
        """

        logging.info(f"Attempt to get {expected_class_handle} from DynamicFromStringConverter.")
        self.__sut.register_packages(packages_to_be_registered)

        logging.info("Invoking get_class_handle method.")
        result = self.__sut.get_class_handle(expected_class_name)

        logging.info("Validating expected result.")
        self.assertEqual(result, expected_class_handle)