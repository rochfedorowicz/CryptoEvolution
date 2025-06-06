# tests/model/test_inception_block.py

# global imports
import logging
import numpy as np
import tensorflow as tf
from ddt import data, ddt, unpack
from unittest import TestCase

# local imports
from source.model import InceptionBlock

@ddt
class InceptionBlockTestCase(TestCase):
    """
    Test case for InceptionBlock class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        kernels = [(1, 1), (3, 3), (5, 5), (3, 3)]
        filters = [16, 16, 16, 16]
        steps = (1, 1)

        self.__sut: InceptionBlock = InceptionBlock(kernels, filters, steps)

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
        ([16, 16, 16, 16], (1, 1), [(1, 1), (3, 3), (5, 5), (3, 3)]),
        ([16, 16, 16, 16], (1, 1), [(1, 3), (3, 3), (5, 3), (3, 3)]),
        ([32, 32, 32, 32], (1, 1), [(1, 2), (3, 4), (5, 6), (4, 4)])
    )
    @unpack
    def test_inception_block_call(self, filters: tuple[int, int, int, int], steps: tuple[int, int],
        kernels: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]) -> None:
        """
        Tests the call method of the InceptionBlock class.

        Verifies that the InceptionBlock layer processes the input tensor correctly
        and returns an output tensor with the expected shape.

        Asserts:
            The output tensor is of type tf.Tensor.
            The shape of the output tensor matches the expected dimensions based on the input tensor
            and the number of filters.
        """

        logging.info("Attempting to call InceptionBlock layer.")
        self.__update_sut(conv_2d_1_kernel_size = kernels[0],
                          conv_2d_2_kernel_size = kernels[1],
                          conv_2d_3_kernel_size = kernels[2],
                          max_pooling_2d_kernel_size = kernels[3],
                          conv_2d_1_nr_of_filters = filters[0],
                          conv_2d_2_nr_of_filters = filters[1],
                          conv_2d_3_nr_of_filters = filters[2],
                          conv_2d_4_nr_of_filters = filters[3],
                          max_pooling_2d_step = steps)
        expected_dimension_size_x = 64
        expected_dimension_size_y = 64
        mocked_input_tensor = tf.random.normal(shape = (1, expected_dimension_size_x,
                                                        expected_dimension_size_y, 3))

        logging.info("Invoking __call__ method.")
        output_tensor = self.__sut.__call__(mocked_input_tensor)

        logging.info("Validating expected results.")
        self.assertIsInstance(output_tensor, tf.Tensor)
        self.assertEqual(output_tensor.shape, (1, expected_dimension_size_x,
                                               expected_dimension_size_y,
                                               np.array(filters).sum()))
