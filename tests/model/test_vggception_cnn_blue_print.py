# tests/model/test_vggception_cnn_blue_print.py

# global imports
import gc
import logging
from ddt import data, ddt, unpack
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, MaxPooling2D, SeparableConv2D
from typing import Any
from unittest import TestCase

# local imports
from source.model import VGGceptionCnnBluePrint

@ddt
class VGGceptionCnnBluePrintTestCase(TestCase):
    """
    Test case for VGGceptionCnnBluePrint class. Stores all the test cases
    and allows for convenient test case execution.
    """

    # local constants
    __COMPARE_MILLIONS = -6

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: VGGceptionCnnBluePrint = VGGceptionCnnBluePrint(should_apply_2d_convolution = False)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    @data(
        ({  # Referral model architecture
            'input_shape': (795,),
            'output_length': 3,
            'spatial_data_shape': (72, 11),
        }, 6, 2048, 12, 560_000_000),
        ({  # Decreased number of filters -> less filters in last layer,
            # therefore less dense layers until reaching output, therefore less parameters
            'input_shape': (795,),
            'output_length': 3,
            'spatial_data_shape': (72, 11),
            'number_of_filters': 16,
        }, 6, 1024, 11, 140_000_000),
        ({  # Increased CNN squezing coefficient -> less Xception blocks,
            # therefore less filters in last layer, therefore less dense layers until reaching
            # output, therefore less parameters
            'input_shape': (795,),
            'output_length': 3,
            'spatial_data_shape': (72, 11),
            'cnn_squeezing_coeff': 3
        }, 4, 512, 10, 35_000_000),
        ({  # Increased dense squezing coefficient -> less dense layers until reaching
            # output, however since first dense layer size with dense squeezing coefficient
            # equal to 3 is bigger than with equal to 2, therefore more parameters
            'input_shape': (795,),
            'output_length': 3,
            'spatial_data_shape': (72, 11),
            'dense_squeezing_coeff': 3
        }, 6, 2048, 8, 600_000_000),
        ({  # Increased dense repetition coefficient -> more dense layers until reaching
            # output, therefore more parameters
            'input_shape': (795,),
            'output_length': 3,
            'spatial_data_shape': (72, 11),
            'dense_repetition_coeff': 2
        }, 6, 2048, 22, 649_000_000),
        ({  # Increased filters number coefficient -> much more filters in last layer (needed to set
            # initial to 3 not to cause memory exhaustion), therefore more dense layers until reaching
            # output, therefore more parameters
            'input_shape': (795,),
            'output_length': 3,
            'spatial_data_shape': (72, 11),
            'number_of_filters': 3,
            'filters_number_coeff': 3
        }, 6, 2187, 12, 582_000_000)
    )
    @unpack
    def test_vggception_cnn_blue_print_instantiate_model(self, input_data: dict[str, Any], expected_nr_of_xception_blocks: int,
                                                         expected_nr_of_filters_in_last_CNN_layer: int, expected_nr_of_dense_layers: int,
                                                         expected_nr_of_parameters: int) -> None:
        """
        Tests VGGceptionCnnBluePrint's instantiate_model functionality with various configurations.

        Verifies that the instantiate_model method correctly creates neural network architectures
        with expected layer counts, shapes, and parameter numbers based on different input
        configurations. Tests multiple variations including different filter counts,
        squeezing coefficients, and layer repetition factors.

        Asserts:
            The generated model has the expected number of layers of each type.
            The convolutional layers have the expected shapes.
            The model has approximately the expected number of parameters.
        """

        logging.info("Starting instantiate model test.")
        expected_nr_of_separable_conv_layers = expected_nr_of_xception_blocks * 2
        expected_nr_of_conv_layers = expected_nr_of_xception_blocks + 2
        expected_nr_of_max_pooling_layers = expected_nr_of_xception_blocks + 1
        expected_last_CNN_layer_shape = (None, 1, input_data['spatial_data_shape'][1], expected_nr_of_filters_in_last_CNN_layer)

        logging.info(f"Instantiating model with params = {input_data}.")
        model = self.__sut.instantiate_model(**input_data).get_model()

        nr_of_parameters = model.count_params()
        nr_of_separable_conv_layers = 0
        nr_of_conv_layers = 0
        nr_of_max_pooling_layers = 0
        nr_of_add_layers = 0
        nr_of_dense_layers = 0
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Flatten):
                last_CNN_layer_shape = model.layers[i - 1].output_shape
            if isinstance(layer, SeparableConv2D):
                # 2 per Xception block
                nr_of_separable_conv_layers += 1
            elif isinstance(layer, Conv2D):
                # 1 per Xception block, 2 per VGG16 block
                nr_of_conv_layers += 1
            elif isinstance(layer, MaxPooling2D):
                # 1 per Xception block, 1 per VGG16 block
                nr_of_max_pooling_layers += 1
            elif isinstance(layer, Add):
                # 1 per Xception block
                nr_of_add_layers += 1
            elif isinstance(layer, Dense):
                # present in flatten part
                nr_of_dense_layers += 1

        del model
        gc.collect()

        logging.info(f"Veryfing created model properties.")
        self.assertEqual(nr_of_separable_conv_layers, expected_nr_of_separable_conv_layers)
        self.assertEqual(nr_of_conv_layers, expected_nr_of_conv_layers)
        self.assertEqual(nr_of_max_pooling_layers, expected_nr_of_max_pooling_layers)
        self.assertEqual(nr_of_add_layers, expected_nr_of_xception_blocks)
        self.assertEqual(last_CNN_layer_shape, expected_last_CNN_layer_shape)
        self.assertEqual(nr_of_dense_layers, expected_nr_of_dense_layers)
        self.assertAlmostEqual(nr_of_parameters, expected_nr_of_parameters, self.__COMPARE_MILLIONS)
