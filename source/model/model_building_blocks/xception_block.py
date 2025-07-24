# model/model_building_blocks/vgg16_block.py

# global imports
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, MaxPooling2D, SeparableConv2D

# local imports

class XceptionBlock:
    """
    Class implementing an Xception block compatible with the TensorFlow API. This block implements
    depthwise separable convolutions followed by max pooling and a residual connection, as seen in
    the Xception architecture.

    Diagram:

    ::

        Input Tensor --> +-----------------------+   +----------------------+   +--------------------+   +-----+
            |            | SeparableConv2D       |   | SeparableConv2D      |   | MaxPooling2D       |   | Add |
            |            | Filters: N1           |-->| Filters: N2          |-->| Pool Size: K3xK3   |-->|     |
            |            | Kernel Size: K1xK1    |   | Kernel Size: K2xK2   |   | Stride: S1xS1      |   |     |
            |            +-----------------------+   +----------------------+   +--------------------+   |     |
            |                                                                                            |     |
            +----------> +-----------------------+                                                       |     |
                         | Conv2D                |                                                       |     |
                         | Filters: N3           |                                                       |     |
                         | Kernel Size: K4xK4    |------------------------------------------------------>|     |
                         | Stride: S2xS2         |                                                       |     |
                         |                       |                                                       +-----+ --> Output Tensor
                         +-----------------------+
    """

    def __init__(self, kernels: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
                 filters: tuple[int, int, int], steps: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """
        Class constructor.

        Parameters:
            kernels (tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]):
                Sizes of all kernels used within this block.
            filters (tuple[int, int, int]): Number of filters used in the convolutional layers.
            steps (tuple[tuple[int, int], tuple[int, int]]): Strides for the max pooling and
                convolutional layers.
        """

        self.__separable_conv_2d_1_kernel_size: tuple[int, int] = kernels[0]
        self.__separable_conv_2d_2_kernel_size: tuple[int, int] = kernels[1]
        self.__max_pooling_2d_kernel_size: tuple[int, int] = kernels[2]
        self.__conv_2d_kernel_size: tuple[int, int] = kernels[3]
        self.__separable_conv_2d_1_nr_of_filters: int = filters[0]
        self.__separable_conv_2d_2_nr_of_filters: int = filters[1]
        self.__conv_2d_nr_of_filters: int = filters[2]
        self.__max_pooling_2d_step: tuple[int, int] = steps[0]
        self.__conv_2d_step: tuple[int, int] = steps[1]

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Applies depthwise separable convolutions with max pooling, and a residual connection to
        the input tensor.

        Parameters:
            input_tensor (tf.Tensor): Input tensor to which the transformations should be applied.

        Returns:
            tf.Tensor: Output tensor after the transformations have been applied.
        """

        # Depthwise separable convolution
        x_1 = SeparableConv2D(self.__separable_conv_2d_1_nr_of_filters,
                              self.__separable_conv_2d_1_kernel_size,
                              padding = 'same', use_bias = False)(input_tensor)
        x_1 = BatchNormalization()(x_1)
        x_1 = Activation('relu')(x_1)
        x_1 = SeparableConv2D(self.__separable_conv_2d_2_nr_of_filters,
                              self.__separable_conv_2d_2_kernel_size,
                              padding = 'same', use_bias = False)(x_1)
        x_1 = BatchNormalization()(x_1)
        x_1 = Activation('relu')(x_1)
        x_1 = MaxPooling2D(self.__max_pooling_2d_kernel_size, strides=self.__max_pooling_2d_step, padding = 'same')(x_1)

        # Residual connection
        x_2 = Conv2D(self.__conv_2d_nr_of_filters, self.__conv_2d_kernel_size, strides = self.__conv_2d_step,
                     padding = 'same', use_bias = False)(input_tensor)
        x_2 = BatchNormalization()(x_2)

        output_tensor = Add()([x_1, x_2])

        return output_tensor
