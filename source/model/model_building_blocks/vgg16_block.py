# model/model_building_blocks/vgg16_block.py

# global imports
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# local imports

class Vgg16Block:
    """
    Class implementing Vgg16 block compatible with tensorflow API. This block is a core component of
    the VGG16 architecture, applying two convolutional layers followed by a max pooling layer to
    downsample and extract features from the input tensor.

    Diagram:

    ::

        Input Tensor --> +-----------------------+   +-----------------------+   +-----------------------+
                         | Conv2D                |   | Conv2D                |   | MaxPooling2D          |
                         | Filters: N1           |-->| Filters: N2           |-->| Pool Size: K3xK3      |
                         | Kernel Size: K1xK1    |   | Kernel Size: K2xK2    |   |                       |
                         +-----------------------+   +-----------------------+   +-----------------------+ --> Output Tensor
    """

    def __init__(self, kernels: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], filters: tuple[int, int]) -> None:
        """
        Class constructor.

        Parameters:
            kernels (tuple[tuple[int, int], tuple[int, int], tuple[int, int]]): Sizes of all kernels used within this block.
            filters (tuple[int, int]): Number of filters used in convolutional layers.
        """

        self.__conv_2d_1_kernel_size: tuple[int, int] = kernels[0]
        self.__conv_2d_2_kernel_size: tuple[int, int] = kernels[1]
        self.__max_pooling_2d_kernel_size: tuple[int, int] = kernels[2]
        self.__conv_2d_1_nr_of_filters: int = filters[0]
        self.__conv_2d_2_nr_of_filters: int = filters[1]

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Applies convolutional transformation with max pooling to input tensor.

        Parameters:
            input_tensor (tf.Tensor): Input tensor that transformations should be applied to.

        Returns:
            (tf.Tensor): Output tensor with applied transformations.
        """

        x = Conv2D(self.__conv_2d_1_nr_of_filters, self.__conv_2d_1_kernel_size,
                   activation = 'relu', padding = 'same')(input_tensor)
        x = Conv2D(self.__conv_2d_2_nr_of_filters, self.__conv_2d_2_kernel_size,
                   activation = 'relu', padding = 'same')(x)

        output_tensor = MaxPooling2D(self.__max_pooling_2d_kernel_size)(x)

        return output_tensor