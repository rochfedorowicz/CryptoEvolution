# model/model_building_blocks/inception_block.py

# global imports
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D

# local imports

class InceptionBlock:
    """
    Class implementing an Inception block compatible with the TensorFlow API. This block uses
    parallel convolutions with different kernel sizes and a max-pooling layer, followed by a
    concatenation of the results, as seen in the Inception architecture.

    Diagram:

    ::

        Input Tensor --> +-----------------------+
            |            | Conv2D - typ. 1xY     |                               +-------------+
            |            | Kernel Size: K1xK1    |------------------------------>| Concatenate |
            |            | Filters: N1           |                               |             |
            |            +-----------------------+                               |             |
            |                                                                    |             |
            +----------> +-----------------------+   +-----------------------+   |             |
            |            | Conv2D - typ. 3xY     |   | Conv2D - typ. 3xY     |   |             |
            |            | Kernel Size: K1xK1    |-->| Kernel Size: K2xK2    |-->|             |
            |            | Filters: N2           |   | Filters: N2           |   |             |
            |            +-----------------------+   +-----------------------+   |             |
            |                                                                    |             |
            +----------> +-----------------------+   +-----------------------+   |             |
            |            | Conv2D - typ. 5xY     |   | Conv2D - typ. 5xY     |   |             |
            |            | Kernel Size: K1xK1    |-->| Kernel Size: K3xK3    |-->|             |
            |            | Filters: N3           |   | Filters: N3           |   |             |
            |            +-----------------------+   +-----------------------+   |             |
            |                                                                    |             |
            +----------> +-----------------------+   +-----------------------+   |             |
                         | MaxPooling2D          |   | Conv2D                |   |             |
                         | Kernel Size: K4xK4    |-->| Kernel Size: K1xK1    |-->|             |
                         | Stride: S1xS1         |   | Filters: N4           |   +-------------+ --> Output Tensor
                         +-----------------------+   +-----------------------+
    """

    def __init__(self, kernels: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
                 filters: tuple[int, int, int, int], steps: tuple[int, int]) -> None:
        """
        Class constructor.

        Parameters:
            kernels (tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]):
                Sizes of all kernels used within this block.
            filters (tuple[int, int, int, int]): Number of filters used in the convolutional layers.
            steps (tuple[int, int]): Strides for the max pooling layer.
        """

        self.__conv_2d_1_kernel_size: tuple[int, int] = kernels[0]
        self.__conv_2d_2_kernel_size: tuple[int, int] = kernels[1]
        self.__conv_2d_3_kernel_size: tuple[int, int] = kernels[2]
        self.__max_pooling_2d_kernel_size: tuple[int, int] = kernels[3]
        self.__conv_2d_1_nr_of_filters: int = filters[0]
        self.__conv_2d_2_nr_of_filters: int = filters[1]
        self.__conv_2d_3_nr_of_filters: int = filters[2]
        self.__conv_2d_4_nr_of_filters: int = filters[3]
        self.__max_pooling_2d_step: tuple[int, int] = steps

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Applies parallel convolutions with different kernel sizes and a max-pooling layer,
        followed by concatenation of the results.

        Parameters:
            input_tensor (tf.Tensor): Input tensor to which the transformations should be applied.

        Returns:
            tf.Tensor: Output tensor after the transformations have been applied.
        """

        # 1xY convolution
        x_1 = Conv2D(self.__conv_2d_1_nr_of_filters, self.__conv_2d_1_kernel_size, padding = 'same',
                     activation = 'relu')(input_tensor)

        # 3xY convolution
        x_2 = Conv2D(self.__conv_2d_2_nr_of_filters, self.__conv_2d_1_kernel_size, padding = 'same',
                     activation = 'relu')(input_tensor)
        x_2 = Conv2D(self.__conv_2d_2_nr_of_filters, self.__conv_2d_2_kernel_size, padding = 'same',
                     activation = 'relu')(x_2)

        # 5xY convolution
        x_3 = Conv2D(self.__conv_2d_3_nr_of_filters, self.__conv_2d_1_kernel_size, padding = 'same',
                     activation = 'relu')(input_tensor)
        x_3 = Conv2D(self.__conv_2d_3_nr_of_filters, self.__conv_2d_3_kernel_size, padding = 'same',
                     activation = 'relu')(x_3)

        # Pooling
        x_4 = MaxPooling2D(self.__max_pooling_2d_kernel_size, strides=self.__max_pooling_2d_step,
                           padding = 'same')(input_tensor)
        x_4 = Conv2D(self.__conv_2d_4_nr_of_filters, self.__conv_2d_1_kernel_size,
                     padding = 'same', activation = 'relu')(x_4)

        output_tensor = Concatenate()([x_1, x_2, x_3, x_4])

        return output_tensor
