# model/model_building_blocks/se_block.py

# global imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Multiply, Reshape

# local imports

class SEBlock:
    """
    Class implementing a Squeeze-and-Excitation (SE) block compatible with the TensorFlow API.
    This block applies global average pooling followed by a squeeze-and-excitation operation,
    as described in the SE-Net architecture.

    Diagram:

    ::

        Input Tensor last dimension length - ITldl
        Reduction rate - Rr

        Input Tensor --> +-----------------+   +---------------+   +--------------------+   +--------------+
            |            | GlobalAvgPool   |-->| Reshape       |   | Dense              |   | Dense        |   +----------+
            |            +-----------------+   | Shape: ITldl  |-->| Nodes: ITldl // Rr |-->| Nodes: ITldl |-->| Multiply |
            |                                  |               |   |                    |   |              |   |          |
            |                                  +---------------+   +--------------------+   +--------------+   |          |
            |                                                                                                  |          |
            +------------------------------------------------------------------------------------------------->|          |
                                                                                                               +----------+ --> Output Tensor
    """

    def __init__(self, reduction_ratio: int = 16) -> None:
        """
        Class constructor.

        Parameters:
            reduction_ratio (int): Reduction ratio used to control the size of the squeeze operation.
        """

        self.__reduction_ratio: int = reduction_ratio

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Applies squeeze-and-excitation operation to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): Input tensor to which the SE operation should be applied.

        Returns:
            tf.Tensor: Output tensor after the SE operation has been applied.
        """

        filters = input_tensor.shape[-1]
        x_shape = (1, 1, filters)

        # Squeeze and excitation (SE)
        x = GlobalAveragePooling2D()(input_tensor)
        x = Reshape(x_shape)(x)
        x = Dense(filters // self.__reduction_ratio, activation = 'relu', use_bias = False)(x)
        x = Dense(filters, activation = 'sigmoid', use_bias = False)(x)

        output_tensor = Multiply()([input_tensor, x])

        return output_tensor
