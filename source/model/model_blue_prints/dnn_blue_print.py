# model/model_blue_prints/dnn_blue_print.py

# global imports
import math
from tensorflow.keras import layers, Model
from typing import Optional

# local imports
from source.model import BluePrintBase, ModelAdapterBase, TFModelAdapter

class DnnBluePrint(BluePrintBase):
    """
    Blueprint for creating a DNN.

    This class implements a model blueprint that constructs a neural network using
    fully connected (dense) layers. Expected input is a 1D vector of features.
    """

    def __init__(self, dense_squeezing_coeff: int = 2, dense_repetition_coeff: int = 1) -> None:
        """
        Initializes the DnnBluePrint with the specified configuration parameters.

        Parameters:
            dense_squeezing_coeff (int): Factor by which dense layer sizes are reduced.
            dense_repetition_coeff (int): Number of dense layers of the same size to use.
        """

        self.__dense_squeezing_coeff = dense_squeezing_coeff
        self.__dense_repetition_coeff = dense_repetition_coeff

    def instantiate_model(self, input_shape: tuple[int, int], output_length: int,
        dense_squeezing_coeff: Optional[int] = None, dense_repetition_coeff: Optional[int] = None) -> ModelAdapterBase:
        """
        Creates and returns a DNN model according to specified parameters.

        The method constructs a neural network that:
        1. Processes a 1D input vector through multiple dense layers
        2. Produces a softmax output for classification

        Parameters:
            input_shape (tuple[int, int]): Shape of the input tensor
            output_length (int): Number of output classes/actions
            dense_squeezing_coeff (int): Factor by which dense layer sizes are reduced
            dense_repetition_coeff (int): Number of dense layers of the same size to use

        Returns:
            Model: Keras model implementing the DNN architecture to be compiled further.
        """

        if dense_squeezing_coeff is None:
            dense_squeezing_coeff = self.__dense_squeezing_coeff
        if dense_repetition_coeff is None:
            dense_repetition_coeff = self.__dense_repetition_coeff

        input_vector = layers.Input((1, input_shape[0]))
        reshaped_input_vector = layers.Reshape((input_shape[0],))(input_vector)

        closest_smaller_power_of_coeff = int(math.pow(dense_squeezing_coeff,
                                                      int(math.log(reshaped_input_vector.shape[-1],
                                                                   dense_squeezing_coeff))))
        dense = layers.Dense(closest_smaller_power_of_coeff, activation='relu')(reshaped_input_vector)
        dense = layers.BatchNormalization()(dense)

        number_of_nodes = closest_smaller_power_of_coeff // dense_squeezing_coeff
        nr_of_dense_layers = int(math.log(closest_smaller_power_of_coeff, dense_squeezing_coeff))
        for _ in range(nr_of_dense_layers):
            for _ in range(dense_repetition_coeff):
                dense = layers.Dense(number_of_nodes, activation='relu')(dense)
            dense = layers.BatchNormalization()(dense)
            number_of_nodes //= dense_squeezing_coeff
            if int(math.log(number_of_nodes, 10)) == int(math.log(output_length, 10)) + 1:
                dense = layers.Dropout(0.3)(dense)
            elif int(math.log(number_of_nodes, 10)) == int(math.log(output_length, 10)):
                break

        output = layers.Dense(output_length, activation='softmax')(dense)

        return TFModelAdapter(Model(inputs = input_vector, outputs = output))