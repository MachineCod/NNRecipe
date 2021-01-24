from.__optimizer import Optimizer
import numpy as np


class GD(Optimizer):
    """
     :note: This class update the weights of the passed layer by using the standard method, the needed hyper-parameter
            in this class is the learning rate (alpha).
    """
    ID = 0

    def __init__(self, learning_rate: float = 0.01, *args, **kwargs):
        """
        :param learning_rate: learning rate
        :type learning_rate: positive real number with default value 0.01 (empirical value)
        """
        if learning_rate <= 0:
            raise Optimizer.LearningRateValueError(learning_rate)
        if type(learning_rate) is not float and type(learning_rate) is not int:
            raise Optimizer.LearningRateTypeError(type(learning_rate))
        self._learning_rate = learning_rate

    def update_delta(self, layer, delta: np.ndarray, number_of_examples):
        """
        :note: This function creates
               1- delta_w which is ∂loss/∂weights (∂L/∂W) by multiplying (delta * ∂Z/∂W) then dividing by (N)
                  delta :(∂L/∂O * ∂O/∂Z * ∂Z/∂W) where (O) is the output of the layer activation function
                                                       (Z) is the input of the layer activation function
                                                       (W) is the layer weights
                                                       (N) is the number of samples in dataset
               2- delta_b which is ∂loss/∂bias (∂L/∂b) by multiplying (delta * ∂Z/∂b) then dividing by (N)
                  delta :(∂L/∂O * ∂O/∂Z * ∂Z/∂W) where (O) is the output of the layer activation function
                                                       (Z) is the input of the layer activation function
                                                       (b) is the layer bias
                                                       (N) is the number of samples in dataset
               This explanation is for the last layer weights updating for deeper layers weights updating the delta is
               a longer chain this function works regardless of the layer depth, this depth is included in delta
        :param delta: the chain rule of multiplying the partial derivative of the loss function by the desired layer
                      weights passing by all activation functions (backpropagation)
        :type delta: np.ndarray
        :return: delta_w, delta_b
        :rtype: np.ndarray
        :param layer: a layer in the training network
        :type layer: layer
        :param number_of_examples: number of examples in the dataset
        :type number_of_examples: positive integer
        """
        delta_w = np.dot(delta, layer.local_grad["dW"]) / number_of_examples
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1] / number_of_examples
        return delta_w, delta_b

    def optimize(self, layer, delta: np.ndarray, number_of_examples, *args, **kwargs) -> None:
        """
        :note:  This function update the layer weights according to this algorithm
                weights = weights - learning rate *  ∂L/∂W
        :param layer: a layer in the training network
        :type layer: layer
        :param delta: the chain rule of multiplying the partial derivative of the loss function by the desired
                      layer weights passing by all activation functions (backpropagation)
        :type delta: np.ndarray
        :param number_of_examples: number of examples in the dataset
        :type number_of_examples: positive integer
        """
        delta_w, delta_b = self.update_delta(layer, delta, number_of_examples)
        layer.weights = layer.weights - self._learning_rate * delta_w
        layer.bias = layer.bias - self._learning_rate * delta_b

    def flush(self, layer):
        pass

    def _save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        return {
            "lr": self._learning_rate
        }

    @staticmethod
    def load(data):
        """
        This function loads the hyper parameters of the optimizing technique
        """
        return GD(learning_rate=data["lr"])

