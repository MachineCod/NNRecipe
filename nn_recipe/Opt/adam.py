from .leakyAdagrad import GDLeakyAdaGrad
import numpy as np


class GDAdam(GDLeakyAdaGrad):
    """
   :note: This class update the weights of the passed layer by using AdaDelta method, the needed hyper-parameters in
   this class is (roh), beta, iteration number (number of epochs), learning rate
    """
    ID = 3

    def __init__(self, beta=0.999, *args, **kwargs):
        """
        :param beta: hyper-parameter set empirically by the user, value ]0 , 1[
        :type beta: positive real number
        """
        super(GDAdam, self).__init__(*args, **kwargs)
        self.__beta = beta

    def optimize(self, layer, delta: np.ndarray, number_of_examples, iteration, *args, **kwargs) -> None:
        """
        :note:  1-This function update the layer weights according to this algorithm
                F = beta * F + (1 - beta) * ∂L/∂W
                A = roh * A + (1 - roh) * square(∂L/∂W)
                modified learning rate = learning rate * (square-root (1 - power(roh, iteration number)) /
                (1 - power(beta, iteration number)
                weights = weights - modified learning rate * power(A, -0.5) * F
                A and F are accumulators initialized by zero matrix with dimensions like the layer's weights
                A and F have two parts the accumulator of the weights part and bias part
                A for weights ==> layer.a     for bias ==> layer.ao
                F for weights ==> layer.f     for bias ==> layer.fo
                2- Initialize accumulators as a layer attribute if they are not already there
        :param layer: a layer in the training network
        :type layer: layer
        :param delta: the chain rule of multiplying the partial derivative of the loss function by the desired layer
                      weights passing by all activation functions (backpropagation)
        :type delta: np.ndarray
        :param number_of_examples: number of examples in the dataset
        :type number_of_examples: positive integer
        :param iteration: iteration number of epochs
        :type iteration: positive integer
        """
        delta_w, delta_b = self.update_delta(layer, delta, number_of_examples)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        if not hasattr(layer, "f"):
            layer.f = np.zeros_like(layer.weights)
            layer.fo = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(delta_w)
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(delta_b)

        layer.f = self.__beta * layer.f + (1 - self.__beta) * delta_w
        layer.fo = self.__beta * layer.fo + (1 - self.__beta) * delta_b

        learning_rate = self._learning_rate * np.power(1 - np.power(self._roh, iteration), 0.5) / \
                        (1 - np.power(self.__beta, iteration))

        layer.weights = layer.weights - learning_rate / np.power(layer.a, 0.5) * layer.f
        layer.bias = layer.bias - learning_rate / np.power(layer.ao, 0.5) * layer.fo

    def flush(self, layer):
        """
        :note: This function deletes the added attributes to objects (accumulators)
        :param layer: a layer in the training network
        :type layer: layer
        """
        del layer.ao
        del layer.a
        del layer.fo
        del layer.f

    def _save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        return {
            "lr": self._learning_rate,
            "beta": self.__beta,
            "roh": self._roh
        }

    @staticmethod
    def load(data):
        """
        This function loads the hyper parameters of the optimizing technique
        """
        return GDAdam(learning_rate=data["lr"], beta=data["beta"], roh=data["roh"])
