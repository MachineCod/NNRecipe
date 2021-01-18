from src.opt.Rms_Nesterov import GDNestrov
import numpy as np


class  GDAdam(GDNestrov):
    def __init__(self, iteration_no, *args, **kwargs ):
        super(GDAdam, self).__init__(*args, **kwargs)
        self.__iteration_no = iteration_no

    def optimize(self, layer, delta: np.ndarray) -> None:
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        if not hasattr(layer, "f"):
            layer.f = np.zeros_like(layer.weights)
            layer.fo = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(np.dot(delta, layer.local_grad["dW"]))
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(np.sum(delta) / delta.shape[1])

        layer.f = self._roh * layer.a + (1 - self._beta) * np.dot(delta, layer.local_grad["dW"])
        layer.fo = self._roh * layer.ao + (1 - self._beta) * (np.sum(delta) / delta.shape[1])

        self._learning_rate = self._learning_rate * np.power(1 - np.power(self._roh,self.__iteration_no), 0.5) / (1 - np.power(self._beta, self.__iteration_no))

        layer.weights = layer.weights - self._learning_rate / np.power(layer.a, 0.5) * layer.f
        layer.bias = layer.bias - self._learning_rate / np.power(layer.ao, 0.5) * layer.fo