from src.opt.optimizer import optimizer
import numpy as np


class GDAdaDelta(optimizer):
    def __init__(self,roh, *args, **kwargs):
        super(AdaDelta, self).__init__(*args, **kwargs)
        self.__roh = roh

    def optimize(self, layer, delta: np.ndarray) -> None:
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        if not hasattr(layer, "k"):
            layer.k = np.zeros_like(layer.weights)
            layer.ko = np.zeros_like(layer.bias)

        layer.a = self.__roh * layer.a + (1 - self.__roh) * np.square(np.dot(delta, layer.local_grad["dW"]))
        layer.ao = self.__roh * layer.ao + (1 - self.__roh) * np.square(np.sum(delta) / delta.shape[1])

        layer.k = self._roh * layer.k + (1 - self._roh) * layer.k / (layer.a + np.finfo(float).eps) * np.square(np.dot(delta, layer.local_grad["dW"]))
        layer.ko = self._roh * layer.ko + (1 - self._roh) * layer.ko / (layer.ao + np.finfo(float).eps) * np.square(np.sum(delta) / delta.shape[1])

        layer.weights = layer.weights -  np.power(layer.k / (layer.a + np.finfo(float).eps), 0.5) * np.dot(delta, layer.local_grad["dW"])
        layer.bias = layer.bias -  np.power(layer.ko / (layer.ao + np.finfo(float).eps), 0.5) * np.sum(delta) / delta.shape[1]
