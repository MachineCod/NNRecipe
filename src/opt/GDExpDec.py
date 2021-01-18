from src.opt.GD import GD
import numpy as np


class GDExpDec(GD):
    def __init__(self, iteration_no, k,*args, **kwargs):
        super(GDExpDec, self).__init__(*args, **kwargs)
        self._iteration = iteration_no
        self._k = k

    def optimize(self, layer, delta: np.ndarray) -> None:
        self._learning_rate = self._learning_rate * np.exp(-self._k * self._iteration)
        layer.weights = layer.weights - self._learning_rate * np.dot(delta, layer.local_grad["dW"])
        layer.bias = layer.bias - self._learning_rate * np.sum(delta) / delta.shape[1]




