from src.opt.GD import GD
import numpy as np


class GDMomentum(GD):
    def __init__(self, beta, *args, **kwargs):

        super(GDMomentum, self).__init__(*args, **kwargs)
        self._beta = beta

    def optimize(self, layer, delta: np.ndarray) -> None:
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)
        layer.v = self._beta * layer.v - self._learning_rate * np.dot(delta, layer.local_grad["dW"])
        layer.vo = self._beta * layer.vo - self._learning_rate * np.sum(delta)/delta.shape[1]
        layer.weights = layer.weights + layer.v
        layer.bias = layer.bias + layer.vo



