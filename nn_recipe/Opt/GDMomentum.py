from .gd import GD
import numpy as np


class GDMomentum(GD):
    ID = 6

    def __init__(self, beta, *args, **kwargs):
        super(GDMomentum, self).__init__(*args, **kwargs)
        self._beta = beta

    def update_delta(self, layer, delta: np.ndarray):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w,delta_b

    def optimize(self, layer, delta: np.ndarray, *args, **kwargs) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)

        layer.v = self._beta * layer.v - self._learning_rate * delta_w
        layer.vo = self._beta * layer.vo - self._learning_rate * delta_b
        layer.weights = layer.weights + layer.v
        layer.bias = layer.bias + layer.vo

    def flush(self, layer):
        del layer.vo
        del layer.v

    def _save(self):
        return {
            "lr": self._learning_rate,
            "beta": self._beta,
        }

    @staticmethod
    def load(data):
        return GDMomentum(learning_rate=data["lr"], beta=data["beta"])




