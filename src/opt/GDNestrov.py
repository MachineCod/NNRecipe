from src.opt.GDMomentum import GDMomentum
import numpy as np

class GDNestrov(GDMomentum):
    def __init__(self, *args, **kwargs):
        super(GDNestrov, self).__init__(*args,**kwargs)

    def optimize(self, layer, delta: np.ndarray) -> None:
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)
        layer.v = self._beta * layer.v - self._learning_rate * np.dot(delta, layer.local_grad["dW"]) * (layer.weights + self._beta * layer.v)
        layer.vo = self._beta * layer.vo - self._learning_rate * np.sum(delta)/delta.shape[1] * (layer.bias + self._beta * layer.vo)
        layer.weights = layer.weights + layer.v
        print("before\n", layer.bias.shape)
        layer.bias = layer.bias + layer.vo
        print("after\n", layer.bias.shape)