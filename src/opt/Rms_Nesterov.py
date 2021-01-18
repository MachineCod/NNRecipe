from src.opt.GDMomentum import GDMomentum
import numpy as np

class GDRmsNestrov(GDMomentum):
    def __init__(self,roh, *args, **kwargs):
        super(GDRmsNestrov, self).__init__(*args,**kwargs)
        self._roh = roh
    def optimize(self, layer, delta: np.ndarray) -> None:
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(np.dot(delta, layer.local_grad["dW"]) * (layer.weights + self._beta * layer.v))
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(np.sum(delta) / delta.shape[1] * (layer.bias + self._beta * layer.vo))

        layer.v = self._beta * layer.v - (self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5)) * np.dot(delta, layer.local_grad["dW"]) * (layer.weights + self._beta * layer.v)
        layer.vo = self._beta * layer.vo - (self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5)) * np.sum(delta)/delta.shape[1] * (layer.bias + self._beta * layer.vo)

        layer.weights = layer.weights + layer.v
        print("before\n", layer.bias.shape)
        layer.bias = layer.bias + layer.vo
        print("after\n", layer.bias.shape)