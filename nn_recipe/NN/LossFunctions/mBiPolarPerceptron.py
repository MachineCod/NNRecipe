from .__loss_function import LossFunction
import numpy as np


class MClassBipolarPerceptron(LossFunction):
    def _compute_loss(self, Y, Y_hat):
        return np.maximum(Y_hat - Y_hat[np.argmax(Y, axis=0)].reshape((-1, 1)), 0)

    def _compute_local_grad(self, Y, Y_Hat):
        pass

"""             np.max(Y, axis=0).reshape((-1, 1))
   Y_hat
0.24 0.60 0.9   0.8          
0.35 0.95 2.3   0.3
0.25 0.35 0.6   0.5
"""
