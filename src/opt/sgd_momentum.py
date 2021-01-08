from src.opt.gcd import GCD
import numpy as np


class SGDMomentum(GCD):
    def __init__(self, learning_rate,  alpha=0.5):
        super(GCDMomentum, self).__init__(learning_rate)
        pass
        # self.__learning_rate = learning_rate
        # self.__alpha = alpha

    def optimize(self, layer, global_grade: np.ndarray ) -> None:
        """
			from layer : 
		    layer.V : V (t-1) = dl/dw(t-1) + alpha dl/dw(t-2) + alpha^2 dl/dw(t-3) + ...
			where alpha  = momentum  : bounded ]0,1[ 
			V , global_grade and layer.weights ve same width

			should be used with mini-batch training
				
        """
        layer.V = self.__alpha*layer.V + global_grade 
        layer.weights = layer.weights - self.__learning_rate *layer.V
