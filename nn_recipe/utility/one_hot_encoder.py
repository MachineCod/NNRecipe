from nn_recipe.utils.exceptions import check_float
import numpy as np
from typing import List


class OneHotEncoder:
    def __init__(self, types:List, active_state:int=1, inactive_state:int=0):
        self.__types = self.__process_types(types)
        self.__active_state = check_float(active_state, "Active State value must be a float")
        self.__inactive_state = check_float(inactive_state, "Inactive State value must be a float")

    def __process_types(self, types:List):
        if type(types) is not list:
            raise Exception
        return types

    def encode(self, Y:np.ndarray):
        out = np.empty(shape=(Y.shape[0], len(self.__types)), dtype=float)
        out[:, :] = self.__inactive_state
        self.__encode(Y, out)
        return out

    def __encode(self, y, out):
        for a in range(len(y)):
            out[a, self.__types.index(y[a,0])] = self.__active_state

    def decode(self, Y:np.ndarray):
        out = []
        for a in range(len(Y)):
            row = Y[a,:]
            out.append(self.__types[np.where(row == 1)[0][0]])
        return out


# a = OneHotEncoder(["Animal", "ALI"])
# y = np.array([["Animal"], ["Animal"], ["ALI"]])
# a.encode(y)