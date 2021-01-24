from . import *


lookup = {
    HardTanh.ID: HardTanh,
    Identity.ID: Identity,
    LeakyReLU.ID: LeakyReLU,
    ReLU.ID: ReLU,
    Sigmoid.ID: Sigmoid,
    Tanh.ID: Tanh
}


def ActivationFunctionFactory(id:int):
    return lookup[id]()