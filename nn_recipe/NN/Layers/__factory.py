from . import *

lookup = {
    Linear.ID: Linear
}


def LayerFactory(data:dict):
    layers = []
    for layer in data:
        layer_id = layer.pop("ID")
        layers.append(lookup[layer_id].load(layer))
    return layers