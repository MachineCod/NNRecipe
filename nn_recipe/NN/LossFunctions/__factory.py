from . import *

lookup = {
    CrossEntropyLoss.ID: CrossEntropyLoss,
    HingeLoss.ID: HingeLoss,
    MeanSquaredLoss.ID: MeanSquaredLoss,
    MClassLogisticLoss.ID: MClassLogisticLoss
}


def LossFunctionFactory(data:int):
    loss_id = data.pop("ID")
    return lookup[loss_id](sum=data["sum"], axis=data["axis"])