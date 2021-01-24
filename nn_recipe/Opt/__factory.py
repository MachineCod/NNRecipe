from . import *

lookup = {
    GDAdaDelta.ID: GDAdaDelta,
    GDAdaGrad.ID: GDAdaGrad,
    GDAdam.ID: GDAdam,
    GD.ID: GD,
    GDExpDec.ID: GDExpDec,
    GDInvDec.ID: GDInvDec,
    GDMomentum.ID: GDMomentum,
    GDLeakyAdaGrad.ID: GDLeakyAdaGrad
}


def OptimizerFactory(data:dict):
    opt_id = data.pop("ID")
    return lookup[opt_id].load(data)