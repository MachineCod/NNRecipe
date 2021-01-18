import numpy as np
from src.NN.layers.linear import Linear
from src.NN.activations.activation import Identity, Sigmoid
from src.NN.losses.MeanSquared import MeanSquaredLoss
from src.opt.GD import GD
from src.opt.GDExpDec import GDExpDec
from src.opt.GDMomentum import GDMomentum
from src.opt.GDNestrov import GDNestrov
from src.opt.Adagrad import AdaGrad
x = np.array([ #4x2
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])
y = np.array([ #4x1
    [-1],
    [1],
    [1],
    [-1]
])
opt = GDNestrov(learning_rate=0.1, beta=3)
msl = MeanSquaredLoss()

l2 = Linear(in_dim=2, out_dim=3, activation=Identity())
print(l2.bias.shape)
for a in range(2):
    out = l2(x)
    print("loss {}:".format(str(a)), msl(y, out))
    msl(y, out)
    delta = np.multiply(msl.local_grad.T, l2.local_grad["dZ"])
    print(delta.shape)
    opt.optimize(l2, delta)

    
    delta = np.dot( l2.local_grad["dX"].T,delta)
    print(delta.shape)

    '''delta = np.multiply(delta, l1.local_grad["dZ"])
    opt.optimize(l1, delta)'''

'''print(l1.weights)
print(l1.bias)'''
print(l2.weights)
print(l2.bias)
# print(l2(l1(np.array([[5,5]]))))

