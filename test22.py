import numpy as np
from nn_recipe.NN.Layers.pooling import*
from nn_recipe.NN.Layers.conv import Conv2D
from nn_recipe.NN.Layers.linear import Linear
from nn_recipe.NN.ActivationFunctions.sigmoid import Sigmoid
from nn_recipe.NN.LossFunctions import *
# from nn_recipe.NN.Op import GD
from nn_recipe.NN.Layers.flatten import Flatten
from nn_recipe.NN import Network
from PIL import Image
import time


# x = np.array([
#     [1, 1],
#     [2, 3],
#     [4, 7],
#     [-1, -1],
#     [-7, -0.2],
#     [-0.1, -3]
# ])

# y = np.array([
#     [1],
#     [1],
#     [1],
#     [-10],
#     [-10],
#     [-10],
# ])

# l1 = Linear(in_dim=2, out_dim=1, activation=Sigmoid())
# l11 = Linear(in_dim=2, out_dim=1, activation=Sigmoid(), weights=np.copy(l1.weights), bias=np.copy(l1.bias))
# net = Network(
#     layers=[l11],
#     optimizer=GD(learning_rate=0.1),

# )
# loss, it_no = net.train(x, y, epsilon=0.1)
# print(loss)
# print(net.evaluate([-7, -0.2]))

# print("####################################################################################################")
# opt = GD(learning_rate=0.1)
# msl = MeanSquaredLoss()
# for a in range(5):
#     out = l1(x)
#     loss = msl(y, out)
#     print("{}".format(loss))
#     delta = msl.local_grad    # dL/dy (last layer)
#     # print("delta", delta)
#     delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
#     opt.optimize(l1, delta)
#     delta = np.dot(delta.T, l1.local_grad["dX"])
#
#     # delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
#     # opt.optimize(l1, delta)

# print(l1.weights)
# print(l1.bias)
# print(l2.weights)
# print(l2.bias)
# print(l2(l1(np.array([[5,5]]))))
from nn_recipe.Opt import GD
from nn_recipe.utility import OneHotEncoder

"""
# fig, ax = plt.subplots(nrows=0,ncols=6)
ex = np.arange(0, 16).reshape((4,4,1))
fltr = np.array([[0,1,0],[0,1,0],[0,1,0]]).reshape((1,3,3,3))
#print(fltr.shape)
#dL = np.array([[-4,-2],[3,-5]]).reshape((1,2,2,1))
# print(fltr.shape)
conv_ex = Conv2D(inChannels=1, filters=1, filters_values=fltr, padding="VALID")
conv_out = conv_ex(ex)
print("weighs", conv_ex.weights)
print(conv_out, conv_out.shape)
print(conv_ex._calc_local_grad(dL))
# print("local grads", conv_ex.local_grad)
# for i in range(1, conv_out.shape[3]+1):
#     plt.subplot(1, 6, i)
#     plt.imshow(conv_out[0, :, :, i-1])
# plt.imshow(conv_out[0])
# plt.show()
"""
#   ------------------------------------------------

# ex = np.arange(0, 16).reshape((4,4))
# temp = np.empty((1, 4, 4, 3))
# for i in range(3):
#     temp[0, :, :, i] = ex
fltr = np.array([[0,0,0],[0,1,0],[0,0,0]]).reshape((1,3,3,1))
sobel_fltr = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).reshape((1,3,3,1))
blur_fltr = np.ones((1,3,3,1))/9
edge_fltr = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((1,3,3,1))
img = Image.open(r'C:\\Users\\mgtmP\\Desktop\\test.png').convert("RGB")


conv_ex = Conv2D(inChannels=3, filters=3, padding="VALID")
t2 = time.time()
conv_out = conv_ex(np.array(img))    # temp
Image.fromarray(conv_out[0].astype(np.uint8) ,"RGB").show()

# pooling
p1 = MaxPool2D(kernelSize=3, strides=2, padding="VALID")
p1_out = p1(conv_out)
Image.fromarray(p1_out[0].astype(np.uint8) ,"RGB").show()
flat = Flatten()
f_out = flat(p1_out)
b, _, cols = f_out.shape
# print (s1*s2*n_c)
l1 = Linear(in_dim=cols, out_dim=3, activation=Sigmoid())
print('time taken', time.time()-t2)
out = l1(f_out[0])

Y = np.array(["Andrew"]).reshape(-1,1)
encoder = OneHotEncoder(
    types=["Andrew", "Mariam"],
    active_state=1,
    inactive_state=0
)
gd = GD()
encoded_Y = encoder.encode(Y)

loss = MClassLogisticLoss()
loss_val = loss(encoded_Y, out)

delta = loss.local_grad
delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
gd.optimize(layer=l1, delta=delta)
delta = np.dot(delta.T, l1.local_grad["dX"])
print(delta)
# print(l1(f_out[0]))

