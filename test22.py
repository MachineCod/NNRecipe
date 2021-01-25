import numpy as np
from nn_recipe.NN.Layers.pooling import*
from nn_recipe.NN.Layers.conv import Conv2D
from nn_recipe.NN.Layers.linear import Linear
from nn_recipe.NN.ActivationFunctions.sigmoid import Sigmoid
from nn_recipe.NN.LossFunctions import *
from nn_recipe.NN.Layers.flatten import Flatten
from nn_recipe.NN.network_CNN import NetworkCNN
from nn_recipe.DataLoader.mnistDataLoader import MNISTDataLoader
from nn_recipe.Opt import GD
from nn_recipe.utility import OneHotEncoder
from PIL import Image
import time
import os


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
"""
# fig, ax = plt.subplots(nrows=0,ncols=6)
ex = np.arange(0, 16).reshape((4,4,1))
fltr = np.array([[0,1,0],[0,1,0],[0,1,0]]).reshape((1,3,3,3))
#print(fltr.shape)
#dL = np.array([[-4,-2],[3,-5]]).reshape((1,2,2,1))
# print(fltr.shape)
conv1 = Conv2D(inChannels=1, filters=1, filters_values=fltr, padding="VALID")
conv_out = conv1(ex)
print("weighs", conv1.weights)
print(conv_out, conv_out.shape)
print(conv1._calc_local_grad(dL))
# print("local grads", conv1.local_grad)
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
# img = Image.open(os.path.join('', 'andrew2.jpeg')).convert("RGB")
# print("img_size", np.array(img).shape)
# t2 = time.time()

# Y = np.array(["Andrew"]).reshape(-1,1)
# encoder = OneHotEncoder(
#     types=["Andrew", "Mariam"],
#     active_state=1,
#     inactive_state=0
# )
# gd = GD()
# encoded_Y = encoder.encode(Y)
# loss = MClassLogisticLoss()
"""working training
for epoch in range(10):
    print(f"epoch{epoch}")
    conv1 = Conv2D(inChannels=3, filters=3, filters_values=edge_fltr, padding="VALID")
    conv_out = conv1(np.array(img))    # temp
    # Image.fromarray(conv_out[0].astype(np.uint8) ,"RGB").show()
    print("conv_out", conv_out.shape)
    # pooling
    p1 = MaxPool2D(kernelSize=3, strides=2, padding="SAME")
    p1_out = p1(conv_out)
    # Image.fromarray(p1_out[0].astype(np.uint8) ,"RGB").show()
    print("pool_out", p1_out.shape)

    flat = Flatten()
    f_out = flat(p1_out)
    b, _, cols = f_out.shape
    print("flat_out", f_out.shape)
    l1 = Linear(in_dim=cols, out_dim=3, activation=Sigmoid())
    out = l1(f_out[0])
    print("Linear out", out.shape)

    loss_val = loss(encoded_Y, out)
    print("loss", loss_val)

    delta = loss.local_grad

    ""Linear layer weights update""
    delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
    gd.optimize(layer=l1, delta=delta)
    print("linear_grad out", delta.shape)

    delta = np.dot(delta.T, l1.local_grad["dX"])
    print("delta_linear", delta.shape)

    delta = flat.calc_local_grad(delta)
    print("delta_flatten", delta.shape)

    delta = p1.calc_local_grad(delta)
    print("pool_weights updates")
    delta = conv1.calc_local_grad(delta['dY'])
    print('delata_pool', delta['dY'].shape)

    print("conv_weights updates")
    gd.optimize(layer=conv1, delta=delta['dW'])
    print("delta_convW", delta['dW'].shape)
    print("delta_convY", delta['dY'].shape)
"""    
mnist = MNISTDataLoader(rootPath=os.path.join('','dataset'), valRatio=0, download=False)
mnist.load()
x_train, y_train = mnist.get_train_data(), mnist.get_train_labels()
x_test, y_test = mnist.get_test_data(), mnist.get_test_labels()
# print(x_train.shape)

# print(np.expand_dims(y_train[0:100], axis=0), y_train.reshape(-1,60000))

net = NetworkCNN(
    layers=[
        Conv2D(inChannels=28, filters=32, padding="SAME"),
        MaxPool2D(kernelSize=2, strides=2, padding="VALID"),
        Conv2D(inChannels=32, filters=64, padding="SAME"),
        MaxPool2D(kernelSize=2, strides=2, padding="VALID"),
        Flatten(),
        Linear(in_dim=3136, out_dim=128, activation=Sigmoid()),
        Linear(in_dim=128, out_dim=10, activation=Sigmoid())
    ],
    optimizer=GD(learning_rate=0.001),
    loss_function=MClassLogisticLoss(sum=True, axis=0),
    batch_size=1
)
loss, itr = net.train(x_train[0:100], np.expand_dims(y_train[0:100], axis=0), notify_func=print, batch_size=1, max_itr=10)

