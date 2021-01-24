from nn_recipe.NN.ActivationFunctions import *
from nn_recipe.NN.LossFunctions import *
from nn_recipe.NN.Layers import *
from nn_recipe.Opt import *
from nn_recipe.NN import Network
import numpy as np
from nn_recipe.utility import OneHotEncoder
from nn_recipe.DataLoader.mnistDataLoader import MNISTDataLoader

net = Network(
    layers=[
        Linear(in_dim=784, out_dim=25, activation=ReLU()),
        Linear(in_dim=25, out_dim=10, activation=Identity())
    ],
    optimizer=GDAdaGrad(learning_rate=0.01),
    loss_function=MClassLogisticLoss(sum=True, axis=0),
)

net.save("C:\\Users\\mgtmP\\Desktop\\mnist_net.net")
mnist = MNISTDataLoader(rootPath="C:\\Users\\mgtmP\\Desktop\\NNRecipe\\mnist")
mnist.load()
X = mnist.get_train_data().reshape((-1, 28*28))
Y = mnist.get_train_labels().reshape((-1, 1))
X_validate = mnist.get_validation_data().reshape((-1, 28*28))
Y_validate = mnist.get_validation_labels().reshape((-1, 1))
X_test = mnist.get_test_data().reshape((-1, 28*28))
Y_test = mnist.get_test_labels().reshape((-1, 1))
X = X / 255
X_validate = X_validate / 255
X_test = X_test / 255


encoder = OneHotEncoder(
    types=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    active_state=1,
    inactive_state=0
)
y_encoded = encoder.encode(Y)


def verify():
    f = net.evaluate(X_validate)
    y_hat = encoder.decode(f)
    y_hat = np.array(y_hat).reshape((-1, 1))
    return np.count_nonzero(y_hat - Y_validate)


net.train(X, y_encoded, notify_func=None, batch_size=100, max_itr=50, verify_func=verify)

# net = Network.load("C:\\Users\\mgtmP\\Desktop\\mnist_net.net")
out = net.evaluate(X_test)
yhat = encoder.decode(out)
yhat = np.array(yhat).reshape((-1, 1))
print("Total Accuracy is :", 1-np.count_nonzero(yhat - Y_test)/Y_test.shape[0])
