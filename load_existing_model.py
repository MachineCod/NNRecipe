from nn_recipe.NN import Network
import numpy as np
from nn_recipe.utility import OneHotEncoder
from nn_recipe.DataLoader import MNISTDataLoader


# Loading Data fro mnist Data set
mnist = MNISTDataLoader(rootPath="C:\\Users\\mgtmP\\Desktop\\NNRecipe\\mnist", download=False)
mnist.load()
X_test = mnist.get_test_data().reshape((-1, 28 * 28))
Y_test = mnist.get_test_labels().reshape((-1, 1))
X_test = X_test / 255

# Creating Label encoder
encoder = OneHotEncoder(
    types=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    active_state=1,
    inactive_state=0
)

# Mnist Dense layer Model construction
net = Network.load("C:\\Users\\mgtmP\\Desktop\\mnist_network_model_3_38.net")


# check for model accuracy using the test dataset
out = net.evaluate(X_test)
yhat = encoder.decode(out)
yhat = np.array(yhat).reshape((-1, 1))
print("Total Accuracy is :", 1 - np.count_nonzero(yhat - Y_test) / Y_test.shape[0])

