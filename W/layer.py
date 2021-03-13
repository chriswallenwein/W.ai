import numpy as np

class Layer():
    def __init__(self):
        return

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

class FullyConnectedBiasTrick(Layer):

    def __init__(self, in_dim, out_dim):
        self.w = 2*np.random.rand(out_dim, in_dim+1) -1

    def forward(self, x):
        ones = np.ones((1, x.shape[1]))
        x = np.concatenate((ones, x))
        y_hat = self.w@x # the bias trick
        return y_hat

    def backward(self, x):
        return x

class FullyConnectedNormal(Layer):

    def __init__(self, in_dim, out_dim):
        self.w = 2*np.random.rand(out_dim, in_dim) -1
        self.b = 2*np.random.rand(out_dim, 1) -1

    def forward(self, x):
        y_hat = self.w@x + self.b
        return y_hat

    def backward(self, x):
        return x

FullyConnected = FullyConnectedBiasTrick

class Softmax(Layer):

    def __init__(self):
        return

    def forward(self, x):
        x = np.exp(x)
        total = np.sum(x, axis=0)
        return x/total

    def backward(self, x):
        return