import numpy as np

class Activation():
    def __init__(self):
        return

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError
    
class Softmax(Activation):

    def __init__(self):
        return

    def forward(self, x):
        x = np.exp(x)
        total = np.sum(x, axis=0)
        return x/total

    def backward(self, x):
        return

class ReLU(Activation):
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self):
        return np.heaviside(self.cache, 1)

class Sigmoid(Activation):
    def forward(self, x):
        return 1/(1+np.exp(-x))

class TanH(Activation):
    def forward(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(-x)+np.exp(x))