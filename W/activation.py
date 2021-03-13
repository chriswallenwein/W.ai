import numpy as np

class Activation():
    def __init__(self):
        return

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError
    
class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.heavyside(x, 1)

class Sigmoid(Activation):
    def forward(self, x):
        return 1/(1+np.exp(-x))

class TanH(Activation):
    def forward(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(-x)+np.exp(x))