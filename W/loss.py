import numpy as np

class Loss():
    def __init__(self):
        return

    def forward(self, y, y_hat):
        raise NotImplementedError

    def backward(self, y, y_hat):
        raise NotImplementedError

class L1(Loss):
    def forward(self, y, y_hat):
        self.y_cache = y
        self.y_hat_cache = y_hat
        elementwise_loss = np.absolute(y-y_hat)
        return elementwise_loss

    def backward(self, y, y_hat):
        local_gradient = np.sign(self.y_hat_cache - self.y_cache) 
        return local_gradient

class L2(Loss):
    def forward(self, y, y_hat):
        self.y_cache = y
        self.y_hat_cache = y_hat
        elementwise_loss = np.square(y_hat - y)
        return elementwise_loss

    def backward(self, y, y_hat):
        local_gradient = 2 * (self.y_hat_cache - self.y_cache)
        return local_gradient