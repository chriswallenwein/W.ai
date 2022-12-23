import numpy as np

# average cost over entire tensor
class Cost():
    # func: e.g. np.sum, np.average
    def __init__(self):
        return

    def forward(self, y, y_hat):
        raise NotImplementedError

    def backward(self, y, y_hat):
        raise NotImplementedError

class L1(Cost):
    
    def forward(self, y, y_hat):
        self.y_cache = y
        self.y_hat_cache = y_hat
        elementwise_loss = np.absolute(y_hat - y)
        avg_cost = np.average(elementwise_loss)
        return avg_cost

    def backward(self):
        cost_gradient = 1 / self.y_cache.size
        loss_gradient = np.sign(self.y_hat_cache - self.y_cache)
        return cost_gradient * loss_gradient

class L2(Cost):

    def forward(self, y, y_hat):
        self.y_cache = y
        self.y_hat_cache = y_hat
        elementwise_loss = np.square(y_hat - y)
        avg_cost = np.average(elementwise_loss)
        return avg_cost

    def backward(self):
        cost_gradient = 1 / self.y_cache.size
        loss_gradient = 2 * (self.y_hat_cache - self.y_cache)
        return cost_gradient * loss_gradient