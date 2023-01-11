import numpy as np


class Cost():
    """Base class for all cost functions.
    """

    def __init__(self):
        return

    def forward(self, y, y_hat):
        """Forward pass of the cost function.
        
        Subclasses should implement this method
        """
        raise NotImplementedError

    def backward(self, y, y_hat):
        """
        Backward propagation of the cost function.

        Subclasses should implement this method
        """
        raise NotImplementedError

class L1(Cost):
    """
    L1 cost function
    """
    def forward(self, y, y_hat):
        """Forward pass of L1 cost.

        Computes the average elementwise difference between y and y_hat.

        Args:
            y:
                The ground truth label
            y_hat:
                The predicted label
        """
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