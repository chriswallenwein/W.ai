import numpy as np

# TODO add np.array-dimensions to docstrings

class Cost():
    """Base class for all cost functions."""

    def __init__(self):
        return

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Computes the cost.
        
        Subclasses should implement this method.
        """
        raise NotImplementedError

    def backward(self, y: np.ndarray, y_hat: np.ndarray):
        """Computes the gradient of the cost function.

        Subclasses should implement this method.
        """
        raise NotImplementedError

class L1(Cost):
    """L1 cost"""

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Computes the L1 cost.

        Computes the average elementwise difference between y and y_hat.

        Arguments:
            y: np.array()
                The ground truth label
            y_hat: np.array()
                The predicted label
        
        Returns:
            scalar cost
        """
        self.y_cache = y
        self.y_hat_cache = y_hat
        elementwise_loss = np.absolute(y_hat - y)
        avg_cost = np.average(elementwise_loss)
        return avg_cost

    def backward(self):
        """Computes gradient of the L1 cost.

        Computes the local gradient of the L1 cost function using the cached y and y_hat.

        Returns:
            gradient of L1 cost
        """
        cost_gradient = 1 / self.y_cache.size
        loss_gradient = np.sign(self.y_hat_cache - self.y_cache)
        return cost_gradient * loss_gradient

class L2(Cost):
    """L2 cost"""

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Computes the L2 cost.

        Computes the average squared elementwise difference between y and y_hat.

        Arguments:
            y: np.array()
                The ground truth label
            y_hat: np.array()
                The predicted label

        Returns:
            scalar cost
        """
        self.y_cache = y
        self.y_hat_cache = y_hat
        elementwise_loss = np.square(y_hat - y)
        avg_cost = np.average(elementwise_loss)
        return avg_cost

    def backward(self):
        """Computes gradient of the L2 cost.
        Computes the local gradient of the L1 cost function using the cached y and y_hat.

        Returns:
            gradient of L2 cost
        """
        cost_gradient = 1 / self.y_cache.size
        loss_gradient = 2 * (self.y_hat_cache - self.y_cache)
        return cost_gradient * loss_gradient