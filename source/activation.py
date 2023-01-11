import numpy as np

class Activation():
    """Base class for neural network activation functions"""
    def __init__(self):
        return

    def forward(self, x: np.ndarray):
        """Computes the output of the activation function.

        Subclasses should implement this method.
        """
        raise NotImplementedError

    def backward(self, x: np.ndarray):
        """Computes the gradient of the activation function.

        Subclasses should implement this method.
        """
        raise NotImplementedError
    
# class Softmax(Activation):

#     def __init__(self):
#         return

#     def forward(self, x: np.ndarray):
#         x = np.exp(x)
#         total = np.sum(x, axis=0)
#         return x/total

#     def backward(self, x: np.ndarray):
#         return

class ReLU(Activation):
    """Rectified linear unit activation function"""

    def forward(self, x: np.ndarray):
        """Computes the output of the ReLU activation function.

        Arguments:
            x: np.array
                input sample

        Returns
            np.array
                output of ReLU
        """
        self.cache = x
        return np.maximum(0, x)

    def backward(self):
        """Computes the gradient of the ReLU activation function.
        
        Computes the local gradient of the ReLU activation function using the cached input during the forward pass.
        """
        return np.heaviside(self.cache, 1)

class Sigmoid(Activation):
    """Sigmoid activation function"""

    def forward(self, x: np.ndarray):
        """Computes the output of the Sigmoid activation function.

        Arguments:
            x: np.array
                input sample

        Returns
            np.array
                output of Sigmoid
        """
        return 1/(1+np.exp(-x))

class TanH(Activation):
    """TanH activation function"""

    def forward(self, x: np.ndarray):
        """Computes the output of the TanH activation function.

        Arguments:
            x: np.array
                input sample

        Returns
            np.array
                output of TanH
        """
        return (np.exp(x)-np.exp(-x))/(np.exp(-x)+np.exp(x))