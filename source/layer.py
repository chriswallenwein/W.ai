import numpy as np

# TODO add np.array-dimensions to docstrings

class Layer():
    """Base class for all neural network layers."""

    def __init__(self):
        return

    def forward(self, x: np.ndarray):
        """Computes the forward pass of the layer.
        
        Subclasses should implement this method.
        """
        raise NotImplementedError

    def backward(self, x: np.ndarray):
        """Computes the gradient of the layer.

        Subclasses should implement this method.
        """
        raise NotImplementedError

class StandardFullyConnected(Layer):
    """Fully connected layer"""

    def __init__(self, in_dim: int, out_dim: int):
        """Initializes the layer

        Initializes the weights of the fully connected layer.

        Arguments:
            in_dim: int > 0
                dimension of one input sample
            out_dim: int > 0
                dimension of one output sample
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = 2*np.random.rand(out_dim, in_dim) -1
        self.b = 2*np.random.rand(out_dim, 1) -1

    def forward(self, x: np.ndarray):
        """Computes the output of the layer.
        
        Computes forward pass of the fully connected layer for n samples

        Arguments:
            x: np.array
                n input samples

        Returns:
            np.array
                output of the layer
        """
        y_hat = self.w@x + self.b
        self.cache = x
        return y_hat

    def backward(self, x: np.ndarray):
        """Computes the gradient of the layer.
        
        Arguments:
            x: np.array
                n input samples

        Returns:
            gradient of the layer
        """
        return x

class FullyConnectedBiasTrick(Layer):
    """Fully connected layer with bias trick.

    Here's a simple explanation of the bias trick:
    https://hetpinvn.wordpress.com/2016/10/26/bias-trick/
    """

    def __init__(self, in_dim: int, out_dim: int):
        """Initializes the layer.

        Initializes the weights of the fully connected layer.

        Arguments:
            in_dim: int > 0
                dimension of one input sample
            out_dim: int > 0
                dimension of one output sample
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = 2*np.random.rand(out_dim, in_dim+1) -1

    def forward(self, x: np.ndarray):
        """Computes the output of the layer.
        
        Computes forward pass of the fully connected layer with the bias trick for n samples

        Arguments:
            x: np.array
                n input samples

        Returns:
            np.array
                output of the layer
        """
        ones = np.ones((1, x.shape[1]))
        x = np.concatenate((ones, x))
        self.cache = x
        y_hat = self.w@x # the bias trick
        return y_hat

    def backward(self, x: np.ndarray):
        """Computes the gradient of the layer.
        
        Arguments:
            x: np.array
                n input samples

        Returns:
            gradient of the layer
        """
        return x

FullyConnected = StandardFullyConnected