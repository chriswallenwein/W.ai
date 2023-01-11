# TODO: check if the dimensions of the different layers fit together

class Model():
    """Neural network model, contains multiple layers"""

    def __init__(self, layers):
        """Initializes the model with a list of layers

        Arguments:
            layers: List[Layer]
                List of all layers of the model
        """
        self.layers = layers

    def forward(self, x):
        """Computes output of the entire model

        Iterates through the layers of the model. It computes the output of each layer
        and uses it as the input for the next layer.
        
        Arguments:
            x: np.array
                Model input

        Returns:
            np.array
                Output of the last layer
        """
        result = x
        for layer in self.layers:
            result = layer.forward(result)
        return result