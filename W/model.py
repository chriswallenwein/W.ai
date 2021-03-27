import numpy as np

class Model():
    def __init__(self, layers):
        self.layers = layers
        return

    def forward(self, x):
        result = x
        for layer in self.layers:
            result = layer.forward(result)
        return result