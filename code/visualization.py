import numpy as np
import matplotlib.pyplot as plt

# TODO: Allow visualizing more than 2 output classes
#   - Wait for softmax to be implemented correctly
#   - Use softmax after forward pass to allow interpreting output as probabilities
#   - Use imshow instead of contourf for plotting of the weights
#   - plt.imshow(predictions, origin="lower", extent=[-10,10,-10,10], cmap=plt.cm.RdYlGn)
#   - Each pixel is showing one color. The color is determined by the class with the highest probability for that pixel
#     then calculate max(class probabilities for that pixel) * color of class with highest probability


# TODO: Display current loss of model

class Visualization():
    def __init__(self, x_min=-10, x_max=10, y_min=-10, y_max=10):
        self.x_min = x_min
        self.x_max = x_max
        self. y_min = y_min
        self.y_max = y_max
        self.ax = plt.axes()
    
    def visualize_model(self, model, data_x, data_y):
        self.visualize_model_weights(model)
        self.visualize_data(data_x, data_y)

    def visualize_layer(self, layer, data_x, data_y):
        # The layer must have 2-dimensional input so it can be visualized in a 2D plot
        if layer.in_dim != 2:
            raise ValueError("Input dimensions of layer must equal 2")
        self.visualize_layer_weights(layer)
        self.visualize_data(data_x, data_y)

    def visualize_model_weights(self, model):
        pass
    
    def visualize_layer_weights(self, layer):
        xs = np.linspace(self.x_min, self.x_max, 250)
        ys = np.linspace(self.y_min, self.y_max, 250)
        mesh_input = np.array(np.meshgrid(xs, ys))
        # flatten mesh to allow forward pass
        mesh_input = mesh_input.reshape(2, 250*250)
        predictions = layer.forward(mesh_input)
        # undo flatten to for squared image
        predictions = predictions.reshape(250,250)
        self.ax.contourf(xs, ys, predictions, levels=100, cmap=plt.cm.RdYlGn)
        
    def visualize_data(self, data_x, data_y):
        scatterplot = self.ax.scatter(data_x[:, 0], data_x[:, 1], c=data_y)
        labels = list(np.unique(data_y))
        self.ax.legend(handles=scatterplot.legend_elements()[0], labels=labels)
