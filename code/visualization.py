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
# TODO: Add 3D plot for 3D input data

class Visualize():

    @staticmethod
    def model(model, features, labels):
        if model.layers[0].in_dim != 2:
            raise ValueError("Input dimensions of layer must equal 2")
        
        boundaries = Visualize.calculate_boundaries(features)
        ax = plt.axes()
        Visualize.visualize_model_weights(ax, model, boundaries)
        Visualize.visualize_data(ax, features, labels)
        plt.xlabel("x_1")
        plt.ylabel("x_2")

    @staticmethod
    def visualize_model_weights(ax, model, boundaries):
        x_min, x_max, y_min, y_max = boundaries
        xs = np.linspace(x_min, x_max, 250)
        ys = np.linspace(y_min, y_max, 250)
        mesh_input = np.array(np.meshgrid(xs, ys))
        # flatten mesh to allow forward pass
        mesh_input = mesh_input.reshape(2, 250*250)
        predictions = model.forward(mesh_input)
        # undo flatten to for squared image
        predictions = predictions.reshape(250,250)
        contourplot = ax.contourf(xs, ys, predictions, levels=100, cmap=plt.cm.RdYlGn)
        
        # add legend to data
        plt.colorbar(contourplot, label= "y_hat") 

        
    
    @staticmethod
    def visualize_data(ax, data_x, data_y):
        scatterplot = ax.scatter(data_x[:, 0], data_x[:, 1], c=data_y)

        # add legend
        labels = list(np.unique(data_y))
        ax.legend(handles=scatterplot.legend_elements()[0], labels=labels, title="y")

    @staticmethod
    def calculate_boundaries(features):
        x = features[:, 0]
        y = features[:, 1]
        
        x_std = np.std(x)
        y_std = np.std(y)
        
        x_min = np.min(x) - x_std
        y_min = np.min(y) - y_std
        
        x_max = np.max(x) + x_std
        y_max = np.max(y) + y_std

        boundaries = (x_min, x_max, y_min, y_max)
        return boundaries
        
    @staticmethod
    def display_loss():
        pass