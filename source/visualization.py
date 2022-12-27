import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

# TODO: Allow visualizing more than 2 output classes
#   - Wait for softmax to be implemented correctly
#   - Use softmax after forward pass to allow interpreting output as probabilities
#   - Use imshow instead of contourf for plotting of the weights
#   - plt.imshow(predictions, origin="lower", extent=[-10,10,-10,10], cmap=plt.cm.RdYlGn)
#   - Each pixel is showing one color. The color is determined by the class with the highest probability for that pixel
#     then calculate max(class probabilities for that pixel) * color of class with highest probability
# TODO: Add 3D plot for 3D input data


class Visualization2dBuilder():

    def setup(self, boundaries):
        # crete new colormap based on cm.coolwarm (but with lower alpha)
        light_coolwarm = plt.cm.coolwarm(np.linspace(0.1, 0.9, 100))
        light_coolwarm[:,3] = light_coolwarm[:,3] * 0.7
        light_coolwarm = LinearSegmentedColormap.from_list("light-coolwarm", light_coolwarm)
        self.colormap = light_coolwarm
        
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.set_box_aspect(1)
        self.ax.set_xlabel("$x_1$")
        self.ax.set_ylabel("$x_2$", rotation="horizontal")
        
        self.boundaries = boundaries
        self.plots = []
        return self

    def add_model(self, forward_function):
        x1_min, x1_max, x2_min, x2_max = self.boundaries
        x1 = np.linspace(x1_min, x1_max, 250)
        x2 = np.linspace(x2_min, x2_max, 250)
        mesh_input = np.array(np.meshgrid(x1, x2))
        # flatten mesh to allow forward pass
        mesh_input = mesh_input.reshape(2, 250*250)
        predictions = forward_function(mesh_input)
        # undo flatten to for squared image
        predictions = predictions.reshape(250,250)
        model_weight_plot = self.ax.contourf(x1, x2, predictions, levels=100, cmap=self.colormap, zorder=-10)
        self.plots.append(model_weight_plot)
        return self

    def add_data(self, x, y):
        data_plot = self.ax.scatter(x[0], x[1], c=y, edgecolors="white", cmap=self.colormap, zorder=-5)
        self.plots.append(data_plot)
        return self
    
    def add_legend(self):
        if len(self.plots) >= 1:
            ticks = np.linspace(0, 1, 11, endpoint=True)
            plt.colorbar(self.plots[0], ticks=ticks) #, label= "y_hat") 
            
            transparent = (0,0,0,0)
            color1 = (0.35, 0.47, 0.89, 0.7)
            color2 = (0.84, 0.32, 0.26, 0.7)

            pa1 = Patch(facecolor=color1)
            pa2 = Patch(facecolor=color2)
            pa3 = Line2D([], [], marker='o', color=transparent,
                markerfacecolor=color1, markeredgecolor="white")
            pa4 = Line2D([], [], marker='o', color=transparent,
                markerfacecolor=color2, markeredgecolor="white")

            self.ax.legend(handles=[pa1, pa3, pa2, pa4],
                labels=["", "", "Model", "Data"],
                ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                loc='upper right', fontsize=12, facecolor=(1, 1,1 ,1))
        else:
            raise Exception("Please add the model or the data before the legend")
        
        return self
            
    def add_cost(self, cost_function, y, y_hat):
        cost = cost_function(y, y_hat).item()
        cost_text = "Cost: " + str(cost)
        at = AnchoredText(
            cost_text, prop=dict(size=15), loc='upper left')
        self.ax.add_artist(at)
        return self
    
class Visualization2dHelper():
    @staticmethod
    def calculate_boundaries(data):
        x1 = data[:, 0]
        x2 = data[:, 1]
        
        x1_std = np.std(x1)
        x2_std = np.std(x2)
        
        x1_min = np.min(x1) - x1_std
        x2_min = np.min(x2) - x2_std
        
        x1_max = np.max(x1) + x1_std
        x2_max = np.max(x2) + x2_std

        boundaries = (x1_min, x1_max, x2_min, x2_max)
        return boundaries