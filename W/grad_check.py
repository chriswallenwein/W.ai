
import numpy as np
import random
import loss, cost, activation

def check_loss(loss:loss.Loss, epsilon=1e-5, iterations=10):

    differences = list()

    for _ in range(iterations):
        random_y = 100 * np.random.randn()
        random_y = round(random_y,5)

        random_y_hat = 100 * np.random.randn()
        random_y_hat = round(random_y_hat,5)

        loss.forward(random_y,random_y_hat)
        calculated_derivative = loss.backward()

        plus = loss.forward(random_y,random_y_hat+epsilon)
        minus = loss.forward(random_y,random_y_hat-epsilon)

        estimated_derivative = (plus - minus)/(2*epsilon)
        difference = calculated_derivative - estimated_derivative
        differences.append(abs(difference))

    print("epsilon: ", epsilon)
    print("biggest difference: ", max(differences))
    return

def check_cost(cost:cost.Cost, epsilon=1e-5, shape=(5,5)):
    random_y = 100 * np.random.randn(*shape)
    random_y_hat = 100 * np.random.randn(*shape)

    differences = list()

    cost.forward(random_y,random_y_hat)
    calculated_derivative = cost.backward()

    for i in range(random_y.shape[0]):
        for j in range(random_y.shape[1]):
                    
            y_hat_plus = random_y_hat.copy()
            y_hat_plus[i,j] += epsilon

            y_hat_minus = random_y_hat.copy()
            y_hat_minus[i,j] -= epsilon
            
            cost_plus = cost.forward(random_y,y_hat_plus)
            cost_minus = cost.forward(random_y,y_hat_minus)

            estimated_derivative = (cost_plus-cost_minus)/(2*epsilon)
            difference = calculated_derivative[i,j] - estimated_derivative
            differences.append(difference)

    print("epsilon: ", epsilon)
    print("biggest difference: ", max(differences))
    return

def check_elementwise_activation(activation:activation.Activation, epsilon=1e-5, iterations=10):
    # This only works for elementwise activation functions like ReLU or Sigmoid,
    # but not for activation functions where one output is dependant on multiple inputs
    
    differences = list()

    for _ in range(iterations):
        random_x = 100 * np.random.randn()
        random_x = round(random_x,5)

        activation.forward(random_x)
        calculated_derivative = activation.backward()

        plus = activation.forward(random_x+epsilon)
        minus = activation.forward(random_x-epsilon)

        estimated_derivative = (plus - minus)/(2*epsilon)
        difference = calculated_derivative - estimated_derivative
        differences.append(abs(difference))

    print("epsilon: ", epsilon)
    print("biggest difference: ", max(differences))
    return
