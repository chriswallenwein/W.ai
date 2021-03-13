import numpy as np

# we need elementwise loss
# average cost
# summed up cost

class Cost():
    # func could be np.sum or np.average
    def __init__(self, func=lambda x:x):
        self.func = func
        return

    def forward(self, y, y_hat):
        raise NotImplementedError

    def backward(self, y, y_hat):
        raise NotImplementedError
    
class L1(Cost):
    
    def forward(self, y, y_hat):
        elementwise_loss = np.absolute(y-y_hat)
        cost = self.func(elementwise_loss)
        #loss = np.absolute(y-y_hat)
        #avg_cost = np.average(loss)
        return cost

    def backward(self, y, y_hat):
        local_gradient = np.sign(y_hat - y) 
        return local_gradient

#class L2(Cost):

    #def forward(self, y, y_hat):
        #cost = 
        #return cost

    #def backward(self, y, y_hat):
        #local_gradient = 
        #return local_gradient