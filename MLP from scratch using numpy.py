#creating a multi-layer perceptron using Numpy:

import numpy as np

#The MLP class take 4 parameter self,input_size, hidden_size ,output_size,num_layers.
#The input_size parameter is used to initilize the weights of the MLP.
#weight:
#The weigths are matrices that transform the input data into the output
#The number of rows in the weight matrix corresponds to the number of elements in the input data
#the number of columns in the weight matrix corresponds to the number of nodes in the current layer of the MLP.
#The __init() method initializes the weight and biases of the MLP using NumPy's randn() function.
class MLP:
    def __init__(self,input_size, hidden_size ,output_size,num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.weights = []
        self.biases = []
        for i in range(num_layers):
            if i == 0:
                self.weights.append(np.random.randn(input_size, hidden_size))
                self.biases.append(np.random.randn(hidden_size))
            elif i == num_layers - 1:
                self.weights.append(np.random.randn(hidden_size, output_size))
                self.biases.append(np.random.randn(output_size))
            else:
                self.weights.append(np.random.randn(hidden_size, hidden_size))
                self.biases.append(np.random.randn(hidden_size))
    

# The forward() method takes in an input x and applies the MLP layers sequentially using NumPy's dot() function and the hyperbolic tangent activation function (np.tanh()).
    def forward(self, x):
        for i in range(self.num_layers): #iterating over the layers on MLP.
            if i == 0: #checking if the current layer is the 1st layer.
                z = np.dot(x, self.weights[i]) + self.biases[i] #if it is , 'x' is passed through linear transformation with 1st set of weigths and biases
            elif i == self.num_layers - 1: #check if the current layer is the last layer
                z = np.dot(z, self.weights[i]) + self.biases[i] #If it is, the output of the previous layer is passed through a linear transformation with the last set of weights and biases.
            else: #check if the current layer is in between the first and last layer
                z = np.dot(np.tanh(z), self.weights[i]) + self.biases[i] #If it is, the output of the previous layer is passed through a linear transformation with the current set of weights and biases, followed by a hyperbolic tangent activation function.
            if i < self.num_layers - 1: #check if the current lauer is not the last layer
                z = np.tanh(z) #f it is, the output of the current layer is passed through a hyperbolic tangent activation function before being passed through the next linear transformation.
        return z #returns the output of the final layer of MLP 
    
    
