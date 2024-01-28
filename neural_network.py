import numpy as np

class NN:
    weights=[]
    biases=[]
    neurons_count = []
    def __init__(self,input_layer_size,no_of_hidden_layers,hidden_layer_size) -> None:
        # self.input_layer_size = input_layer_size
        # self.no_of_hidden_layers = no_of_hidden_layers
        # self.hidden_layer_size = hidden_layer_size
        self.neurons_count.append(input_layer_size)
        for i in range(no_of_hidden_layers):
            self.neurons_count.append(hidden_layer_size)
        self.neurons_count.append(2)
        for i in range(len(self.neurons_count)-1):
            weights = np.random.rand(self.neurons_count[i+1],self.neurons_count[i])
            biases = np.zeros((self.neurons_count[i+1],1))
            self.weights.append(weights)
            self.biases.append(biases)

        print(self.biases,self.weights,self.neurons_count)



NN(3,2,3)