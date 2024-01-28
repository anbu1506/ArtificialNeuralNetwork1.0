import numpy as np


anglesBlue = np.random.uniform(0,2*np.pi,100)
radiusBlue = np.random.uniform(0,1,100)
# radiusBlue = 3
x1 = radiusBlue * np.cos(anglesBlue)
y1 = radiusBlue * np.sin(anglesBlue)

anglesRed = np.random.uniform(0,2*np.pi,50)
radiusRed = np.random.uniform(1,2,50)
x2 = radiusRed * np.cos(anglesRed)
y2 = radiusRed * np.sin(anglesRed)


x1=np.concatenate((x1,x2))
y1=np.concatenate((y1,y2))
data = np.column_stack((x1,y1))
np.random.shuffle(data)
label = np.array([1 if np.sqrt(i**2+j**2) <=1 else 0 for i,j in data])

class NN:
    weights={}
    biases={}
    neurons_count = []

    forward_cache = {}

    derivatives = {}

    #  init and forward_propogation comibe to form a sequential neural network
    def __init__(self,input_layer_size,no_of_hidden_layers,hidden_layer_size) -> None:
        # self.input_layer_size = input_layer_size
        # self.no_of_hidden_layers = no_of_hidden_layers
        # self.hidden_layer_size = hidden_layer_size
        self.neurons_count.append(input_layer_size)
        for i in range(no_of_hidden_layers):
            self.neurons_count.append(hidden_layer_size)
        self.neurons_count.append(1)
        all_weights=[]
        all_biases=[]
        for i in range(len(self.neurons_count)-1):
            weights = np.random.rand(self.neurons_count[i+1],self.neurons_count[i])
            biases = np.zeros((self.neurons_count[i+1],1))
            self.weights[i] =weights
            self.biases[i] =biases
        print(self.weights,self.biases)

    def forward_propagation(self,input):
        self.forward_cache["a"] = {}
        self.forward_cache["z"] = {}
        self.forward_cache["a"][0] = input
        for i in range(1,len(self.neurons_count)-1):
            self.forward_cache["z"][i] =  np.dot(self.weights[i-1],self.forward_cache["a"][i-1])+self.biases[i-1]
            self.forward_cache["a"][i] = np.tanh(self.forward_cache["z"][i])
        self.forward_cache["z"][len(self.neurons_count)-1] =  np.dot(self.weights[len(self.neurons_count)-2],self.forward_cache["a"][len(self.neurons_count)-2])+self.biases[len(self.neurons_count)-2]
        self.forward_cache["a"][len(self.neurons_count)-1] =1/(1+ np.exp(-self.forward_cache["z"][len(self.neurons_count)-1]))
        print(self.forward_cache)
    
    def cost_function(self,y):  # y is the actual label
        m = y.shape[0]  
        epsilon = 1e-15  

        # Clip predicted values to avoid numerical instability
        self.forward_cache["a"][len(self.neurons_count)-1] = np.clip(self.forward_cache["a"][len(self.neurons_count)-1], epsilon, 1 - epsilon)

        # Binary cross-entropy formula
        cost = - (1 / m) * np.sum(y * np.log(self.forward_cache["a"][len(self.neurons_count)-1]) + (1 - y) * np.log(1 - self.forward_cache["a"][len(self.neurons_count)-1]))
        print("the cost is:",cost)
        return cost
    
    def back_propogation(self,y): # y is the actual label
        self.derivatives["dz"] = {}
        self.derivatives["db"] = {}
        self.derivatives["dw"] = {}

        m = y.shape[0]


        self.derivatives["dz"][len(self.neurons_count)-1] = self.forward_cache["a"][len(self.neurons_count)-1] - y

        self.derivatives["dw"][len(self.neurons_count)-1] = (1/m)*np.dot(self.forward_cache["z"][len(self.neurons_count)-1],self.forward_cache["a"][len(self.neurons_count)-2].T)

        self.derivatives["db"][len(self.neurons_count)-1] = np.mean(self.derivatives["dz"][len(self.neurons_count)-1],axis=1,keepdims=True)

        for i in range(len(self.neurons_count)-2,0,-1):
            self.derivatives["dz"][i] = self.forward_cache["a"][i] - y

            self.derivatives["dw"][i] = (1/m)*np.dot(self.forward_cache["z"][i],self.forward_cache["a"][i-1].T)

            self.derivatives["db"][i] = np.mean(self.derivatives["dz"][i],axis=1,keepdims=True)


    def update_params(self,learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.derivatives["dw"][i+1]
            self.biases[i] -= learning_rate * self.derivatives["db"][i+1]






# input = np.array([1,2,3]).reshape(3,1)
# label = np.array([1,0,1])
model = NN(2,2,3)
model.forward_propagation(data.T)
model.cost_function(label)
model.back_propogation(label)
model.update_params(0.3)
model.forward_propagation(data.T)
model.cost_function(label)
