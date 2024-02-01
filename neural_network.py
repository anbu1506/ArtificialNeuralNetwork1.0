import numpy as np


anglesBlue = np.random.uniform(0,2*np.pi,100)
radiusBlue = np.random.uniform(0,1,100)
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

    def __init__(self,input_layer_size,no_of_hidden_layers,hidden_layer_size) -> None:
        self.neurons_count.append(input_layer_size)
        for i in range(no_of_hidden_layers):
            self.neurons_count.append(hidden_layer_size)
        self.neurons_count.append(1)
        for i in range(len(self.neurons_count)-1):
            weights = np.random.rand(self.neurons_count[i+1],self.neurons_count[i])
            biases = np.zeros((self.neurons_count[i+1],1))
            self.weights[i+1] =weights
            self.biases[i+1] =biases
        print(self.weights,self.biases)

    def forward_propagation(self,input):
        self.forward_cache["a"] = {}
        self.forward_cache["z"] = {}
        self.forward_cache["a"][0] = input
        length = len(self.neurons_count)
        for i in range(1,length-1):
            self.forward_cache["z"][i] =  np.dot(self.weights[i],self.forward_cache["a"][i-1])+self.biases[i]
            self.forward_cache["a"][i] = np.tanh(self.forward_cache["z"][i])
        self.forward_cache["z"][length-1] =  np.dot(self.weights[length-1],self.forward_cache["a"][length-2])+self.biases[length-1]
        self.forward_cache["a"][length-1] =1/(1+ np.exp(-self.forward_cache["z"][length-1]))

    def cost_function(self,y, y_pred):
        m = y.shape[0]  # Number of examples
        epsilon = 1e-15  # Small value to avoid taking the log of zero

        # Clip predicted values to avoid numerical instability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Binary cross-entropy formula
        cost = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        print("the cost is:",cost)
        return cost
    
    def back_propogation(self,y): # y is the actual label

        length = len(self.neurons_count)
        self.derivatives["dz"] = {}
        self.derivatives["db"] = {}
        self.derivatives["dw"] = {}

        m = y.shape[0]


        self.derivatives["dz"][length-1] = self.forward_cache["a"][length-1] - y

        self.derivatives["dw"][length-1] = (1/m)*np.dot(self.derivatives["dz"][length-1],self.forward_cache["a"][length-2].T)

        self.derivatives["db"][length-1] = np.mean(self.derivatives["dz"][length-1],axis=1,keepdims=True)

        for i in range(length-2,0,-1):
            self.derivatives["dz"][i] = np.dot(self.weights[i+1].T,self.derivatives["dz"][i+1])*(1-np.tanh(self.forward_cache["z"][i])**2)

            self.derivatives["dw"][i] = (1/m)*(np.dot(self.derivatives["dz"][i],self.forward_cache["a"][i-1].T))

            self.derivatives["db"][i] = np.mean(self.derivatives["dz"][i],axis=1,keepdims=True)


    def update_params(self,learning_rate):
        for i in range(1,len(self.weights)+1):
            self.weights[i] -= learning_rate * self.derivatives["dw"][i]
            self.biases[i] -= learning_rate * self.derivatives["db"][i]
        
    def fit(self,data,label,epoch,learning_rate):

        for i in range(epoch):
            self.forward_propagation(data.T*data.T)
            self.cost_function(label,self.forward_cache["a"][len(self.neurons_count)-1])
            self.back_propogation(label)
            self.update_params(learning_rate)

    def predict(self,data):
        forward_cache={}
        forward_cache["a"] = {}
        forward_cache["z"] = {}
        forward_cache["a"][0] = data
        for i in range(1,len(self.neurons_count)-1):
            forward_cache["z"][i] =  np.dot(self.weights[i-1],forward_cache["a"][i-1])+self.biases[i-1]
            forward_cache["a"][i] = np.tanh(forward_cache["z"][i])
        forward_cache["z"][len(self.neurons_count)-1] =  np.dot(self.weights[len(self.neurons_count)-2],forward_cache["a"][len(self.neurons_count)-2])+self.biases[len(self.neurons_count)-2]
        forward_cache["a"][len(self.neurons_count)-1] =1/(1+ np.exp(-forward_cache["z"][len(self.neurons_count)-1]))
        return forward_cache
    
model = NN(2,2,2)
model.fit(data,label,200,0.3)