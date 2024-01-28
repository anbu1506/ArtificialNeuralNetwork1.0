# import numpy as np
# y_pred = np.random.rand(1,4)
# y = np.array([1,0,0,1])
# m=2
# l=np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
# print(l)


# u = np.array([[1,2,3,4]])
# print(u*y)









# a= np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a.shape)
# print(a)

# b = np.array([1,2,3])
# print(b.shape)
# print(b)

# c = np.array([1,2,3,4,5,6]).reshape(3,2)
# print(c.shape)
# print(c)

# print(a.dot(c)+b)


# input = np.array([1,2,3]).reshape(3,1)
# label = np.array([1,0,1])
model = NN(2,2,3)
model.forward_propagation(data.T)
model.cost_function(label)
model.back_propogation(label)
model.update_params(0.3)
model.forward_propagation(data.T)
model.cost_function(label)