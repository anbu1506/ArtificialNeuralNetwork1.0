import numpy as np
y_pred = np.random.rand(3,1)
y = np.ones((3,1))

m=3
l=np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
print(l)
cost = - (1 / m) * l
print(cost)