import numpy as np
input = np.array([3,2,3]).reshape(3,1)
z = np.ones((3,3))
print(z,input)
print(np.dot(z,input))


