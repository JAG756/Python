import numpy as np

arry = np.ones((3, 3))
arry[2,1] = 10
print(arry)
print(arry + 5)
print(np.sum(arry))
print(np.mean(arry))

a = np.array([[1, 2], [23, 4]])
b = np.array([[4, 6], [2, 8]])
print(a.dot(b))
