import numpy as np

def softmax(x):
    exp = np.exp(x)
    s = np.sum(exp)
    return exp/s


print(softmax(np.array([1,2,3])),
    softmax(np.array([1,2,3])/10),
    softmax(np.array([1,2,3])*10))
