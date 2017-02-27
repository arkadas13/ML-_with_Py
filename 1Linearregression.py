import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
input = np.array([
    [10 + float(np.random.rand(1, 1)), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1)],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1)],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1)],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1)],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1)],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1)],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1)],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1)],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1)],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1)],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1)],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1)],
])
m = len(input)
X = np.array([np.ones(m), input[:,0], input[:,1]]).T
y = np.vstack(np.array(input[:, -1]))
betaHat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
fig=plt.figure(1)
xx = input[:,0]
yy = input[:,1]
zz = np.array(betaHat[0] + betaHat[1] * xx + betaHat[2] * yy)
ax = fig.gca(projection='3d')
ax.scatter(xx, yy, zz, color='r', marker='o')
ax.scatter(input[:,0],input[:,1],input[:,2])
ax.legend()
plt.show()