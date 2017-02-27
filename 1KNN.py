import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10
arr = np.array([
    [10, 10, 10, 1],
    [20, 20, 20, 2],
    [30, 30, 30, 3],
    [40, 40, 40, 4],
    [10+np.random.rand(1,1), 10+np.random.rand(1,1), 10+np.random.rand(1,1), 1],
    [20+np.random.rand(1,1), 20+np.random.rand(1,1), 20+np.random.rand(1,1), 2],
    [30+np.random.rand(1,1), 30+np.random.rand(1,1), 30+np.random.rand(1,1), 3],
    [40+np.random.rand(1,1), 40+np.random.rand(1,1), 40+np.random.rand(1,1), 4],
    [10+np.random.rand(1,1), 10+np.random.rand(1,1), 10+np.random.rand(1,1), 1],
    [20+np.random.rand(1,1), 20+np.random.rand(1,1), 20+np.random.rand(1,1), 2],
    [30+np.random.rand(1,1), 30+np.random.rand(1,1), 30+np.random.rand(1,1), 3],
    [40+np.random.rand(1,1), 40+np.random.rand(1,1), 40+np.random.rand(1,1), 4],
    [10+np.random.rand(1,1), 10+np.random.rand(1,1), 10+np.random.rand(1,1), 1],
    [20+np.random.rand(1,1), 20+np.random.rand(1,1), 20+np.random.rand(1,1), 2],
    [30+np.random.rand(1,1), 30+np.random.rand(1,1), 30+np.random.rand(1,1), 3],
    [40+np.random.rand(1,1), 40+np.random.rand(1,1), 40+np.random.rand(1,1), 4],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 1],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 2],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 3],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 4],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 1],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 2],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 3],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 4],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 1],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 2],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 3],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 4],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 1],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 2],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 3],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 4],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 1],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 2],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 3],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 4],
    [10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 10 + np.random.rand(1, 1), 1],
    [20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 20 + np.random.rand(1, 1), 2],
    [30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 30 + np.random.rand(1, 1), 3],
    [40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 40 + np.random.rand(1, 1), 4],
])
k=15
noofclass=4
unknownpt=np.array([36,36,0])
distances=np.zeros((len(arr)),dtype=float)
for i in range(len(arr)):
    sum=0
    for j in range(len(arr[0])-1):
        sum+=pow((unknownpt[j]-arr[i,j]),2)
    distances[i]=math.sqrt(sum)

pos=np.ones(k,dtype=int)*-1
classcount=np.zeros(noofclass,dtype=int)
max,index=0,0
for i in range(k):
    min=9999999
    for j in range(len(arr)):
        if min>distances[j] and j not in pos:
            min=distances[j]
            pos[i]=j
    classcount[int(arr[pos[i], -1] - 1)] += 1
    if(classcount[int(arr[pos[i], -1] - 1)]>max):
        max=classcount[int(arr[pos[i], -1] - 1)]
        index=arr[pos[i], -1]
print("Max Probability of unknown point to be of class:",index)
fig=plt.figure(1)
ax = fig.gca(projection='3d')
ax.scatter(arr[:,0], arr[:,1], arr[:,2], color='r', marker='o')
ax.scatter(unknownpt[0],unknownpt[1],unknownpt[2], color='b', marker='+')
ax.legend()
plt.show()