import numpy as np
import math
iter,laticerow,laticecolumn=100,4,4 #laticedepth=number of features

def laticedistribution(laticerow, laticecolumn, laticedepth):
    latice = np.random.rand(laticerow, laticecolumn, laticedepth)
    multiply = np.random.choice([-1, 1], (laticerow, laticecolumn, laticedepth), p=[0.5, 0.5])
    latice = latice * multiply
    return latice

def rad(maxiter, currentiter):
    radius=(max(laticerow,laticecolumn)/2)*(1-currentiter/maxiter)
    return radius

def learnrate(ilr,currentiter,maxiter,dist,radius):
    learn=math.exp(-pow(dist/radius,2)/2)*ilr*(1-currentiter/maxiter)
    return learn

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
inputclass= np.array([])
features,noofsam=len(input[0]),len(input)
maximum,minimum,inputscaled=np.zeros(len(input[0]),dtype=float),np.zeros(len(input[0]),dtype=float),np.zeros((len(input),len(input[0])),dtype=float)

for i in range(len(input[0])):
    maximum[i]=max(input[:,i])
    minimum[i]=min(input[:,i])
    for j in range(len(input)):
        inputscaled[j,i]=(input[j,i]-minimum[i])/(maximum[i]-minimum[i])

initiallr=1 #initial learning rate
latice=laticedistribution(laticerow,laticecolumn,len(input[0]))

for a in range(iter):
    for i in range(len(inputscaled)):
        mini = 999
        for j in range(len(latice)):
            for k in range(len(latice[0])):
                distance=0
                for l in range(len(latice[0,0])):
                    distance+=pow((inputscaled[i,l]-latice[j,k,l]),2)
                distance=math.sqrt(distance)
                if mini>distance:
                    mini=distance
                    pos=np.array([j,k]) #Best Matching Unit
        radius=rad(iter,a)
        for k in range(len(latice)):
            for l in range(len(latice[0])):
                dist=0.414*min(abs(pos[0]-k),abs(pos[1]-l))+max(abs(pos[0]-k),abs(pos[1]-l))
                if(dist<=radius):
                    lr = learnrate(initiallr, a, iter, dist, radius)
                    for m in range(len(latice[0][0])):
                        latice[k,l,m]=latice[k,l,m]+lr*(inputscaled[i,m]-latice[k,l,m])