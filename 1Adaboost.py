import numpy as np
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
input=np.array([])

noofclassifiers=10
maximum,minimum,midpoint=np.zeros(len(input[0])-1,dtype=float), np.zeros(len(input[0])-1,dtype=float), np.zeros(len(input[0])-1,dtype=float)
decisionstumps,weights=np.zeros((2,len(input[0])-1), dtype=int),np.ones(len(input),dtype=float)*1/len(input)

for i in range(len(input[0])-1):
    maximum[i]=max(input[:,i])
    minimum[i]=min(input[:,i])
    midpoint[i]=(maximum[i]-minimum[i])/2
    countpositiveless,countnegativeless,countpositivemore,countnegativemore,=0,0,0,0
    for j in range(len(input)):
        if midpoint[i]<=input[i,j]:
            if input[j,-1]==-1:
                countnegativeless+=1
            else:
                countpositiveless+=1
        else:
            if input[j,-1]==-1:
                countnegativemore+=1
            else:
                countpositivemore+=1
    if(countpositivemore+countnegativeless>countnegativemore+countpositiveless):
        decisionstumps[0,i],decisionstumps[1,i]=-1,1
    else:
        decisionstumps[0,i], decisionstumps[1,i] = 1, -1

markclsfr,alpha,clsfrlocstore,markmissclsfdpts=np.zeros(len(input[0])-1, dtype=int),np.zeros(noofclassifiers,dtype=float),\
                                               np.zeros(noofclassifiers,dtype=int),np.zeros(len(input),dtype=int)

for i in range(noofclassifiers):
    minweightederror, pos =99999,0
    for j in range(len(input[0])-1):
        if markclsfr[j]==0:
            tempmarkmisclsfdpts,weightederror=np.ones(len(input),dtype=int), np.zeros(len(input[0])-1, dtype=float)
            for k in range(len(input)):
                if input[k,j]<=midpoint[j]:
                    if decisionstumps[0,j]!=input[k,-1]:
                        weightederror[j]+=weights[k]
                        tempmarkmisclsfdpts[k]=-1
                else:
                    if decisionstumps[1,j]!=input[k,-1]:
                        weightederror[j]+=weights[k]
                        tempmarkmisclsfdpts[k] =-1
            if minweightederror>weightederror[j]:
                minweightederror,pos,markmissclsfdpts=weightederror[j],j,tempmarkmisclsfdpts
    markclsfr[pos],clsfrlocstore[i],alpha[i]=1,pos,0.5*math.log((1-minweightederror)/minweightederror,base=math.e)
    sum=0
    for j in range(len(input)):
        weights[j]=weights[j]*pow(math.e,-alpha[i]*markmissclsfdpts[j])
        sum+=weights[j]
    weights/=sum