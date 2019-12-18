"""
AUXILIARY FUNCTIONS DESCRIBING THE NEUTRON STAR + DARK MATTER SYSTEM

*****************************
********* EXAMPLE ***********
*****************************

*****************************
*****************************
"""

import numpy as np
import matplotlib
from decimal import Decimal
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import time
from scipy import integrate
from scipy.integrate import quad
from scipy import interpolate
from .ADMNSParameters import *

##########################
#### UPPER ODD NUMBER TO f
##########################
def odd(f):
    return np.ceil(f) // 2 * 2 + 1
###########################
### DECOMPOSITION OF F WITH 
### EXPONENT AND MANTISSA 
###########################
def fexp(f):
    return int(np.floor(np.log10(abs(f)))) if f != 0 else 0
def fman(f):
    return f/10**fexp(f)
########################################################
#### ROTATION MATRIX OF NORMAL (ux,uy,uz) OF ANGLE alpha
########################################################
def ROTMAT(ux,uy,uz,alpha):
    d,s=np.cos(alpha),np.sin(alpha)                      
    ROT=np.zeros(shape=(3,3))
    ### LIGN 1
    ROT[0][0]=(ux*ux*(1-d)+d)
    ROT[0][1]=ux*uy*(1-d)-uz*s
    ROT[0][2]=ux*uz*(1-d)+uy*s
    ### LIGN 2
    ROT[1][0]=ux*uy*(1-d)+uz*s
    ROT[1][1]=(uy*uy*(1-d)+d)
    ROT[1][2]=uy*uz*(1-d)-ux*s
    ### LIGN 3
    ROT[2][0]=ux*uz*(1-d)-uy*s
    ROT[2][1]=uy*uz*(1-d)+ux*s
    ROT[2][2]=(uz*uz*(1-d)+d)  
    return ROT
#####################################
### INDEXES OF POINTS ON SQUARE GRID
### EQUIVALENT BY ROTATION TO (i0,j0)
#####################################
def IJEquiv(i0,j0,imax):
    i1,j1=i0,j0
    i2,j2=imax-i0,j0
    i3,j3=imax-i0,imax-j0
    i4,j4=i0,imax-j0
    i5,j5=j0,i0
    i6,j6=imax-j0,i0
    i7,j7=imax-j0,imax-i0
    i8,j8=j0,imax-i0
    IJ=np.array([(i1,j1),(i2,j2),(i3,j3),(i4,j4),(i5,j5),(i6,j6),(i7,j7),(i8,j8)])
    IJ=np.unique(IJ,axis=0)
    return IJ
#################################
### RETURNS THE GRID WITH PIXELS
### ACCOUNTING FOR 100*p % ARE 
### HIGHLIGHTED (=10) WHILE OTHER
### NON ZERO ARE =1
#################################
def GridHighest(Grid,p):
    FlattenedGrid=Grid.flatten()
    SortArgFlat=np.argsort(FlattenedGrid)
    SortedFlattened=FlattenedGrid[SortArgFlat]
    starget=p*Grid.sum()
    f=0
    ngrid=Grid.shape[0]
    CumSum=np.cumsum(SortedFlattened[::-1])
    f=1+np.argmax((CumSum-starget)>0)
    #s=0
    #while s<starget:
    #    s+=SortedFlattened[len(SortedFlattened)-f-1]
    #    f+=1
    NewGrid=np.zeros(shape=(ngrid,ngrid))
    NewGrid[np.nonzero(Grid)]=1.
    #f(i,j)=i+j*ngrid
    #j(f)=int(f/ngrid)
    #i(f)=f-j(f)*ngrid
    F=SortArgFlat[len(SortedFlattened)-f:]
    J=(F/ngrid).astype("int")
    I=F-J*ngrid
    for k in range(len(I)):
        NewGrid[J[k]][I[k]] = 10
    return NewGrid
##################################
### RETURNS RADIUS OF CIRCLE
### WITHIN WHICH THE GRID CONTAINS 
### 100*p(=95)% OF THE TOTAL SIGNAL
##################################
def RP(Grid,Lgrid,p=0.95):
    im=int(0.5*(Grid.shape[0]-1))
    jm=im
    dx=Lgrid/(Grid.shape[0]-1)
    GridIndexes=np.zeros(shape=(Grid.shape[0],Grid.shape[0]))
    for i in range(Grid.shape[0]):
        for j in range(Grid.shape[0]):
            GridIndexes[j][i]=np.sqrt((i-im)**2+(j-jm)**2)*dx
    s=0
    dr=0.001*dx
    stot=Grid.sum()
    sp=p*stot
    rp=0
    while s<sp:
        rp+=dr
        mask=(GridIndexes<=rp)
        s=Grid[mask].sum()
    return rp

