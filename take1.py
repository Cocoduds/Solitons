# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:39:57 2022

@author: dudle
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



h = 1
a = 1
stepsizeEuler = 1


def KdeV(x,a,t):
    return 12*a**2*(1/(np.cosh(a*(x-4*(a**2)*t)))**2)

# def wavespeed(x,h,t):
#     return (1/(4*h)) * ((u(x+h, a, t))**2 - (u(x-h, a, t))**2)

# def dispersion(x,h,t):
#     return (1/(2*h**3)) * (u(x+(2*h), a, t) - 2*(u(x+h, a, t) - u(x-h, a, t)) - u(x-(2*h), a, t))

def wavespeed(x, h):
    uiPlus1 = x[1:] + x[:1]
    uiPlus1 = uiPlus1[0]
    uiMinus1 = x[-1:] + x[:-1]
    uiMinus1 = uiMinus1[0]
    difference=[]
    output=[]
    for i in range (0,len(x)):
        print(uiPlus1[1])
        uiPlus1[i] = uiPlus1[i]**2
        uiMinus1[i] = (uiMinus1[i])**2
        difference.append(uiPlus1[i] - uiMinus1[i])
        output.append(1/((4*h)) * difference[i])
    return output

def dispersion(x, h):
    uiPlus1 = x[1:] + x[:1]
    uiMinus1 = x[-1:] + x[:-1]
    uiPlus2 = x[2:] + x[:2]
    uiMinus2 = x[-2:] + x[:-2]
    uiPlus1 = uiPlus1[0]
    uiPlus2 = uiPlus2[0]
    uiMinus1 = uiMinus1[0]
    uiMinus2 = uiMinus2[0]
    difference=[]
    output=[]
    for i in range (0,len(x)):
        difference.append(uiPlus2[i] - 2*uiPlus1[i] + 2*uiMinus1[i] - uiMinus2[i])
        output.append(1/((2*h**3) ) * difference[i])
    return output

def uprime(t, x, h):
    output = []
    for i in range (0,len(x)):
        output.append(-(wavespeed(x, h)[i] + dispersion(x, h)[i]))
    return output

def euler(yn, stepsize, t, h):
    return(yn + (stepsize * uprime(t, yn, h)))

def rungeKutta(xn, tn, stepSize, h): #alpha = 1/2
    k1 = stepSize * uprime(tn, xn, h)
    k2 = stepSize * uprime(tn+stepSize/2, (xn+k1*stepSize)/2, h)
    return(xn + k2 + stepSize**3)



#%% initial condition
x = np.linspace(0,10,100)
uvalue=[[]]
for i in x:
    uvalue[0].append(KdeV(i,a,1))  
plt.plot(x,uvalue[0])
    
#%% 2 euler method part
# yn = 1
# yvals=[yn]
# for i in range (1,100):
#       yn = euler(yn, stepsizeEuler, i, h)
#       yvals.append (yn)
# x = np.linspace(1,100,100)    
# plt.plot(x,yvals)


#%%runge kutta
# yn = 1
# yvals=[yn]
# for i in range (1,100):
#       yn = rungeKutta(yn, i, 1, h)
#       yvals.append (yn)
# x = np.linspace(1,100,100)    
# plt.plot(x,yvals)

# for j in range(1,100):
#     uvalue.append([0])
#     for i in range (1, len(uvalue[0])):
#           yn = rungeKutta(uvalue[j-1][i], 0.01, 0.1, h)
#           uvalue[j].append (yn)
#     plt.plot(x,uvalue[j])

print(rungeKutta(uvalue, 0.01, 0.1, h))

