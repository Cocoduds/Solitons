# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:09:09 2022

@author: dudle
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
h = 0.2
a = 1
stepsizeEuler = 1



def KdeV(x,a,t):
    return 12*a**2*(1/(np.cosh(a*(x-4*(a**2)*t)))**2)


def rk4(x, dt, dx):   #REWRITE THIS
    k1 = dt * uprime(x, dx)
    k2 = dt * uprime(x + k1 * 0.5, dx)
    k3 = dt * uprime(x + k2 * 0.5, dx)
    k4 = dt * uprime(x + k3, dx)
    return x + 1/6. * (k1 + 2*k2 + 2*k3 + k4)

def wavespeed(x, h):
    uiPlus1 = np.append(x[1:],x[:1])
    uiMinus1 = np.append(x[-1:],x[:-1])
    return (1/(4*h)) * (uiPlus1**2 - uiMinus1**2)
    # return x * (1/(2*h)) * (uiPlus1 - uiMinus1)

def dispersion(x, h):
    uiPlus1 = np.append(x[1:],x[:1])
    uiMinus1 = np.append(x[-1:],x[:-1])
    uiPlus2 = np.append(x[2:],x[:2])
    uiMinus2 = np.append(x[-2:],x[:-2])
    return 1/((2*h**3)) * (uiPlus2 - 2*uiPlus1 + 2*uiMinus1 - uiMinus2)

def uprime(x, h):
    return -(wavespeed(x,h) + dispersion(x,h))

def rungeKutta(xn, stepSize, h): #alpha = 1/2
    k1 = stepSize * uprime(xn, h)
    k2 = stepSize * uprime(xn+k1/2, h)
    return(xn + k2 + stepSize**3)

#%% initial condition
x = np.linspace(0,20,100)
u = KdeV(x, a, 1) + KdeV(x, 0.8, 5)
plt.plot(x,u)



for i in range (0, 100000):
    # u = rungeKutta(u, 0.001, h)
    u = rk4(u, 0.0001, h)
    if i%1000 == 0:
        plt.plot(x,u)
        plt.show()