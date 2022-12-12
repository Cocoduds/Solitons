# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 18:01:27 2022

@author: dudle
"""


from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def kdv(x,t,dx):
    up1 = np.hstack([x[1:], x[:1]])
    up2 = np.hstack([x[2:], x[:2]])
    up3 = np.hstack([x[3:], x[:3]])
    up4 = np.hstack([x[4:], x[:4]])
    um1 = np.hstack([x[-1:], x[:-1]])
    um2 = np.hstack([x[-2:], x[:-2]])
    um3 = np.hstack([x[-3:], x[:-3]])
    um4 = np.hstack([x[-4:], x[:-4]])
    
    # O(h^2) Central differences
    #ux1 = (up1 - um1) / (2 * dx)
    #ux3 = (up2 - 2 * up1 + 2 * um1 - um2) / (2 * dx * dx * dx)

    # O(h^4) Central differences
    #ux1 = (-(up2 - um2) + 8 * (up1 - um1)) / (12 * dx)
    #ux3 = (-(up3 - um3) + 8 * (up2 - um2) - 13 * (up1 - um1)) / (8 * dx * dx * dx)
    
    #O(h^6) Central differences
    ux1 = ((up3 - um3) - 9 * (up2 - um2) + 45 * (up1 - um1)) / (60 * dx)
    ux3 = (7 * (up4 - um4) - 72 * (up3 - um3) + 338 * (up2 - um2) - 488 * (up1 - um1)) / (240 * dx * dx * dx)
    
    return -6 * x * ux1 - ux3

def rk4(x, dt, dx):
    k1 = dt * kdv(x, 0, dx)
    k2 = dt * kdv(x + k1 * 0.5, 0, dx)
    k3 = dt * kdv(x + k2 * 0.5, 0, dx)
    k4 = dt * kdv(x + k3, 0, dx)
    return x + 1/6. * (k1 + 2*k2 + 2*k3 + k4)

def kdvExact(x,t,v,x0):
    a = np.cosh(0.5 * np.sqrt(v) * (x - v * t - x0))
    return v / (2 * a * a)

x = np.linspace(0,10,100)
u = kdvExact(x, 0, 16, 4)
plt.plot(x,u)


for i in range (0, 1000):
    # u = rungeKutta(u, 0.001, h)
    u = rk4(u, 0.001, 0.1)
    plt.plot(x,u)
    plt.show()
