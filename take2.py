# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:09:09 2022

@author: dudle
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
h = 0.1
a = 1
stepsizeEuler = 1

def KdeV(x,a,t):
    return 12*a**2*(1/(np.cosh(a*(x-4*(a**2)*t)))**2)

def RK2(x,dt,dx):
    a = 1/2
    fa = uprime(x,dx)
    fb = uprime(x + a*dt*fa, dx)
    return x + 1/(2*a) * ((2*a - 1)*fa + fb) * dt

def RK4(x, dt, dx):
    fa = uprime(x, dx)
    fb = uprime(x + fa*dt/2, dx)
    fc = uprime(x + fb*dt/2, dx)
    fd = uprime(x + fc*dt, dx)
    return x + 1/6 *(fa + 2*fb + 2*fc + fd) * dt
    
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

#%% Dynamics plots
# plt.figure(1)
# h=0.1
# xmin=-5
# xmax=10
# x = np.linspace(int(xmin),int(xmax),int((xmax-xmin)/h))
# u = KdeV(x, 1, 0)
# plt.plot(x,u)
# dt=0.001
# maxes = []
# times=[]
# for i in range (0, 2000):
#     u = RK4(u, dt, h)
#     maxes.append(np.max(u))
#     times.append(i*dt)
#     if i%600 == 0:
#         plt.plot(x,u, label = 't = '+str(i*0.001))
# plt.legend()
# plt.xlabel('Distance') 
# plt.ylabel('Amplitude')   
# plt.title('The Dynamics of a Single Soliton')     
# plt.show()
       


#%% heights
# plt.figure(2)
# plt.plot(times,maxes)
# plt.ylim([0,15])
# plt.xlabel('Time')
# plt.ylabel('Amplitude')   
# plt.title('Ampitude of a Single Soliton Over 2 Time Units')         
# plt.show()

#%% Speed vs height
h=0.1
xmin=-5
xmax=10
x = np.linspace(int(xmin),int(xmax),int((xmax-xmin)/h))
dt=0.001
maxes=[]
speeds=[]
colors = ['blue', 'orange', 'red', 'green', 'purple']
for i in [0.5, 0.7, 0.9 ,1.1 , 1.3]:
    u=KdeV(x,i,0)
    plt.plot(x,u,  label = 'α = '+str(i), color=colors[int((i-0.4)*5)])
    for j in range (0, 1000):
        u = RK4(u, dt, h)
    maxes.append(np.max(u))
    speed = (np.where(np.isclose(u, np.max(u))))
    speeds.append(speed[0][0]/10-5)
    plt.plot(x,u, linestyle='dashed', color=colors[int((i-0.4)*5)])
plt.xlabel('Distance')
plt.ylabel('Displacement')   
plt.title('The Varying Speeds of Different Size Solitons')     
plt.legend()

plt.show()

maxestheory=[]
speedstheory=[]
for i in np.arange(0.1,1.5,0.2):
    u=KdeV(x,i,0)
    plt.plot(x,u, label = str(2*i/10))
    u=KdeV(x,i,1)
    maxestheory.append(np.max(u))
    speed = (np.where(np.isclose(u, np.max(u))))
    speedstheory.append(speed[0][0]/10-5)
    plt.plot(x,u, label = str(j*0.001), linestyle='dashed')


plt.figure(4)

plt.plot(maxestheory,speedstheory, linestyle='dashed', label = 'Theoretical relation')    
plt.scatter(maxes,speeds, label = 'Numerical values')
plt.xlabel('Amplitude')
plt.ylabel('Speed (Arbitrary Units)')   
plt.title('Speed of Solitons in Respect to Amplitude')     
plt.legend()

#%% Collisions plots
# h=0.1
# xmin=0
# xmax=40
# x = np.linspace(int(xmin),int(xmax),int((xmax-xmin)/h))
# u = KdeV(x, a, 1) + KdeV(x, 0.8, 4)
# plt.plot(x,u)

# for i in range (0, 9000):
#     u = RK4(u, 0.001, h)
#     if i%3000 == 0:
#         plt.plot(x,u)
# plt.show()
        

#%% Wave Breaking plots
# def sin(x,t,a):
#     y=np.array([])
#     for i in range(len(x)):       
#         if 0 < x[i] and x[i] < np.pi: 
#             y = np.append(y, a*np.sin(x[i]+t))
#         else:
#             y = np.append(y, 0)
#         print(y)
#     return(y)
    
# x = np.linspace(0,np.pi,100)
# u = sin(x,0,1)
# plt.plot(x,u)

# for i in range (0, 20000):
#     u = RK4(u, 0.001, h)
#     if i%5000 == 0:
#         plt.plot(x,u)
# plt.show()
        
#%% Shock plots
# def Shockwave(x, h):
#     return -(wavespeed(x,h))

# def RK4Shock(x, dt, dx):
#     fa = Shockwave(x, dx)
#     fb = Shockwave(x + fa*dt/2, dx)
#     fc = Shockwave(x + fb*dt/2, dx)
#     fd = Shockwave(x + fc*dt, dx)
#     return x + 1/6 *(fa + 2*fb + 2*fc + fd) * dt

# def Diffusion(x, h, D):
#     uiPlus1 = np.append(x[1:],x[:1])
#     uiMinus1 = np.append(x[-1:],x[:-1])
#     ui = x
#     return D * (uiPlus1 - 2*ui + uiMinus1)/h**2

# def ShockDiffusion(x, h, D):
#     return -(wavespeed(x,h) - Diffusion(x,h, D))

# def RK4ShockDiffusion(x, dt, dx, D):
#     fa = ShockDiffusion(x, dx, D)
#     fb = ShockDiffusion(x + fa*dt/2, dx, D)
#     fc = ShockDiffusion(x + fb*dt/2, dx, D)
#     fd = ShockDiffusion(x + fc*dt, dx, D)
#     return x + 1/6 *(fa + 2*fb + 2*fc + fd) * dt

# x = np.linspace(0,10,1000)
# u = KdeV(x, a, 1)
# plt.plot(x,u)
# D = 1
# for i in range (0, 9000):
#     u = RK4ShockDiffusion(u, 0.001, h, D)
#     if i%1500 == 0:
#         plt.plot(x,u)
# plt.show()
        
