# -*- coding: utf-8 -*-
"""
Fig. 3.1. Distribucion prior y posterior

@author: COTP
"""
#librerias a usar
import numpy as np
import matplotlib.pyplot as plt
import math 


#funciones a usar

def fun1(theta, n, Sy):
    prob=theta**Sy*(1-theta)**(n-Sy)
    return prob

def fun2(theta, n, Sy):
    gama=(math.gamma(Sy+1)*math.gamma(n-Sy+1))/math.gamma(n+2)
    prob= theta**Sy*(1-theta)**(n-Sy)/gama
    return prob

#variables
total=129
exitos=118

#distribuciones


plt.figure(figsize=(10,10))


plt.subplot(211)
plt.plot(np.linspace(0,1,1000), fun1(np.linspace(0,1,1000),total,exitos))
plt.title('Distribución prior' )
plt.ylabel(r'$p(y_1,y_2,\cdots,y_{129}|\theta)$')
plt.xlabel(r'$\theta$')

plt.subplot(212)
plt.plot(np.linspace(0,1,1000), fun2(np.linspace(0,1,1000),total,exitos),np.linspace(0,1,1000),np.linspace(1,1,1000))
plt.title('Distribución posterior')
plt.ylabel(r'$p(\theta|y_1,y_2,\cdots,y_{129})$')
plt.xlabel(r'$\theta$')

plt.show()




