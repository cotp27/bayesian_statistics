# -*- coding: utf-8 -*-
"""
Fig. 3.2. y 3.3. Distribucion binomial 

@author: COTP
"""
#librerias a usar
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

#fijando la semilla
np.random.seed(0)

#variables
n=np.array([10,100])
teta=np.array([0.2,0.8])

#función masa de probabilidad (binomial)
def fun1(teta,n):
    X=ss.binom(n,teta)
    X1=X.pmf(np.arange(n+1))
    return X1

#distribuciones n=10

plt.figure(figsize=(11,5))

plt.subplot(1,2,1)
plt.bar(np.arange(n[0]+1),fun1(teta[0],n[0]))
plt.ylabel(r'$p(Y=y|\theta=0.2,n=10)$')
plt.xlabel(r'$y$')


plt.subplot(1,2,2)
plt.bar(np.arange(n[0]+1),fun1(teta[1],n[0]))
plt.ylabel(r'$p(Y=y|\theta=0.8,n=10)$')
plt.xlabel(r'$y$')

plt.show()

#distribuciones n=100

plt.figure(figsize=(11,5))

plt.subplot(1,2,1)
plt.bar(np.arange(n[1]+1),fun1(teta[0],n[1]))
plt.ylabel(r'$p(Y=y|\theta=0.2,n=100)$')
plt.xlabel(r'$y$')


plt.subplot(1,2,2)
plt.bar(np.arange(n[1]+1),fun1(teta[1],n[1]))
plt.ylabel(r'$p(Y=y|\theta=0.8,n=100)$')
plt.xlabel(r'$y$')

plt.show()


