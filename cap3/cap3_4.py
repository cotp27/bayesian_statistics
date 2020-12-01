# -*- coding: utf-8 -*-
"""
Fig. 3.4. beta prior y posterior 

@author: COTP
"""

#librerias a usar
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

#fijando la semilla
np.random.seed(0)

#función de probabilidad (beta)
def fun1(n,a,b):
    X=ss.beta(a,b)
    X1=X.pdf(n)
    return X1

#parametros
n=np.array([5,100])
Sy=np.array([1,20])

#valores del eje x
x=np.linspace(0,1,1000)

#distribuciones
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.plot(x, fun1(x,1,1), x,fun1(x,1+Sy[0],1+n[0]-Sy[0]))
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel(r'$\theta$')
plt.title('beta(1,1) prior, '+r'$n=5,\sum y_i=1$')
plt.legend(["prior","posterior"])

plt.subplot(2,2,2)
plt.plot(x, fun1(x,3,2),x,fun1(x,1+Sy[0],1+n[0]-Sy[0]))
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel(r'$\theta$')
plt.title('beta(3,2) prior, '+r'$n=5,\sum y_i=1$')
plt.legend(["prior","posterior"])

plt.subplot(2,2,3)
plt.plot(x, fun1(x,1,1),x,fun1(x,1+Sy[1],1+n[1]-Sy[1]))
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel(r'$\theta$')
plt.title('beta(1,1) prior, '+r'$n=100,\sum y_i=20$')
plt.legend(["prior","posterior"])

plt.subplot(2,2,4)
plt.plot(x, fun1(x,3,2),x,fun1(x,1+Sy[1],1+n[1]-Sy[1]))
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel(r'$\theta$')
plt.title('beta(1,1) prior, '+r'$n=100,\sum y_i=20$')
plt.legend(["prior","posterior"])

plt.show()



