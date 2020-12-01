# -*- coding: utf-8 -*-
"""
Fig. 3.5. HPD

@author: COTP
"""

#librerias a usar
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

#parametros
a=1 
b=1
n=10
y=2

#cuantiles
q=ss.beta.ppf([0.025,0.975],3,9, loc=0, scale=1)

#distribucion

plt.figure()
plt.plot(np.linspace(0,1,1000),ss.beta(a+y,b+n-y).pdf(np.linspace(0,1,1000)))
plt.axvline(q[0],c='r')
plt.axvline(q[1],c='r')
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel(r'$\theta$')


