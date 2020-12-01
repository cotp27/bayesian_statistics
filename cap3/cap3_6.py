# -*- coding: utf-8 -*-
"""
Fig. 3.6. HPD

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

#HDP

x=np.linspace(0,1,1000)

#hallando el (x,y) maximo de la distribución
y_max=np.max(ss.beta.pdf(x,3,9))
pos_x=np.argmax(ss.beta.pdf(x,3,9))
x_max=x[pos_x]

f1=0
f2=0
f3=0

for i in np.arange(y_max,0,-0.01):
    theta1=x_max
    theta2=x_max
    
    # Primero se busca la parte creciente
    for jx in np.arange(x_max,0,-0.001):
        
        if abs(ss.beta.pdf(jx,a+y,b+n-y)-i)<0.01:
            
            if abs(ss.beta.pdf(theta1,a+y,b+n-y)-i) >= abs(ss.beta.pdf(jx,a+y,b+n-y)-i):            
                theta1=jx
                
    
    # segundo se busca la parte decreciente
    for jy in np.arange(x_max,1,0.001):
        
        if abs(ss.beta.pdf(jy,a+y,b+n-y)-i)<0.01:
            
            if abs(ss.beta.pdf(theta2,a+y,b+n-y)-i) >= abs(ss.beta.pdf(jy,a+y,b+n-y)-i):            
                theta2=jy
    
    print(abs((ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y))-0.5))
    if abs((ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y))-0.50) <= 0.01:
        int_50=np.array([theta1,theta2,abs(ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y)-0.50)])
        f1=f1+1
    
    if abs((ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y))-0.75) <= 0.01:
        int_75=np.array([theta1,theta2,abs(ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y)-0.75)])
        f2=f2+1
        
    if abs((ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y))-0.95) <= 0.01:
        int_95=np.array([theta1,theta2,abs(ss.beta.cdf(theta2,a+y,b+n-y)-ss.beta.cdf(theta1,a+y,b+n-y)-0.95)])
        f3=f3+1
          





