# -*- coding: utf-8 -*-
# Librerias
import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Semilla
np.random.seed(50)

# Datos
data_y = [9.37, 10.18, 9.16, 11.60, 10.33]

# Parametros prior
sigma2= 1
tao2 = 10
mu = 5
n = len(data_y)

# Parametros posterior
mu_n = np.mean(data_y) * ( (n/sigma2) / (n/sigma2 + 1/tao2)) + \
       mu * ( (1/tao2) / (n/sigma2 + 1/tao2) )

tao2_n = 1 / (n/sigma2 + 1/tao2)       

# Se utiliza el algoritmo de Metropolis para realizar el muestreo, sin necesidad de utilizar
# el posterior
# Parametros
theta  = 0          # Empezamos con theta = 0
delta2 = 2

# Lista para guardar los thetas generados
thetas_sim = [theta]

# Se realiza la simulacion
samps = 10000
for i in range(1, samps):
    # En este ejemplo se utiliza la distribucion normal para el draw de cada parametro
    # Se toma una muestra de theta
    theta_samp = np.sqrt(delta2) * np.random.randn() + theta
    
    # Se calcula la probabilidad conjunta para el theta draw. 
    #Se trabaja con la varianza del modelo y precision del parametro prior
    # Bajo el nuevo theta (se trabaja con algoritmos por estabilidad numerica)
    prob_new_theta = np.sum(norm.logpdf(data_y, theta_samp, np.sqrt(sigma2))) + \
                     norm.logpdf(theta_samp, mu, np.sqrt(tao2)) 
    
    # Bajo el viejo theta (se trabaja con algoritmos por estabilidad numerica)
    prob_ant_theta = np.sum(norm.logpdf(data_y, theta, np.sqrt(sigma2))) + \
                     norm.logpdf(theta, mu, np.sqrt(tao2))
    
    # Como estan en logaritmos, la division para el "r" se convierte en resta
    log_r = prob_new_theta - prob_ant_theta
        
    # Se elige el nuevo theta con probabilidad r
    if np.log(np.random.rand()) < log_r:
        thetas_sim.append(theta_samp)
        theta = theta_samp
    else:
        thetas_sim.append(theta)                     
# ---------------------------------------------------------------------------#    
# Fig. 10.2
# Se ven la convergencia en las iteraciones        
plt.subplot(1, 3, 1)
plt.plot(range(0, samps), thetas_sim)
plt.xlabel(r'iteraciones')

# Histograma de simulaciones
plt.subplot(1, 3, 2)
plt.hist(thetas_sim[100:], 30, normed=1) # Tomo desde la iteracion 100
plt.xlabel(r'$ \theta $ MH')

# Se analiza la distribucion obtenida con la posterior analitica
plt.subplot(1, 3, 3)
domi = np.linspace(7,13,100)
plt.plot(domi, norm.pdf(domi, mu_n, np.sqrt(tao2_n)) )
plt.xlabel(r'$ \theta $ analitica')       