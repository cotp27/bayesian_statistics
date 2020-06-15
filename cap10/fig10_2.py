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
dir_trab = r'C:\Users\coyol\Desktop\bayesian_python\py_programs'
data_all = pd.read_csv(dir_trab  + r'\datos\sparrows.csv', sep=',' ,index_col=0)
age = data_all['age'].to_numpy()
fledged = data_all['fledged'].to_numpy()

# ---------------------------------------------------------------------------#
mean_beta = np.array([0, 0, 0])
sd_beta = np.sqrt(np.array([100, 100, 100]))

size_grid = 100
b1_grid = np.linspace(.27-1.75, .27+1.75, size_grid)
b2_grid = np.linspace(.68-1.50, .68+1.5, size_grid)
b3_grid = np.linspace(-.13-.25,-.13+.25, size_grid)

# Se evaluan todos los puntos de los grid
vero_bets = np.zeros([size_grid, size_grid, size_grid]) 
conteo =0

# Se trabaja con las probabilidades en logaritmos por la estabilidad numerica
for i in range(0, size_grid):
    for j in range(0, size_grid):
        for k in range(0, size_grid):
            theta = b1_grid[i] + b2_grid[j]*age + b3_grid[k]*age**2
            vero_bets[i, j, k] = norm.logpdf(b1_grid[i], mean_beta[0], sd_beta[0]) +    \
                                 norm.logpdf(b2_grid[j], mean_beta[1], sd_beta[1]) +    \
                                 norm.logpdf(b3_grid[k], mean_beta[2], sd_beta[2]) +    \
                                 np.sum(poisson.logpmf(fledged,np.exp(theta)))


# Se suma todas las probabilidades                                 
prob_bets = np.exp(vero_bets) / np.sum(np.exp(vero_bets ))

# Se calculan las marginales para cada parametro
marg_beta1  = [ np.sum(prob_bets[i,:,:]) for i in range(0, size_grid)]
marg_beta2  = [ np.sum(prob_bets[:,j,:]) for j in range(0, size_grid)]
marg_beta3  = [ np.sum(prob_bets[:,:,k]) for k in range(0, size_grid)]

# El resultado aqui es una matriz para que sea mas facil la lectura
marg_beta23 = np.zeros([size_grid, size_grid]) 
for j in range(0, size_grid):
    for k in range(0, size_grid):    
        # Se guarda la prob asociada al b2 y b3 especifico
        marg_beta23[j,k] = np.sum(prob_bets[:,j,k])


# ---------------------------------------------------------------------------#    
# Fig. 10.2
# Se presenta las marginales        
plt.plot(b1_grid, marg_beta1)
plt.xlabel(r'$ \beta_1$')

plt.subplot(1, 3, 1)
plt.plot(b2_grid, marg_beta2)
plt.xlabel(r'$ \beta_2$')

plt.subplot(1, 3, 2)
plt.plot(b3_grid, marg_beta3)
plt.xlabel(r'$ \beta_3$')
        
# Se hace el contour
X, Y = np.meshgrid(b2_grid, b3_grid)

# Se suma un valor muy pequenho para q en la grafica no salga una linea en negro
marg_beta23 = marg_beta23 + 9.16675E-234
plt.subplot(1, 3, 3) 
plt.contour(X, Y, marg_beta23)
