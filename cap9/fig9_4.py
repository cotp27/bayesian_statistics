# -*- coding: utf-8 -*-
# Libreria
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Semilla
np.random.seed(10)

# Datos
dict_dat = {}
dict_dat['edad'] = [23, 22, 22, 25, 27, 20, 31, 23, 27, 28, 22, 24]

dict_dat['capac_pulm'] = [-0.87, -10.74, -3.27, -1.97, 7.50, 
              -7.25, 17.05, 4.96, 10.40, 11.05, 0.26,2.51]

dict_dat['tip_ejer'] = ['running']*6 + ['aerobic']*6

oxi_df = pd.DataFrame(dict_dat)

# Matriz de informacion
x1 = np.array([1]* len(dict_dat['edad'])) 
x2 = np.array([0 if i=='running' else 1 for i in dict_dat['tip_ejer']])
x3 = np.array(dict_dat['edad'])
x4 = np.multiply(x2,x3)

# Forma matricial
X = np.array([x1, x2, x3, x4]).T
Y = np.array(dict_dat['capac_pulm'])

# Parametros prior
p = X.shape[1]
n = len(Y)
g = n
nu_0 = 1

inv_XTX = inv(np.dot(X.T,X))
beta_ols = np.dot(np.dot(inv_XTX, X.T), Y)
resid_ols = Y - np.dot(X, beta_ols)
SSR_ols = np.dot(resid_ols.T, resid_ols)
sigma2_0 = SSR_ols/(n-p)

# Prior para beta
beta_0  = [0, 0, 0, 0]
Sigma_0 = g *sigma2_0 * inv(np.dot(X.T,X))

X_inv_XTX_XT_g = (g/(1+g)) * np.dot(np.dot(X, inv_XTX), X.T)
SSR_g = np.dot(np.dot(Y.T, (np.identity(n) - X_inv_XTX_XT_g)), Y)

# Se realiza  la simulacion - Recordar que es Montecarlo
# se ajusta la funcion de numpy para q la funcion gamma sea como hoff
def gamma_inv_hoff(a,b, samps=1):
    b_inv = b**(-1)
    result = 1 / np.random.gamma(a, b_inv, samps)
    return result

# Simulaciones
sims = 2000
gam_samp = gamma_inv_hoff((nu_0+n)/2, (nu_0*sigma2_0+SSR_g)/2, sims)    

draws_beta_sigma = []
for i in range(0, sims):    
    mv_samp = np.random.multivariate_normal(g/(1+g) * beta_ols, g*gam_samp[i]/(1+g) * inv_XTX)
    draws_beta_sigma.append([mv_samp, gam_samp[i]])

draws_beta_sigma = np.array(draws_beta_sigma)

#----------------------------------------------------------------------------#
# Se predice la diferencia entre los que hacen ejercicio y no lo hacen para cada anho
pos_bet2 = 1
beta2 = np.array([i[pos_bet2] for i in draws_beta_sigma[:,0]])

pos_bet4 = 3
beta4 = np.array([i[pos_bet4] for i in draws_beta_sigma[:,0]])

edad_ci = []
edades = range(20,32)
for i in edades:
    pred_ed = beta2 + i * beta4
    edad_ci.append(pred_ed)
    
# Se agrega la linea horizontal en cero
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
    
# Se realiza la grafica    
ax.boxplot(edad_ci)
ax.set_xticklabels(edades)
ax.axhline(0)
