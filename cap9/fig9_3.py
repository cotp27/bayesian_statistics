# -*- coding: utf-8 -*-
# Libreria
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

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
# Se realizan los graficos
fig, axs = plt.subplots(1, 3)

# Grafico 1
pos_bet2 = 1
beta2 = [[i[pos_bet2]] for i in draws_beta_sigma[:,0]]
kde = KernelDensity(kernel='gaussian', bandwidth=7).fit(beta2)
domi2 = np.linspace(-100,100,200)[:, np.newaxis]   
log_dens2 = kde.score_samples(domi2)
axs[0].plot(domi2[:,0], np.exp(log_dens2))
axs[0].plot(domi2[:,0], norm.pdf(domi2[:,0], beta_0[pos_bet2], np.sqrt(Sigma_0[pos_bet2, pos_bet2])))
axs[0].axvline(x=0, color='k', linestyle='--')
axs[0].set_xlabel(r'$ \beta_{2} $')


# Grafico 2
pos_bet4 = 3
beta4 = [[i[pos_bet4]] for i in draws_beta_sigma[:,0]]
kde = KernelDensity(kernel='gaussian').fit(beta4)
domi4 = np.linspace(-4,4,200)[:, np.newaxis]   
log_dens4 = kde.score_samples(domi4)
axs[1].plot(domi4[:,0], np.exp(log_dens4))
axs[1].plot(domi4[:,0], norm.pdf(domi4[:,0], beta_0[pos_bet4], np.sqrt(Sigma_0[pos_bet4, pos_bet4])))
axs[1].axvline(x=0, color='k', linestyle='--')
axs[1].set_xlabel(r'$ \beta_{4} $')


# Grafico 3
# Kernel muktivariado
beta2_mod = [i[0] for i in beta2]
beta4_mod = [i[0] for i in beta4]
axs[2].scatter(beta2_mod, beta4_mod)

