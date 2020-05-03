# -*- coding: utf-8 -*-
# Librerias
import pandas as pd
import numpy as np
import math
from scipy.special import gamma
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Semilla
np.random.seed(50)

# Datos
dir_trab = r'C:\Users\coyol\Desktop\bayesian_python\py_programs'
data_train = pd.read_csv(dir_trab  + r'\datos\diabetes_train.csv', sep=',' ,index_col=0)
data_test  = pd.read_csv(dir_trab  + r'\datos\diabetes_test.csv', sep=',' ,index_col=0)

# Funcion para estimar Beta por MCO
def my_mco(X,Y):
    inv_XTX = inv(np.dot(X.T,X))
    beta_ols = np.dot(np.dot(inv_XTX, X.T), Y)
    resid_ols = Y - np.dot(X, beta_ols)
    SSR_ols = np.dot(resid_ols.T, resid_ols)
    n = X.shape[0]
    p = X.shape[1]
    sigma2_0 = SSR_ols/(n-p)
    t_est = beta_ols / ((sigma2_0 * np.diag(inv_XTX))**(0.5))
    return beta_ols, sigma2_0, t_est

# Funcion pr(y|X,z) en logaritmo
def ln_pr_y_Xz(nu0, n, g, sigma2_0, X, Y):
    pz = X.shape[1]
    inv_XTX = inv(np.dot(X.T,X))
    X_inv_XTX_XT_g = (g/(1+g)) * np.dot(np.dot(X, inv_XTX), X.T)
    SSR_g = np.dot(np.dot(Y.T, (np.identity(n) - X_inv_XTX_XT_g)), Y)
    # Se calcula la verosimilitud de los datos dado el modelo
    ln_pr = -0.5*( n*np.log(2*np.pi) + pz*np.log(1+g) + (nu0+n)*np.log(.5*(nu0*sigma2_0 + SSR_g)) - nu0*np.log(.5*nu0*sigma2_0) ) + np.log(gamma((nu0+n)/2)) - np.log(gamma(nu0/2))
    #ln_pr = -0.5*(n*np.log(2*np.pi)) + np.log(gamma(.5*(nu0+n))) - np.log(gamma(.5*nu0)) - .5*pz*np.log(1+g) + .5*nu0*np.log(.5*nu0*sigma2_0) - 0.5*(nu0+n)*np.log(.5*(nu0*sigma2_0 + SSR_g))
    return ln_pr

# se ajusta la funcion de numpy para q la funcion gamma sea como hoff
def gamma_inv_hoff(a,b, samps=1):
    b_inv = b**(-1)
    result = 1 / np.random.gamma(a, b_inv, samps)
    return result


# ---------------------------------------------------------------------------#
# Se realiza las simulaciones
X_orig = data_train.iloc[:,1:].copy()
X_orig.insert(0, 'beta1', 1)    
Y_orig = data_train.iloc[:,0].copy()   

# Se corre el primer modelo
X_train = np.array(X_orig)
Y_train = np.array(data_train.iloc[:,0])

X_test = np.array(data_test.iloc[:,1:])
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
Y_test = np.array(data_test.iloc[:,0])

# vector de z = 1
z_vec = np.ones(X_orig.shape[1])         

# Parametros prior
nu0 = 1
n = Y_orig.shape[0]
g = n
sigma2_0 = my_mco(X_train, Y_train)[1]
ln_pr = ln_pr_y_Xz(nu0, n, g, sigma2_0, X_train, Y_train)

# Se crea diccionario para guardar los valores de beta
dict_betas = {ih:[] for ih in X_orig.columns}   

# Se crea lista para guardar las simulaciones de los vectores Z
sims_z = [z_vec.tolist()]

# Simulaciones
sims = 10000
for ix in range(0, sims):
    # Se construye vector con el orden con el cual se evalua la simulacion
    reorder = np.random.permutation(range(0,X_orig.shape[1]))
    for jx in reorder:
        # Paso 1: Se guada el vector original
        z_orig = z_vec.copy()
        
        # Paso 2: Se cambia el jx componente del vector
        z_camb = z_vec.copy()        
        z_camb[jx] = 1 - z_camb[jx]

        # Paso 3: Se calcula la verosimilitud con el nuevo vector 0->1 o 1->0
        z_bool = [True if i==1 else False for i in z_camb] # Se convierte el vecto en bool
        X_cond = np.array(X_orig.loc[:,z_bool])
        ln_new = ln_pr_y_Xz(nu0, n, g, sigma2_0, X_cond, Y_train)         
        
        # Paso 4: Se calcula prob de nuevo vector(logit = log(pi/(1-pi)) = log(pi) - log(1-pi))
        if z_orig[jx] == 1:         # Si el elemento original del vector es 1 
            var_prob = ln_pr - ln_new
        else:                       # Si el elemento original del vector es 0
            var_prob = ln_new - ln_pr  
        
        # Para recuperar la probabilidad del logit se hace exp(x)/(1+exp(x)) = 1/(1+exp(-x))
        prob = 1 / (1+math.exp(-var_prob))
        
        # Paso 5: Se ejecuta el draw para simular el nuevo zi
        z_vec[jx] = np.random.binomial(1, prob)
        
        # Paso 6: Se actualiza prob en caso draw cambie el valor 0->1 o 1->0
        if z_orig[jx] != z_vec[jx]:
            ln_pr = ln_new

    # Se guarda el vector z simulado
    sims_z.append(z_vec.tolist())

    # Una vez que tengo el nuevo vector z genero names y beta
    # Paso 7: Se genera un sigma2 dado el nuevo z
    z_asoc = [True if i==1 else False for i in z_vec]        
    X_asoc = np.array(X_orig.loc[:,z_asoc])
    inv_XTX_asoc = inv(np.dot(X_asoc.T,X_asoc))
    X_inv_XTX_XT_g_asoc = (g/(1+g)) * np.dot(np.dot(X_asoc, inv_XTX_asoc), X_asoc.T)
    SSR_g = np.dot(np.dot(Y_train.T, (np.identity(n) - X_inv_XTX_XT_g_asoc)), Y_train)    

    # Simulacion
    sigm2_samp = gamma_inv_hoff((nu0+n)/2, (nu0*sigma2_0+SSR_g)/2)    

    # Paso 8: Se hace la simulacion para el nuevo beta
    beta_ols = my_mco(X_asoc, Y_train)[0]        
    mv_samp = np.random.multivariate_normal(g/(1+g) * beta_ols, g*sigm2_samp/(1+g) * inv_XTX_asoc)
        
    # Se guardan los betas en su dicionario
    counter1 = 0
    for bet, names in zip(z_vec, dict_betas):
        if bet == 1:
            list_check = dict_betas[names]
            list_check.append(mv_samp[counter1])
            dict_betas[names] = list_check
            counter1 = counter1 + 1
        else:
            list_check = dict_betas[names]
            list_check.append(0)
            dict_betas[names] = list_check
       

# ---------------------------------------------------------------------------#
# Se realizan los graficos
fig, axs = plt.subplots(1, 2)

# ---------------------------------------------------------------------------#
# Grafico 1
# Lista de simulaciones en matriz
sims_z_matrix = np.reshape(np.ravel(sims_z), (len(sims_z), -1))
axs[0].bar(range(0, sims_z_matrix.shape[1]), np.mean(sims_z_matrix, axis=0))
axs[0].set_xlabel(r'$ \beta $')


# ---------------------------------------------------------------------------#
# Grafico 2
betas_by = pd.DataFrame.from_dict(dict_betas).mean(axis=0).values
Y_pred_elim = np.dot(X_test, betas_by)
print(np.mean((Y_test-Y_pred_elim)**2))
axs[1].scatter(Y_test, Y_pred_elim)
axs[1].plot(axs[1].get_ylim(), axs[1].get_ylim(), color='k', linestyle='--')
axs[1].set_xlabel(r'$ y_{test} $')


