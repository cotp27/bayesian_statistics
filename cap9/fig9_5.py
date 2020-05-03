# -*- coding: utf-8 -*-
# Librerias
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Semilla
np.random.seed(10)

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

# Matriz de informacion - Forma matricial    
X_train = np.array(data_train.iloc[:,1:])
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
Y_train = np.array(data_train.iloc[:,0])

X_test = np.array(data_test.iloc[:,1:])
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
Y_test = np.array(data_test.iloc[:,0])

# Se estima el beta con los datos del train
beta_train = my_mco(X_train, Y_train)[0]

# Se hace la prediccion con la bd test
Y_test_pred = np.dot(X_test, beta_train)

# Error cuadratico medio
print(np.mean((Y_test-Y_test_pred)**2))
print(np.mean(Y_test**2))

# ---------------------------------------------------------------------------#
# Se realizan los graficos
fig, axs = plt.subplots(1, 3)

# Grafico 1
axs[0].scatter(Y_test, Y_test_pred)
axs[0].plot(axs[0].get_ylim(), axs[0].get_ylim(), color='k', linestyle='--')
axs[0].set_xlabel(r'$ y_{test} $')
axs[0].set_ylabel(r'$ \hat{y}_{test} $')

# ---------------------------------------------------------------------------#
# Grafico 2
axs[1].bar(range(0, len(beta_train)), beta_train)
axs[1].set_xlabel(r'$ \beta $')

# ---------------------------------------------------------------------------#
# Grafico 3
# Se extraen variables que no sean relevantes
t_min_val = 1.65
t_eval = 0
names_var = list(data_train.columns)[1:]

# eliminacion iterativa
while t_eval < t_min_val:

    # Se elimina las variables que tienen un t-est menor q 1.65
    X_sel = data_train.loc[:,names_var].copy()
    X_train_rev = np.array(X_sel)
    X_train_rev = np.c_[np.ones(X_train_rev.shape[0]), X_train_rev]
    Y_train_rev = np.array(data_train.iloc[:,0])  
      
    # Se prescinde de ttest asociado a la constante
    ttest_train = my_mco(X_train_rev, Y_train_rev)[2][1:]  
    
    t_eval = min(abs(ttest_train))
    if t_eval < t_min_val:
        t_busc = list(abs(ttest_train))
        index_busc = t_busc.index(t_eval)
        names_var.remove(X_sel.iloc[:,index_busc].name)

# Betas del ultimo modelo
betas_fin = my_mco(X_train_rev, Y_train_rev)[0]

# Se predice con el modelo anidado
X_elim = np.array(data_test.loc[:,names_var])
X_elim = np.c_[np.ones(X_elim.shape[0]), X_elim]
Y_pred_elim = np.dot(X_elim, betas_fin)

# Error cuadratico medio
print(np.mean((Y_test-Y_pred_elim)**2))        

# se grafica
axs[2].scatter(Y_test, Y_pred_elim)
axs[2].plot(axs[2].get_ylim(), axs[2].get_ylim(), color='k', linestyle='--')
axs[2].set_xlabel(r'$ y_{test} $')

    