# -*- coding: utf-8 -*-
# Librerias
import pandas as pd
import numpy as np
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


# Se reordena el vector de endogenas
y_new_train = np.random.permutation(data_train.iloc[:,0])

# ---------------------------------------------------------------------------#
# Se realizan los graficos
fig, axs = plt.subplots(1, 2)

# ---------------------------------------------------------------------------#
# Grafico 1
X_train = np.array(data_train.iloc[:,1:])
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
Y_train = np.array(data_train.iloc[:,0])

# Se estima el beta con los datos del train
beta_alea1 = my_mco(X_train, Y_train)[0]

axs[0].bar(range(0, len(beta_alea1)), beta_alea1)
axs[0].set_xlabel(r'$ \beta $')

# ---------------------------------------------------------------------------#
# Grafico 2
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
    Y_train_rev = np.array(y_new_train)  
      
    # Se prescinde de ttest asociado a la constante
    ttest_train = my_mco(X_train_rev, Y_train_rev)[2][1:]  
    
    t_eval = min(abs(ttest_train))
    if t_eval < t_min_val:
        t_busc = list(abs(ttest_train))
        index_busc = t_busc.index(t_eval)
        names_var.remove(X_sel.iloc[:,index_busc].name)

# Betas del ultimo modelo
beta_alea2 = my_mco(X_train_rev, Y_train_rev)[0]

# se grafica
axs[1].bar(range(0, len(beta_alea2)), beta_alea2)
axs[1].set_xlabel(r'$ \beta $')




    