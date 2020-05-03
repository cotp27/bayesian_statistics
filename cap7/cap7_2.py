# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------#
# librerias 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import invwishart
from scipy.stats import wishart, multivariate_normal
from numpy.linalg import inv

#----------------------------------------------------------------------------#
# Semilla para simulaciones
seed_rep = 10

#----------------------------------------------------------------------------#
# Directorio de trabajo
dir_trab = r'C:\Users\coyol\OneDrive\Escritorio\bayesian_python\py_programs'
data = pd.read_csv(dir_trab  + r'\datos\reading.csv', sep=',' ,index_col=0)

# Parametros Prior
mu0 = [50, 50]
Lamb0 = np.array([[625, 312.5],[312.5,625]])     # Corr=0.5

nu0 = 4  # p + 2 = nu0, p= numero de variables. No tan centrado en S0
S0 = np.array([[625, 312.5],[312.5,625]])

# Medias, varianzas y covarianzas muestrales
Medias = data.mean().to_numpy()
Sigma = data.cov().to_numpy()

# Numero datos
n = len(data)

# Arreglo para guardar simulaciones
param = []
data_pred = []

# Se realiza el Gibbs Sampling
np.random.seed(seed=seed_rep)

#----------------------------------------------------------------------------#
# Se realizan 5000 simulaciones
#for i in range(0,5000):
for i in range(0,5000):
    # Se genera un draw para las medias (theta)
    lambn = inv(inv(Lamb0) + n*inv(Sigma))
    mun = lambn.dot(inv(Lamb0).dot(mu0)  + n*inv(Sigma).dot(Medias))
    draw_theta = np.random.multivariate_normal(mun, lambn)
    
    # Se genera draw para la varianza -cov
    rest_mean = (data - draw_theta)
    Sn = S0 + np.dot(rest_mean.T, rest_mean)
    Sigma = inv(wishart.rvs(nu0+n, inv(Sn) , size=1))
    
    # Se genera los las predicciones para y1, y2
    draw_ys = np.random.multivariate_normal(mun, Sigma)
    data_pred.append(draw_ys) 
    
    # Se guardan los resultados
    param.append([draw_theta, Sigma])
    
#----------------------------------------------------------------------------#
# Analisis estadisticos y graficos
rest_med = [i[0][1] - i[0][0] for i in param]
print(np.quantile(rest_med, [0.025, 0.5, 0.975]))    
print(np.mean([1 if i > 0 else 0 for i in rest_med]))    

#----------------------------------------------------------------------------#
# Analisis graficos - Primer grafico
y1_1_grid = np.linspace(38, 62, num=1000)
y2_1_grid = np.linspace(38, 68, num=1000)

X, Y = np.meshgrid(y1_1_grid ,y2_1_grid)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

thetas1 = [i[0][0] for i in param]
thetas2 = [i[0][1] for i in param]
cov1 = np.cov(thetas1, thetas2)

rv = multivariate_normal([np.mean(thetas1), np.mean(thetas2)], cov1).pdf(pos)
plt.contour(X, Y, rv)
plt.plot(y1_1_grid, y2_1_grid)

# Segundo grafico
y1_2_grid = np.linspace(0, 100, num=1000)
y2_2_grid = np.linspace(0, 100, num=1000)

X2, Y2 = np.meshgrid(y1_2_grid ,y2_2_grid)
pos2 = np.empty(X2.shape + (2,))
pos2[:, :, 0] = X2
pos2[:, :, 1] = Y2

ys1 = [i[0] for i in data_pred]
ys2 = [i[1] for i in data_pred]
cov2 = np.cov(ys1, ys2)

rv2 = multivariate_normal([np.mean(ys1), np.mean(ys2)], cov2).pdf(pos2)
plt.contour(X2, Y2, rv2)
plt.plot(y1_2_grid, y2_2_grid)

rest_med2 = [i[1] - i[0] for i in data_pred]
print(np.mean([1 if i > 0 else 0 for i in rest_med2]))    
