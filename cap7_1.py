# -*- coding: utf-8 -*-
# Capitulo 7 - Hoff
# Librerias
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

#----------------------------------------------------------------------------#
# Media para distribucion multivariad
mean1 = [50, 50]
cov1 = [[64, -48],[-48, 144]]
cov2 = [[64, 0]  ,[0, 144]]
cov3 = [[64, 48] ,[48, 144]]

y1_1_grid = np.linspace(20, 80, num=3000)
y2_1_grid = np.linspace(20, 80, num=3000)

X, Y = np.meshgrid(y1_1_grid ,y2_1_grid)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# Multivariado normal con covarianza negativa
y1_1, y2_1 = np.random.multivariate_normal(mean1, cov1, 30).T
rv1 = multivariate_normal(mean1, cov1).pdf(pos)

# Multivariado normal sin covarianza  
y1_2, y2_2 = np.random.multivariate_normal(mean1, cov2, 30).T
rv2 = multivariate_normal(mean1, cov2).pdf(pos)

# Multivariado normal con covarianza positiva
y1_3, y2_3 = np.random.multivariate_normal(mean1, cov3, 30).T
rv3 = multivariate_normal(mean1, cov3).pdf(pos)

#----------------------------------------------------------------------------#
plt.subplot(1, 3, 1)
plt.contour(X, Y, rv1)
plt.scatter(y1_1, y2_1)

plt.subplot(1, 3, 2)
plt.contour(X, Y, rv2)
plt.scatter(y1_2, y2_2)

plt.subplot(1, 3, 3)
plt.contour(X, Y, rv3)
plt.scatter(y1_3, y2_3)
