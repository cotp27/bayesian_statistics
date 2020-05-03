# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------#
# librerias 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------------------------------#
# Directorio de trabajo
dir_trab = r'C:\Users\coyol\OneDrive\Escritorio\bayesian_python\py_programs'
data_full = pd.read_csv(dir_trab  + r'\datos\pima_full.csv', sep=',' ,index_col=0)

#----------------------------------------------------------------------------# 
# with regression
sns.pairplot(data_full, kind="reg")
plt.show()

 
